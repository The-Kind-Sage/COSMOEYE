import os
import json
import h5py
import numpy as np
import torch

from model import CustomUNet
from dataset import normalize_standardized_image, stack_temporal_stacks, NORM_STATS_PATH
from spatial_math import extract_spatial_metrics


def build_standardized_input(raw_matrix):
    """Convert any supported input layout into the standardized temporal-stack layout.

    - 13 channels (raw Landslide4Sense): bands map 1:1 into channels 0-12,
      NDVI is computed from band 3 (Red) and band 7 (NIR) into channel 13.
    - 14 channels (legacy single-frame SEN12 tiles): used as-is.
    - 4D (N, H, W, 14) (temporal-stack tiles): early/mid/late windows, as-is.
    Single-frame inputs become a one-window container so the temporal
    network can still score them.
    """
    if raw_matrix.ndim == 4:
        return raw_matrix.astype(np.float32)

    height, width, channels = raw_matrix.shape

    if channels == 13:
        fused_matrix = np.zeros((height, width, 14), dtype=np.float32)
        fused_matrix[:, :, :13] = raw_matrix
        red = raw_matrix[:, :, 3]
        nir = raw_matrix[:, :, 7]
        computed_ndvi = (nir - red) / (nir + red + 1e-8)
        fused_matrix[:, :, 13] = computed_ndvi
    elif channels == 14:
        fused_matrix = raw_matrix
    else:
        raise ValueError(f"Unsupported channel count: {channels} (expected 13, 14 or a temporal stack)")

    return fused_matrix[None]


def load_normalization_stats():
    """Load the dataset-global z-score stats used during training."""
    if os.path.exists(NORM_STATS_PATH):
        with np.load(NORM_STATS_PATH) as d:
            return d["mean"].astype(np.float32), d["std"].astype(np.float32)
    return None, None


def tta_probabilities(model, image_tensor, device):
    """Average sigmoid probabilities over identity + 3 flips."""
    views = [
        (image_tensor, ()),
        (torch.flip(image_tensor, dims=[3]), (3,)),
        (torch.flip(image_tensor, dims=[2]), (2,)),
        (torch.flip(image_tensor, dims=[2, 3]), (2, 3)),
    ]
    probs = None
    with torch.no_grad():
        for view, flip_dims in views:
            logits = model(view)
            logits = torch.flip(logits, dims=list(flip_dims)) if flip_dims else logits
            prob = torch.sigmoid(logits.float())
            probs = prob if probs is None else probs + prob
    return probs / len(views)


def execute_vision_inference_pass(sample_file_name, csv_path="local_highways.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_img_path = f"./datasets/TestData/img/{sample_file_name}"
    vision_weights = "landslide_unet_weights.pth"

    if not os.path.exists(vision_weights):
        raise FileNotFoundError("Missing local brain weights file! Run train.py first.")

    with h5py.File(test_img_path, "r") as f:
        raw_matrix = np.array(f["img"]).astype(np.float32)

    fused_matrix = build_standardized_input(raw_matrix)

    # CRITICAL: apply the exact same normalization used during training,
    # otherwise the network receives out-of-distribution values.
    norm_mean, norm_std = load_normalization_stats()
    normalized = normalize_standardized_image(fused_matrix, norm_mean, norm_std)

    image_tensor = torch.from_numpy(
        stack_temporal_stacks(normalized)
    ).unsqueeze(0).to(device)

    unet_model = CustomUNet(in_channels=42, out_channels=1)
    unet_model.load_state_dict(torch.load(vision_weights, map_location=device))
    unet_model.to(device).eval()

    probability_mask = tta_probabilities(unet_model, image_tensor, device).squeeze().cpu().numpy()

    # Use the IoU-tuned threshold saved during training (falls back to 0.5)
    threshold = 0.5
    threshold_path = "landslide_best_threshold.json"
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as jf:
            threshold = json.load(jf).get("threshold", 0.5)
    binary_mask = (probability_mask > threshold).astype(np.uint8)

    spatial_metrics = extract_spatial_metrics(binary_mask, csv_path=csv_path)

    print("================== GEOSPATIAL ANALYSIS PASS COMPLETE ==================")
    print(f"Processed Target File: {sample_file_name}")
    print(f"Discovered Hazard Anomalies Records: {spatial_metrics}")
    print("=======================================================================\n")

    return spatial_metrics, binary_mask


if __name__ == "__main__":
    os.makedirs("./datasets/TestData/img/", exist_ok=True)
    mock_test_file = "dummy_test_patch.h5"
    mock_path = f"./datasets/TestData/img/{mock_test_file}"

    if not os.path.exists(mock_path):
        with h5py.File(mock_path, "w") as f:
            f.create_dataset("img", data=np.random.rand(128, 128, 14))

    try:
        execute_vision_inference_pass(mock_test_file)
    except Exception as e:
        print(f"System status check complete. Core validation track: {str(e)}")
