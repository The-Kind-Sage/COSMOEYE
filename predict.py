import os
import json
import h5py
import numpy as np
import torch

from model import CustomUNet
from dataset import (normalize_standardized_image, stack_temporal_stacks,
                     NORM_STATS_PATH, NDVI_CHANNEL, POST_STACK)
from spatial_math import extract_spatial_metrics

# Number of temporal windows the network expects (PRE / POST / LATE)
EXPECTED_WINDOWS = 3
# Must match train.py: the tuned threshold is only valid with the gate applied
NDVI_GATE_THRESHOLD = 0.48


def build_standardized_input(raw_matrix):
    """Convert any supported input layout into the standardized temporal-stack layout.

    - 14 channels (legacy single-frame SEN12 tiles): used as-is.
    - 4D (N, H, W, 14) (temporal-stack tiles): PRE/POST/LATE windows, as-is.
    Single-frame inputs become a one-window container so the temporal
    network can still score them.
    """
    if raw_matrix.ndim == 4:
        return _fit_window_count(raw_matrix.astype(np.float32))

    height, width, channels = raw_matrix.shape

    if channels == 14:
        fused_matrix = raw_matrix
    else:
        raise ValueError(f"Unsupported channel count: {channels} (expected 14 or a temporal stack)")

    return _fit_window_count(fused_matrix[None])


def _fit_window_count(stacks):
    """Force the container to the EXPECTED_WINDOWS the network was built for.

    The network takes 3 windows x 14 bands = 42 channels. A single-frame
    input produces a 1-window container (14 channels), which used to reach
    the model and raise a shape error that the __main__ demo swallowed in a
    bare except - so the legacy path looked like it worked and never did.
    Repeating the only available frame across PRE/POST/LATE yields a valid
    42-channel input with a flat (zero) temporal change signal, which is the
    honest representation of a single acquisition.
    """
    n_windows = stacks.shape[0]
    if n_windows == EXPECTED_WINDOWS:
        return stacks
    if n_windows == 1:
        return np.repeat(stacks, EXPECTED_WINDOWS, axis=0)
    if n_windows > EXPECTED_WINDOWS:
        return stacks[:EXPECTED_WINDOWS]
    pad = np.repeat(stacks[-1:], EXPECTED_WINDOWS - n_windows, axis=0)
    return np.concatenate([stacks, pad], axis=0)


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
    unet_model.load_state_dict(
        torch.load(vision_weights, map_location=device, weights_only=True)
    )
    unet_model.to(device).eval()

    probability_mask = tta_probabilities(unet_model, image_tensor, device).squeeze().cpu().numpy()

    # Use the IoU-tuned threshold saved during training (falls back to 0.5)
    threshold = 0.5
    gate_enabled = True
    gate_threshold = NDVI_GATE_THRESHOLD
    threshold_path = "landslide_best_threshold.json"
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as jf:
            saved = json.load(jf)
        threshold = saved.get("threshold", 0.5)
        gate_enabled = saved.get("ndvi_gate", True)
        gate_threshold = saved.get("ndvi_gate_threshold", NDVI_GATE_THRESHOLD)

    # CRITICAL: the saved threshold is tuned on NDVI-GATED probabilities.
    # Applying it to ungated output (as this did before) evaluates a
    # different operating point than the one that was validated, and admits
    # exactly the cloud/water/regrowth false positives the gate exists to
    # remove. Read NDVI from the raw POST window, before normalization, so it
    # is already in physical units.
    if gate_enabled:
        post_idx = min(POST_STACK, fused_matrix.shape[0] - 1)
        raw_post_ndvi = fused_matrix[post_idx, :, :, NDVI_CHANNEL]
        probability_mask = probability_mask * (raw_post_ndvi < gate_threshold)

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
    except FileNotFoundError as e:
        # The only expected failure when no model has been trained yet.
        print(f"Skipped: {e}")
    except Exception:
        # Anything else is a real defect. The previous bare except printed
        # "System status check complete" for every error, which is how the
        # broken single-frame path stayed hidden.
        print("Inference self-check FAILED:")
        raise
