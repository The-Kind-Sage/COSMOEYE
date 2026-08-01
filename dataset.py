import os
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Channel 13 of the standardized 14-band layout always holds NDVI.
NDVI_CHANNEL = 13
NORM_STATS_PATH = "./datasets/TrainData/norm_stats.npz"

# Stack index of the post-event frame inside the temporal-stack layout
# (the LAST window always holds the near-event/late state).
POST_STACK = -1


def build_or_load_norm_stats(img_dir, sample_size=1500, seed=0):
    """Dataset-global per-channel mean/std, computed once and cached.

    Per-image min-max normalization was discarded: it destroys the physical
    intensity relationships between bands (every band is stretched to the
    same range regardless of what it measures). Global z-score statistics
    keep those cross-band relations intact and give the network a
    consistent input scale for every tile.

    Works for both the legacy (H, W, 14) layout and the temporal-stack
    (N, H, W, 14) layout: stats are computed per spectral channel over
    every spatial/stack position.
    """
    if os.path.exists(NORM_STATS_PATH):
        with np.load(NORM_STATS_PATH) as d:
            return d["mean"].astype(np.float32), d["std"].astype(np.float32)

    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".h5"))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(files), min(sample_size, len(files)), replace=False)

    n_pix = 0.0
    s = np.zeros(14, dtype=np.float64)
    ss = np.zeros(14, dtype=np.float64)
    for i in idx:
        with h5py.File(os.path.join(img_dir, files[i]), "r") as f:
            arr = np.array(f["img"]).astype(np.float64)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        flat = arr.reshape(-1, arr.shape[-1])
        n_pix += flat.shape[0]
        s += flat.sum(axis=0)
        ss += (flat ** 2).sum(axis=0)

    mean = s / n_pix
    var = np.maximum(ss / n_pix - mean ** 2, 0.0)
    std = np.sqrt(var)
    np.savez(NORM_STATS_PATH, mean=mean.astype(np.float32), std=std.astype(np.float32))
    return mean.astype(np.float32), std.astype(np.float32)


def stack_temporal_stacks(raw_img):
    """Convert the raw image container into the (C, H, W) network layout.

    Temporal-stack files (N, H, W, 14) are flattened channel-wise:
    [window 0: 14 bands | window 1: 14 bands | ...] = N*14 channels.
    Legacy single-frame files (H, W, 14) are accepted as a single-window
    container (no temporal signal, treated as the late-state stack).
    """
    if raw_img.ndim == 4:
        stacks = raw_img
    else:
        stacks = raw_img[None]
    # stacks: (N, H, W, C) -> transpose -> (N, C, H, W) -> (N*C, H, W)
    return np.ascontiguousarray(
        stacks.transpose(0, 3, 1, 2).reshape(
            stacks.shape[0] * stacks.shape[-1], stacks.shape[-3], stacks.shape[-2]
        )
    )


def normalize_standardized_image(img_array, mean=None, std=None):
    """Normalize a (H, W, C) satellite patch for the network.

    Primary path: global per-channel z-score using cached dataset stats.
    Fallback (e.g. inference without a training folder): per-channel
    min-max, which still maps everything into [0, 1].
    """
    if mean is not None and std is not None:
        denom = np.where(std > 1e-6, std, 1.0)
        return ((img_array.astype(np.float32) - mean.astype(np.float32)) / denom).astype(np.float32)

    out = np.zeros_like(img_array, dtype=np.float32)
    for c in range(img_array.shape[-1]):
        ch = img_array[..., c].astype(np.float32)
        if c == NDVI_CHANNEL:
            ch = np.clip(ch, -1.0, 1.0)
            out[..., c] = (ch + 1.0) / 2.0
        else:
            ch_min = float(np.min(ch))
            ch_max = float(np.max(ch))
            if ch_max - ch_min > 1e-8:
                out[..., c] = (ch - ch_min) / (ch_max - ch_min)
    return out


def compute_label_weights(mask, raw_ndvi):
    """Per-pixel training weights that compensate for noisy polygon labels.

    Audit of the dataset showed ~40% of mask pixels actually look like
    vegetation (the hand-drawn polygons over-cover the real landslide),
    and there are also unlabeled bare/rock areas that are probably missed
    landslides. The loss therefore trusts:
      - mask=1 pixels that look like bare ground (NDVI < 0.48): fully,
      - mask=1 pixels that still look vegetated (over-cover noise): weakly,
      - mask=0 pixels that look like vegetation: fully,
      - mask=0 pixels that look like bare ground (possible unlabeled
        landslide): partially.
    This stops the model from burning capacity on fitting label noise.
    """
    weights = np.ones_like(mask, dtype=np.float32)
    landslide_like = raw_ndvi < 0.48
    weights[(mask > 0) & (~landslide_like)] = 0.40  # noisy positive (over-cover)
    weights[(mask == 0) & (landslide_like)] = 0.65  # suspicious negative (under-cover)
    return weights


class LandslideDataset(Dataset):
    def __init__(self, img_dir="./datasets/TrainData/img/",
                 mask_dir="./datasets/TrainData/mask/",
                 augmentations=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augmentations = augmentations
        # Only index tiles that can actually feed the temporal network:
        # the (3,128,128,14) change-pair layout packs to ~2.75 MB on disk.
        # Any stale/legacy file (e.g. old single-frame (128,128,14) tiles)
        # must never enter a batch - a single mismatched layout crashes the
        # DataLoader collate step deterministically at the same epoch.
        self.file_list = sorted(
            f for f in os.listdir(img_dir)
            if f.endswith(".h5") and os.path.getsize(os.path.join(img_dir, f)) >= 2_000_000
        )
        self.norm_mean, self.norm_std = build_or_load_norm_stats(img_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        target_filename = self.file_list[idx]

        img_path = os.path.join(self.img_dir, target_filename)
        mask_path = os.path.join(self.mask_dir, target_filename)

        with h5py.File(img_path, "r") as f_img:
            raw_img = np.array(f_img["img"]).astype(np.float32)

        with h5py.File(mask_path, "r") as f_msk:
            mask_array = np.array(f_msk["mask"]).astype(np.float32)

        # Sanitize non-finite values (NaN/inf from occluded tiles would
        # otherwise poison the z-score transform and the loss).
        raw_img = np.nan_to_num(raw_img, nan=0.0, posinf=0.0, neginf=0.0)

        # Global z-score normalization (keeps cross-band intensity relations)
        img_array = normalize_standardized_image(raw_img, self.norm_mean, self.norm_std)

        # Flatten the temporal stacks into a single (N*14, H, W) image BEFORE
        # augmentation so geometric/photometric transforms hit every stack
        # identically (the inter-window change signal must be preserved).
        img_array = stack_temporal_stacks(img_array).transpose(1, 2, 0)

        # Hard-binarize the label map: 1 = landslide, 0 = background
        mask_u8 = (mask_array > 0.5).astype(np.uint8)

        # Per-pixel label-trust weights from the raw post-event NDVI band
        if raw_img.ndim == 4:
            post_ndvi = raw_img[POST_STACK, ..., NDVI_CHANNEL]
        else:
            post_ndvi = raw_img[..., NDVI_CHANNEL]
        weight_map = compute_label_weights(mask_u8, post_ndvi)

        # Boost loss weight on the label boundary ring (3x3 morph ring):
        # noisy polygons over-cover their edges, so boundary pixels get
        # extra supervision to carve out accurate landslide outlines.
        if mask_u8.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            boundary = cv2.dilate(mask_u8, kernel) - cv2.erode(mask_u8, kernel)
            weight_map[boundary > 0] *= 1.5

        if self.augmentations is not None:
            augmented = self.augmentations(image=img_array, mask=mask_u8, weight=weight_map)
            img_array = augmented["image"].astype(np.float32)
            mask_u8 = augmented["mask"]
            weight_map = augmented["weight"].astype(np.float32)

        # torch.tensor() COPIES the numpy buffer, giving each tensor its own
        # resizable storage. torch.from_numpy would alias non-resizable numpy
        # storage, which intermittently crashes the DataLoader collate step
        # ("Trying to resize storage that is not resizable").
        image_tensor = torch.tensor(
            np.ascontiguousarray(img_array.transpose(2, 0, 1))
        )
        mask_tensor = torch.tensor(
            np.ascontiguousarray((mask_u8 > 0).astype(np.float32))
        ).unsqueeze(0)
        weight_tensor = torch.tensor(
            np.ascontiguousarray(weight_map)
        ).unsqueeze(0)

        return image_tensor, mask_tensor, weight_tensor

if __name__ == "__main__":
    dataset = LandslideDataset()
    print(f"Total training files indexed: {len(dataset)}")
    if len(dataset) > 0:
        img, msk, w = dataset[0]
        print(f"Image tensor shape: {img.shape}")
        print(f"Mask tensor shape: {msk.shape}")
        print(f"Weight tensor shape: {w.shape}")
        print(f"Z-scored input bounds: Min={img.min().item():.4f}, Max={img.max().item():.4f}")
        print(f"Weight map unique values: {torch.unique(w).tolist()}")
