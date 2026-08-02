import random

import cv2
import numpy as np
import albumentations as A

# Per-window band layout (must match convert_sen12.py):
#   0-9 = spectral B02..B12, 10 = DEM, 11 = slope, 12 = reserved, 13 = NDVI
BANDS_PER_WINDOW = 14
SPECTRAL_PER_WINDOW = 10


def _spectral_channels(n_channels):
    """Indices of the spectral channels in the flattened multi-window layout.

    DEM/slope/NDVI are physical channels: they are static terrain or a
    derived index the NDVI gate depends on, so photometric augmentation must
    never touch them.
    """
    n_windows = max(1, n_channels // BANDS_PER_WINDOW)
    return [
        w * BANDS_PER_WINDOW + b
        for w in range(n_windows)
        for b in range(SPECTRAL_PER_WINDOW)
    ]


def band_drop(image, **kwargs):
    """Simulate a failed/dropped spectral band by zeroing one spectral
    channel. DEM/slope/NDVI (physical channels) are never dropped.
    Zero == the z-score mean, i.e. "no information".
    """
    spectral = _spectral_channels(image.shape[-1])
    out = image.copy()
    for _ in range(random.randint(1, 2)):
        out[..., random.choice(spectral)] = 0.0
    return out


def window_jitter(image, **kwargs):
    """Per-window acquisition jitter: each temporal window's SPECTRAL bands
    get an independent gain/bias, simulating different illumination and
    atmospheric conditions across the pre/post/late scenes.

    Only bands 0-9 of each window are touched. DEM and slope are byte-identical
    in every window (static terrain) and NDVI carries the physical scale the
    inference gate thresholds at 0.48 - jittering either would inject a
    change signal that does not exist in the real data.
    """
    out = image.copy()
    n_windows = max(1, image.shape[-1] // BANDS_PER_WINDOW)
    for w in range(n_windows):
        gain = random.uniform(0.9, 1.1)
        bias = random.uniform(-0.1, 0.1)
        start = w * BANDS_PER_WINDOW
        sl = slice(start, start + SPECTRAL_PER_WINDOW)
        out[..., sl] = out[..., sl] * gain + bias
    return out


def brightness_contrast(image, **kwargs):
    """Brightness/contrast jitter that is safe on z-scored data.

    albumentations' RandomBrightnessContrast assumes float images live in
    [0, 1] and CLIPS its output to that range. Our inputs are z-scores
    (roughly [-5, 5]), so that transform silently destroyed every negative
    value and everything above 1. This re-implements the same effect around
    each channel's own mean, without any clipping.
    """
    spectral = _spectral_channels(image.shape[-1])
    out = image.copy()
    alpha = 1.0 + random.uniform(-0.1, 0.1)   # contrast
    beta = random.uniform(-0.1, 0.1)          # brightness (in z units)
    block = out[..., spectral]
    centre = block.mean(axis=(0, 1), keepdims=True)
    out[..., spectral] = (block - centre) * alpha + centre + beta
    return out


def gauss_noise(image, **kwargs):
    """Additive sensor noise in z-score units (no clipping).

    sigma is expressed as a fraction of one standard deviation, which is the
    natural scale after z-scoring.
    """
    spectral = _spectral_channels(image.shape[-1])
    out = image.copy()
    sigma = random.uniform(0.02, 0.06)
    block = out[..., spectral]
    out[..., spectral] = block + np.random.normal(
        0.0, sigma, size=block.shape
    ).astype(np.float32)
    return out


def gaussian_blur(image, **kwargs):
    """3x3 Gaussian blur applied per channel via cv2 (no clipping).

    cv2.GaussianBlur handles at most 512 channels at once and preserves the
    float range, unlike albumentations' clipping float path.
    """
    out = image.copy()
    out[:] = cv2.GaussianBlur(out, (3, 3), sigmaX=0)
    return out


def get_training_augmentations():
    """Lightweight augmentation pipeline for landslide segmentation.

    Landslide pixels cover roughly 1-2% of each 128x128 tile, so elastic
    warping and perspective distortions are intentionally avoided: they
    smear the very small positive regions the model must learn. The
    pipeline adds mild scale jitter, coarse hole-dropout (clouds), and
    random band dropout (sensor failures) on top of flips/rotations and
    light photometric noise. Zeros = z-score mean, so holes/dropped bands
    read as "no information" instead of dark artifacts.

    Every photometric step is a custom Lambda: the stock albumentations
    intensity transforms clamp float images to [0, 1], which is invalid for
    z-scored satellite input.
    """
    return A.Compose(
        [
            # 1. Geometric: flips and 90-degree rotations (label-safe)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # 2. Mild scale jitter for robustness to tile boundary effects
            A.Affine(scale=(0.9, 1.1), p=0.3),
            # 3. Coarse hole dropout simulating cloud cover. fill_mask=0 also
            #    zeroes the weight map (declared as a mask target), so occluded
            #    pixels are excluded from the loss instead of being taught as
            #    background.
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(8, 24),
                hole_width_range=(8, 24),
                fill=0, fill_mask=0, p=0.2,
            ),
            # 4. Random spectral band dropout (failed-sensor simulation)
            A.Lambda(image=band_drop, p=0.3),
            # 5. Per-window acquisition-intensity jitter (temporal robustness)
            A.Lambda(image=window_jitter, p=0.4),
            # 6. Mild photometric variation to improve illumination robustness
            A.Lambda(image=brightness_contrast, p=0.3),
            # 7. Light sensor-noise / focus simulation (kept subtle)
            A.Lambda(image=gauss_noise, p=0.2),
            A.Lambda(image=gaussian_blur, p=0.15),
        ],
        additional_targets={"weight": "mask"},
    )


# --- Local Pipeline Verification Routine ---
if __name__ == "__main__":
    print("Initializing Augmentation Verification Check...")

    # Dummy z-scored multi-spectral patch (H, W, 42) + label mask + weights
    rng = np.random.RandomState(0)
    fake_image = rng.randn(128, 128, 42).astype(np.float32)
    fake_mask = (rng.rand(128, 128) > 0.98).astype(np.uint8)
    fake_weight = np.ones((128, 128), dtype=np.float32)

    transform_pipeline = get_training_augmentations()

    # Range preservation is the property that matters: a clipping transform
    # collapses every negative z-score to 0 and everything above 1 to 1.
    worst_min, worst_max = 0.0, 0.0
    for _ in range(200):
        out = transform_pipeline(
            image=fake_image, mask=fake_mask, weight=fake_weight
        )["image"]
        worst_min = min(worst_min, float(out.min()))
        worst_max = max(worst_max, float(out.max()))

    augmented_data = transform_pipeline(
        image=fake_image, mask=fake_mask, weight=fake_weight
    )
    warped_image = augmented_data["image"]
    warped_mask = augmented_data["mask"]
    warped_weight = augmented_data["weight"]

    print("\n================== AUGMENTATION OUTPUT ==================")
    print(f"Input  : {fake_image.shape} range [{fake_image.min():.2f}, {fake_image.max():.2f}]")
    print(f"Output : {warped_image.shape} | mask {warped_mask.shape} | weight {warped_weight.shape}")
    print(f"Range over 200 draws: [{worst_min:.2f}, {worst_max:.2f}]")

    shapes_ok = (
        warped_image.shape == fake_image.shape
        and warped_mask.shape == fake_mask.shape
        and warped_weight.shape == fake_weight.shape
    )
    # If any transform clipped, the observed minimum would be exactly 0.0
    no_clipping = worst_min < -1.0 and worst_max > 1.0

    if shapes_ok and no_clipping:
        print("Success: spatial dims preserved and z-score range intact (no [0,1] clipping).")
    else:
        print(f"FAILURE: shapes_ok={shapes_ok} no_clipping={no_clipping}")
    print("==========================================================\n")
