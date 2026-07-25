import h5py
import numpy as np
import torch


def load_and_standardize_satellite_matrix(test_img_path):
    """Dynamically audits input matrix structures to ensure absolute shape matching

    for both raw 13-band and pre-stacked 14-band satellite image configurations.
    """
    # 1. Ingest raw array out of the local h5 container
    with h5py.File(test_img_path, "r") as f:
        raw_matrix = np.array(f["img"]).astype(np.float32)

    # 2. Extract shape variables -> (Height, Width, Channels)
    height, width, channels = raw_matrix.shape

    # 3. Dynamic Conditional Adaptation Layer
    if channels == 13:
        print(
            f"Native 13-band Sentinel-2 file detected. Launching manual step-stacking matrix layer..."
        )

        # Extract Red (Index 3) and NIR (Index 7) channels
        red = raw_matrix[:, :, 3]
        nir = raw_matrix[:, :, 7]

        # Compute NDVI feature grid from scratch
        computed_ndvi = (nir - red) / (nir + red + 1e-8)

        # Expand dimension from (128, 128) to (128, 128, 1)
        computed_ndvi = np.expand_dims(computed_ndvi, axis=-1)

        # Fused matrix output shape becomes exactly (128, 128, 14)
        fused_matrix = np.concatenate([raw_matrix, computed_ndvi], axis=-1)

    elif channels == 14:
        print(
            f"Pre-stacked 14-band benchmark file detected. Passing directly to tensor converter..."
        )
        fused_matrix = raw_matrix

    else:
        raise ValueError(
            f"Matrix Dimension Failure: Expected 13 or 14 channels, but received {channels}."
        )

    # 4. Transpose axes to match standard PyTorch formatting -> (14, 128, 128)
    image_tensor = torch.from_numpy(fused_matrix).permute(2, 0, 1)

    return image_tensor
