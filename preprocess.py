import os
import h5py
import numpy as np


def compute_spectral_ndvi(h5_file_path):
    """Load a Landslide4Sense .h5 file and compute the NDVI matrix."""

    # Open file and read image tensor (H, W, Bands)
    with h5py.File(h5_file_path, "r") as f:
        # Fixed internal key to match the tekbahadurkshetri/landslide4sense layout
        image = np.array(f["img"])

    # Extract Red (B4) and NIR (B8) bands
    red = image[:, :, 3].astype(np.float32)
    nir = image[:, :, 7].astype(np.float32)

    # NDVI = (NIR - Red) / (NIR + Red)
    numerator = nir - red
    denominator = nir + red + 1e-8  # avoid division by zero
    ndvi = numerator / denominator

    return ndvi


if __name__ == "__main__":
    data_dir = "./datasets/TrainData/img/"

    print("Starting NDVI preprocessing check...")

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
    else:
        files = sorted(os.listdir(data_dir))

        if not files:
            print("No files found in the directory.")
        else:
            sample_file = files[0]
            sample_path = os.path.join(data_dir, sample_file)

            print(f"Using sample file: {sample_file}")

            ndvi_result = compute_spectral_ndvi(sample_path)

            print("\n------ NDVI Output Summary ------")
            print(f"Shape        : {ndvi_result.shape}")
            print(
                f"Min NDVI (Rock/Mud Anomaly)    : {np.min(ndvi_result):.4f}"
            )
            print(
                f"Max NDVI (Dense Forest Canopy) : {np.max(ndvi_result):.4f}"
            )
            print(f"Mean NDVI    : {np.mean(ndvi_result):.4f}")
            print("----------------------------------\n")
