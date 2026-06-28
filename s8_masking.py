import os
import h5py
import numpy as np
from preprocess import compute_spectral_ndvi


def isolate_landslide_signature(img_path, mask_path):
    """Compute NDVI and split pixels using the ground-truth mask."""
    
    # Get NDVI from Step 7
    ndvi = compute_spectral_ndvi(img_path)

    # Load corresponding mask (1 = landslide, 0 = background)
    with h5py.File(mask_path, "r") as f:
        mask = np.array(f["mask"]).astype(np.uint8)

    # Separate NDVI values using the mask
    landslide_pixels = ndvi[mask == 1]
    background_pixels = ndvi[mask == 0]

    return landslide_pixels, background_pixels, mask


if __name__ == "__main__":
    img_dir = "./datasets/TrainData/img/"
    mask_dir = "./datasets/TrainData/mask/"

    print("Starting landslide masking check...")

    images = sorted(os.listdir(img_dir)) if os.path.exists(img_dir) else []
    masks = sorted(os.listdir(mask_dir)) if os.path.exists(mask_dir) else []

    if not images or not masks:
        print("Image or mask folder is empty / missing.")
    else:
        img_path = os.path.join(img_dir, images[0])
        mask_path = os.path.join(mask_dir, masks[0])

        print(f"Sample image: {images[0]}")
        print(f"Sample mask : {masks[0]}")

        landslide_vals, background_vals, mask = isolate_landslide_signature(
            img_path, mask_path
        )

        print("\n------ Masking Output Summary ------")
        print(f"Mask shape           : {mask.shape}")
        print(f"Landslide pixels     : {len(landslide_vals)}")
        print(f"Background pixels    : {len(background_vals)}")

        if len(landslide_vals) > 0:
            print(f"Mean NDVI (landslide): {np.mean(landslide_vals):.4f}")
            print(f"Mean NDVI (background): {np.mean(background_vals):.4f}")
        else:
            print("No landslide pixels in this sample.")
        print("------------------------------------\n")