import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class LandslideDataset(Dataset):
    """Custom PyTorch Dataset class built from scratch to parse, process,

    and stream multi-spectral .h5 satellite matrices.
    """

    def __init__(self, img_dir, mask_dir=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        # Sort the directories to ensure file indexes align perfectly
        self.img_files = sorted(os.listdir(img_dir))
        if mask_dir:
            self.mask_files = sorted(os.listdir(mask_dir))

            # Academic sanity check to ensure images match masks exactly
            assert len(self.img_files) == len(
                self.mask_files
            ), "Mismatched file count between images and masks folders."

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 1. Resolve absolute local file paths
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 2. Extract multi-spectral image tensor
        with h5py.File(img_path, "r") as f:
            # Shape format inside .h5 is (128, 128, 14) -> (H, W, C)
            image_array = torch.tensor(f["img"][:], dtype=torch.float32)

        # CRITICAL RE-SHAPING FOR PYTORCH
        # PyTorch convolutional layers strictly expect (Channels, Height, Width)
        # We use permute to convert (128, 128, 14) to (14, 128, 128)
        image_tensor = image_array.permute(2, 0, 1)

        # 3. Extract matching ground-truth mask if available (Training/Validation sets)
        if self.mask_dir:
            mask_name = self.mask_files[idx]
            mask_path = os.path.join(self.mask_dir, mask_name)

            with h5py.File(mask_path, "r") as f:
                # Shape format inside mask file is (128, 128) -> (H, W)
                mask_array = torch.tensor(f["mask"][:], dtype=torch.float32)

            # PyTorch segmentation loss functions require explicit channel metrics
            # Unsqueeze adds a dimension, converting (128, 128) to (1, 128, 128)
            mask_tensor = mask_array.unsqueeze(0)

            return image_tensor, mask_tensor

        # For unmasked inference testing (TestData)
        return image_tensor


# --- Local Pipeline Verification Routine ---
if __name__ == "__main__":
    print("Initializing Custom Dataset Pipeline Verification Check...")

    # Point directly to your extracted training repositories
    train_images = "./datasets/TrainData/img/"
    train_masks = "./datasets/TrainData/mask/"

    if not os.path.exists(train_images) or not os.path.exists(train_masks):
        print(
            "Error: Datasets folder structure missing. Please verify local files pathing."
        )
    else:
        # 1. Initialize the dataset container
        dataset = LandslideDataset(img_dir=train_images, mask_dir=train_masks)
        print(f"Total localized database files indexed: {len(dataset)}")

        # 2. Pull a single sample using array indexing to verify matrix modifications
        sample_img, sample_mask = dataset[0]

        print("\n------ Single Tensor Extraction Summary ------")
        print(f"PyTorch Image Tensor Dimensions : {sample_img.shape}")
        print(f"PyTorch Mask Tensor Dimensions  : {sample_mask.shape}")
        print(f"Image Maximum Pixel Value       : {torch.max(sample_img):.4f}")
        print(f"Image Minimum Pixel Value       : {torch.min(sample_img):.4f}")
        print("----------------------------------------------")

        # 3. Test the PyTorch batch generator loader configuration
        # Batch size 4 means it packages 4 tensors together down the pipeline at once
        data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Pull the very first stacked matrix block
        loader_iter = iter(data_loader)
        batch_images, batch_masks = next(loader_iter)

        print("\n================== STEP 10 DATA LOADER VERIFICATION ==================")
        print("Custom PyTorch Dataset pipeline compiled successfully.")
        print(
            f"Stacked Loader Batch Output Image Grid Dimension: {batch_images.shape}"
        )
        print(
            f"Stacked Loader Batch Output Mask Grid Dimension:  {batch_masks.shape}"
        )
        print("Core data routing is fully prepared for backpropagation loops.")
        print("======================================================================\n")