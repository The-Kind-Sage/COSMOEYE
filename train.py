import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom modules directly from your workspace folder
from model import CustomUNet
from dataset import LandslideDataset
from augmentations import get_training_augmentations


def train_pipeline(epochs=5, batch_size=16, lr=1e-4):
    """Executes the complete deep learning optimization loop, forcing graphics memory

    hardware acceleration as primary computing resource.
    """
    # 1. Primary GPU Hardware Selection Check
    # Priority: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU fallback
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable benchmark mode to optimize convolutional operations for constant shapes
        torch.backends.cudnn.benchmark = True
        print(f"Accelerating via Primary Resource (NVIDIA GPU): {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Accelerating via Primary Resource (Apple Silicon MPS GPU)")
    else:
        device = torch.device("cpu")
        print("Primary GPU assets missing. Defaulting to Secondary Resource: Host CPU")

    # 2. Local Dataset Path Definitions
    train_img_dir = "./datasets/TrainData/img/"
    train_mask_dir = "./datasets/TrainData/mask/"

    if not os.path.exists(train_img_dir) or not os.path.exists(train_mask_dir):
        print(f"Error: Missing data directories. Ensure data exists at {train_img_dir}")
        return

    # 3. Instantiate Dataset Container
    print("Loading structural data files into local environment...")
    train_dataset = LandslideDataset(img_dir=train_img_dir, mask_dir=train_mask_dir)

    # 4. Bind Batches to Active DataStreams with Async Thread Pinning
    # num_workers=2 and pin_memory=True allows CPU RAM to pre-fetch and stage 
    # tensor batches into memory arrays ahead of GPU execution steps.
    use_pin = True if device.type != "cpu" else False
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=2,
        pin_memory=use_pin
    )

    # 5. Initialize Model Architecture and allocate layers instantly to primary device
    model = CustomUNet(in_channels=14, out_channels=1).to(device)

    # 6. Define Loss Function and Layer Weight Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load augmentation transformation pipeline matrix constraints from Step 11
    transforms = get_training_augmentations()

    print("\n================== STARTING ACCELERATED MODEL TRAINING ==================")
    print(f"Total Epochs scheduled: {epochs} | Batch Processing Size: {batch_size}")
    print(f"Active Memory Computing Node target: {device}")
    print("=========================================================================\n")

    # 7. Optimization Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            augmented_images = []
            augmented_masks = []

            # Process augmentations locally on CPU before pushing matrices to GPU
            for i in range(images.shape[0]):
                img_np = images[i].permute(1, 2, 0).numpy()
                mask_np = masks[i].squeeze(0).numpy()

                augmented = transforms(image=img_np, mask=mask_np)

                aug_img_tensor = torch.from_numpy(augmented["image"]).permute(2, 0, 1).float()
                aug_mask_tensor = torch.from_numpy(augmented["mask"]).unsqueeze(0).float()

                augmented_images.append(aug_img_tensor)
                augmented_masks.append(aug_mask_tensor)

            # Re-stack augmented list components into batch tensors
            # Explicitly load tensors into the primary device memory space (.to(device))
            images = torch.stack(augmented_images).to(device, non_blocking=True)
            masks = torch.stack(augmented_masks).to(device, non_blocking=True)

            # --- Forward Pass Optimization Pass on GPU ---
            optimizer.zero_grad(set_to_none=True)  # set_to_none=True minimizes memory footprint overhead
            predictions = model(images)  
            loss = criterion(predictions, masks)  

            # --- Backpropagation Parameter Updates on GPU ---
            loss.backward()  
            optimizer.step()  

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Current Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"--- Epoch [{epoch+1}/{epochs}] Summary Complete. Mean Training Loss: {epoch_loss:.4f} ---\n")

    # 8. Save Optimized Model State Weights Local Package File
    weights_output_path = "landslide_unet_weights.pth"
    # Ensure saved weights are mapped to CPU storage space for flexible offline deployment
    torch.save(model.to("cpu").state_dict(), weights_output_path)
    print(f"Optimization complete. Local weights saved successfully to: {weights_output_path}")


if __name__ == "__main__":
    # If running on an external cloud GPU notebook (Colab/Kaggle), batch size can be raised to 16 or 32
    # If testing locally on a standard laptop CPU footprint, adjust batch size down to 2 or 4
    train_pipeline(epochs=5, batch_size=16)
