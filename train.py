import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom modules directly from your workspace folders
from model import CustomUNet
from dataset import LandslideDataset
from augmentations import get_training_augmentations


def train_pipeline(epochs=5, batch_size=4, lr=1e-4):
    """Executes the complete deep learning optimization and backpropagation

    loop to calibrate model layer weights.
    """
    # 1. Hardware Selection (Uses GPU if available, otherwise defaults to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting System Computing Node: {device}")

    # 2. Local Dataset Path Definitions
    train_img_dir = "./datasets/TrainData/img/"
    train_mask_dir = "./datasets/TrainData/mask/"
    valid_img_dir = "./datasets/ValidData/img/"
    # Landslide4Sense validation data is unmasked for blind ranking checks,
    # so we will focus the monitoring metrics on our training pipeline step data split.

    if not os.path.exists(train_img_dir):
        print(f"Error: Missing data matrices folder path at {train_img_dir}")
        return

    # 3. Instantiate Dataset Containers
    print("Loading structural data files into local environment...")
    train_dataset = LandslideDataset(
        img_dir=train_img_dir, mask_dir=train_mask_dir
    )

    # 4. Bind Batches to Active DataStreams
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # 5. Initialize Model Architecture with 14 Multi-Spectral Inputs
    model = CustomUNet(in_channels=14, out_channels=1).to(device)

    # 6. Define Loss Function and Layer Weight Optimizer
    # BCEWithLogitsLoss or BCELoss measures binary pixel divergence grids
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load augmentation transformation pipeline matrix constraints
    transforms = get_training_augmentations()

    print("\n================== STARTING MODEL TRAINING ==================")
    print(f"Total Epochs scheduled: {epochs} | Batch Processing Size: {batch_size}")
    print("=============================================================\n")

    # 7. Optimization Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            # Move data matrices to active compute node memory (RAM or VRAM)
            images = images.to(device)
            masks = masks.to(device)

            # --- Apply Local Augmentations To Active Batch ---
            # Convert tensors to numpy arrays to pass through Albumentations
            for i in range(images.shape[0]):
                img_np = (
                    images[i].permute(1, 2, 0).cpu().numpy()
                )  # Back to (H,W,C)
                mask_np = masks[i][0].cpu().numpy()  # Extract (H,W)

                augmented = transforms(image=img_np, mask=mask_np)

                # Convert processed numpy arrays back to PyTorch variables
                images[i] = (
                    torch.from_numpy(augmented["image"])
                    .permute(2, 0, 1)
                    .to(device)
                )
                masks[i][0] = torch.from_numpy(augmented["mask"]).to(device)

            # --- Forward Pass Optimization Pass ---
            optimizer.zero_grad()  # Reset old gradient calculations
            predictions = model(images)  # Compute matrix guesses
            loss = criterion(predictions, masks)  # Calculate current loss score

            # --- Backpropagation Parameter Updates ---
            loss.backward()  # Calculate partial derivatives
            optimizer.step()  # Update network layers weights vector

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Current Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / len(train_loader)
        print(
            f"--- Epoch [{epoch+1}/{epochs}] Summary Complete. Mean Training Loss: {epoch_loss:.4f} ---\n"
        )

    # 8. Save Optimized Model State Weights Local Package File
    weights_output_path = "landslide_unet_weights.pth"
    torch.save(model.state_dict(), weights_output_path)
    print(
        f"Optimization complete. Local weights saved successfully to: {weights_output_path}"
    )


if __name__ == "__main__":
    # Test execution pass running for a single step epoch on local hardware configuration
    train_pipeline(epochs=1, batch_size=2)