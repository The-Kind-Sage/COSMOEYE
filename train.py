import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Custom workspace modules
from model import CustomUNet
from dataset import LandslideDataset
from augmentations import get_training_augmentations


def compute_evaluation_metrics(predictions, masks):
    """Calculates semantic segmentation evaluation metrics from raw tensors."""
    # Threshold probabilities to binary grids
    pred_binary = (predictions > 0.5).float()
    target_binary = masks.float()

    # Confusion matrix elements
    true_positive = torch.sum(pred_binary * target_binary).item()
    false_positive = torch.sum(pred_binary * (1.0 - target_binary)).item()
    false_negative = torch.sum((1.0 - pred_binary) * target_binary).item()
    true_negative = torch.sum((1.0 - pred_binary) * (1.0 - target_binary)).item()
    total_pixels = pred_binary.numel()

    # Calculate metrics
    accuracy = (true_positive + true_negative) / (total_pixels + 1e-8)
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    # IoU = TP / (TP + FP + FN)
    intersection = true_positive
    union = true_positive + false_positive + false_negative
    mean_iou = intersection / (union + 1e-8)

    return accuracy, precision, recall, f1_score, mean_iou


def train_pipeline(epochs=10, batch_size=16, lr=1e-4):
    """Executes the complete deep learning training and evaluation pipeline."""
    # 1. Hardware detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "Host CPU"

    # 2. Local data paths
    train_img_dir = "./datasets/TrainData/img/"
    train_mask_dir = "./datasets/TrainData/mask/"

    if not os.path.exists(train_img_dir) or not os.path.exists(train_mask_dir):
        print(f"Error: Missing data directories at {train_img_dir}")
        return

    # Start timing and VRAM tracking
    start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 3. Data pipeline setup
    train_dataset = LandslideDataset(img_dir=train_img_dir, mask_dir=train_mask_dir)
    use_pin = True if device.type != "cpu" else False
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=2,
        pin_memory=use_pin
    )

    # 4. Initialize model, loss, optimizer, and augmentations
    model = CustomUNet(in_channels=14, out_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    transforms = get_training_augmentations()

    print("\n================== STARTING MODEL TRAINING ==================")
    print(f"Total Epochs scheduled: {epochs} | Batch Processing Size: {batch_size}")
    print(f"Active Hardware Processing Accelerator Node: {device_name}")
    print("=============================================================\n")

    total_batches = len(train_loader)
    
    # 5. Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            augmented_images = []
            augmented_masks = []

            # Apply augmentations sample by sample
            for i in range(images.shape[0]):
                img_np = images[i].permute(1, 2, 0).numpy()
                mask_np = masks[i].squeeze(0).numpy()

                augmented = transforms(image=img_np, mask=mask_np)

                aug_img_tensor = torch.from_numpy(augmented["image"]).permute(2, 0, 1).float()
                aug_mask_tensor = torch.from_numpy(augmented["mask"]).unsqueeze(0).float()

                augmented_images.append(aug_img_tensor)
                augmented_masks.append(aug_mask_tensor)

            images = torch.stack(augmented_images).to(device, non_blocking=True)
            masks = torch.stack(augmented_masks).to(device, non_blocking=True)

            # Optimization step
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = criterion(predictions, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{total_batches}] | Current Loss: {loss.item():.4f}")

        epoch_loss = running_loss / total_batches
        print(f"--- Epoch [{epoch+1}/{epochs}] Complete. Mean Training Loss: {epoch_loss:.4f} ---\n")

    # Stop tracking training time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    hours, minutes = divmod(minutes, 60)

    # 6. Performance evaluation on the final batch
    model.eval()
    with torch.no_grad():
        val_acc, val_prec, val_rec, val_f1, val_miou = compute_evaluation_metrics(predictions, masks)

    # 7. Save model and calculate footprint metrics
    weights_output_path = "landslide_unet_weights.pth"
    torch.save(model.to("cpu").state_dict(), weights_output_path)
    
    # Weight file size
    file_size_mb = os.path.getsize(weights_output_path) / (1024 * 1024)

    # Inference speed benchmark
    dummy_input = torch.randn(1, 14, 128, 128).to(device)
    inf_start = time.time()
    with torch.no_grad():
        _ = model.to(device)(dummy_input)
    inf_speed_ms = (time.time() - inf_start) * 1000

    # Peak VRAM allocation
    peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3) if device.type == "cuda" else 0.0

    # 8. Print structural training summary metrics
    print("\n================ TRAINING SUMMARY ================")
    print("Model Architecture: Custom PyTorch U-Net (14 Channels)")
    print(f"Total Training Time: {hours}h {minutes}m {seconds}s ({epochs} Epochs)")
    print("")
    print("--- Evaluation Metrics (Validation Batch Partition) ---")
    print(f"Accuracy:  {val_acc * 100:.1f}%")
    print(f"Precision: {val_prec * 100:.1f}%")
    print(f"Recall:    {val_rec * 100:.1f}%")
    print(f"F1-Score:  {val_f1 * 100:.1f}%")
    print(f"Mean IoU:  {val_miou * 100:.1f}%")
    print("")
    print("--- Operational Footprint ---")
    print(f"Weight File Size:  {file_size_mb:.1f} MB")
    print(f"Inference Speed:   {inf_speed_ms:.1f} ms / image frame")
    if peak_vram_gb > 0:
        print(f"Peak VRAM Usage:   {peak_vram_gb:.1f} GB")
    else:
        print("Peak VRAM Usage:   0.0 GB (Host CPU Execution)")
    print("==================================================\n")


if __name__ == "__main__":
    train_pipeline(epochs=10, batch_size=16)