import os
import csv
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model import CustomUNet
from dataset import LandslideDataset, NDVI_CHANNEL
from train import evaluate_dataset, WeightedBCEDiceLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_img_dir = "./datasets/TrainData/img/"
train_mask_dir = "./datasets/TrainData/mask/"
split_path = "./datasets/TrainData/val_indices.json"
weights_path = "landslide_unet_weights.pth"

full_dataset = LandslideDataset(img_dir=train_img_dir, mask_dir=train_mask_dir)

with open(split_path, "r") as jf:
    split_data = json.load(jf)
val_indices = split_data["val"]
print(f"Val tiles: {len(val_indices)}")

val_dataset = Subset(
    LandslideDataset(img_dir=train_img_dir, mask_dir=train_mask_dir),
    val_indices,
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

ndvi_mean = float(full_dataset.norm_mean[NDVI_CHANNEL])
ndvi_std = float(full_dataset.norm_std[NDVI_CHANNEL])

model = CustomUNet(in_channels=42, out_channels=1).to(device)
model.load_state_dict(
    torch.load(weights_path, map_location=device, weights_only=True)
)
model.to(device).eval()
criterion = WeightedBCEDiceLoss().to(device)

final_metrics = evaluate_dataset(
    model, val_loader, criterion, device,
    thresholds=np.arange(0.01, 0.99, 0.01),
    scales=(0.9, 1.0, 1.1),
    ndvi_gate=True, ndvi_mean=ndvi_mean, ndvi_std=ndvi_std,
)

print("\n================ TRAINING SUMMARY (reproduced on saved checkpoint) ================")
print("Model Architecture: Custom PyTorch U-Net (42 Channels Temporal 3-Window, 3-Level)")
print("                    SE + Residual + Skip Spatial-Attention + Deep Supervision")
print("")
print("--- Evaluation Metrics (Held-Out Validation Split, NDVI-Gated, Tuned Threshold, 12-view TTA) ---")
print(f"Best Threshold: {final_metrics['threshold']:.2f}")
print(f"Accuracy:  {final_metrics['accuracy'] * 100:.1f}%")
print(f"Precision: {final_metrics['precision'] * 100:.1f}%")
print(f"Recall:    {final_metrics['recall'] * 100:.1f}%")
print(f"F1-Score:  {final_metrics['f1'] * 100:.1f}%")
print(f"Mean IoU:  {final_metrics['iou'] * 100:.1f}%  <-- Best validation track (NDVI-gated)")
print(f"Precision at recall >= 60%: {final_metrics['precision_at_recall'] * 100:.1f}% "
      f"(threshold {final_metrics['threshold_at_recall']:.2f})")
print(f"Raw (ungated) reference: P {final_metrics['raw_precision'] * 100:.1f}% | "
      f"R {final_metrics['raw_recall'] * 100:.1f}% | F1 {final_metrics['raw_f1'] * 100:.1f}% | "
      f"IoU {final_metrics['raw_iou'] * 100:.1f}% @ {final_metrics['raw_threshold']:.2f}")
print("================================================================================")

results_path = "final_results.csv"
file_exists = os.path.exists(results_path)
with open(results_path, "a", newline="") as cf:
    writer = csv.writer(cf)
    if not file_exists:
        writer.writerow([
            "run_id", "val_tiles", "threshold", "accuracy", "precision",
            "recall", "f1", "iou", "precision_at_recall60", "raw_precision",
            "raw_recall", "raw_f1", "raw_iou", "raw_threshold", "weights_path",
        ])
    writer.writerow([
        time.strftime("%Y-%m-%d_%H-%M-%S"), len(val_indices),
        f"{final_metrics['threshold']:.2f}",
        f"{final_metrics['accuracy'] * 100:.1f}",
        f"{final_metrics['precision'] * 100:.1f}",
        f"{final_metrics['recall'] * 100:.1f}",
        f"{final_metrics['f1'] * 100:.1f}",
        f"{final_metrics['iou'] * 100:.1f}",
        f"{final_metrics['precision_at_recall'] * 100:.1f}",
        f"{final_metrics['raw_precision'] * 100:.1f}",
        f"{final_metrics['raw_recall'] * 100:.1f}",
        f"{final_metrics['raw_f1'] * 100:.1f}",
        f"{final_metrics['raw_iou'] * 100:.1f}",
        f"{final_metrics['raw_threshold']:.2f}",
        weights_path,
    ])
print(f"Final results appended to {results_path}")
