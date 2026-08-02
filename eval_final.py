import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import csv
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model import CustomUNet
from dataset import (LandslideDataset, NDVI_CHANNEL,
                     normalize_standardized_image, stack_temporal_stacks)
from train import evaluate_dataset, WeightedBCEDiceLoss
from predict import tta_probabilities, PIXEL_AREA_SQM
from paths import PATHS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_img_dir  = PATHS.TRAIN_IMG_DIR
train_mask_dir = PATHS.TRAIN_MASK_DIR
split_path     = PATHS.VAL_INDICES
weights_path   = PATHS.WEIGHTS
threshold_path = PATHS.THRESHOLD_JSON

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

from predict import infer_base_filters
state = torch.load(weights_path, map_location=device, weights_only=True)
base_filters = infer_base_filters(state)
model = CustomUNet(in_channels=42, out_channels=1, base_filters=base_filters).to(device)
model.load_state_dict(state)
model.eval()
criterion = WeightedBCEDiceLoss().to(device)

final_metrics = evaluate_dataset(
    model, val_loader, criterion, device,
    thresholds=np.arange(0.01, 0.99, 0.01),
    scales=(0.9, 1.0, 1.1),
    ndvi_gate=False, ndvi_mean=ndvi_mean, ndvi_std=ndvi_std,
)

print("\n================ TRAINING SUMMARY (reproduced on saved checkpoint) ================")
print("Model Architecture: Custom PyTorch U-Net (42 Channels Temporal 3-Window, 3-Level)")
print("                    SE + Residual + Skip Spatial-Attention + Deep Supervision")
print("")
print("--- Evaluation Metrics (Held-Out Validation Split, Raw/Ungated, Tuned Threshold, 12-view TTA) ---")
print(f"Best Threshold: {final_metrics['threshold']:.2f}")
print(f"Accuracy:  {final_metrics['accuracy'] * 100:.1f}%")
print(f"Precision: {final_metrics['precision'] * 100:.1f}%")
print(f"Recall:    {final_metrics['recall'] * 100:.1f}%")
print(f"F1-Score:  {final_metrics['f1'] * 100:.1f}%")
print(f"Mean IoU:  {final_metrics['iou'] * 100:.1f}%  <-- Primary validation track (Raw/Ungated)")
print(f"Precision at recall >= 60%: {final_metrics['precision_at_recall'] * 100:.1f}% "
      f"(threshold {final_metrics['threshold_at_recall']:.2f})")
print(f"NDVI-gated reference: P {final_metrics['raw_precision'] * 100:.1f}% | "
      f"R {final_metrics['raw_recall'] * 100:.1f}% | F1 {final_metrics['raw_f1'] * 100:.1f}% | "
      f"IoU {final_metrics['raw_iou'] * 100:.1f}% @ {final_metrics['raw_threshold']:.2f}")
print("================================================================================")

# ---------------------------------------------------------------- per-image pass
with open(threshold_path, "r") as jf:
    saved_threshold = json.load(jf)
threshold = saved_threshold.get("threshold", 0.5)

rows = []
per_img_dir = PATHS.PER_IMAGE_VAL_DIR
os.makedirs(per_img_dir, exist_ok=True)

with torch.no_grad():
    for local_i, global_i in enumerate(val_indices):
        fname = full_dataset.file_list[global_i]
        sample = full_dataset[global_i]
        images, masks, _ = sample
        images = images.unsqueeze(0).to(device)

        probs = tta_probabilities(model, images, device).squeeze().cpu().numpy()

        # Per-image pass uses raw (ungated) probabilities to match the primary
        # evaluation metric tracked during training (ndvi_gate=False).
        pred = (probs > threshold).astype(np.uint8)
        gt = (masks.squeeze().numpy() > 0.5).astype(np.uint8)

        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())

        iou = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        gt_px = int(gt.sum())
        pred_px = int(pred.sum())
        rows.append([
            fname, pred_px, gt_px, tp, fp, fn,
            f"{iou:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}",
            f"{pred_px * PIXEL_AREA_SQM:,.0f}",
            f"{gt_px * PIXEL_AREA_SQM:,.0f}",
        ])

per_img_csv = PATHS.PER_IMAGE_VAL_CSV
with open(per_img_csv, "w", newline="") as cf:
    writer = csv.writer(cf)
    writer.writerow(["image", "pred_pixels", "gt_pixels", "tp", "fp", "fn",
                     "iou", "precision", "recall", "f1",
                     "pred_area_sqm", "gt_area_sqm"])
    writer.writerows(rows)

detected = [r for r in rows if int(r[1]) > 0]
hits = [r for r in rows if float(r[6]) > 0.0]
print(f"\nPer-image results saved: {per_img_csv}")
print(f"  Images with any prediction : {len(detected)}/{len(rows)}")
print(f"  Images with IoU > 0        : {len(hits)}/{len(rows)}")

# Top-10 tiles by IoU (the clearest landslide insights)
print("\nTop 10 val tiles by IoU (model confidently finds the landslide):")
print(f"{'image':<42} {'pred_px':>8} {'gt_px':>7} {'IoU':>6} {'P':>6} {'R':>6} {'F1':>6}")
for r in sorted(rows, key=lambda r: -float(r[6]))[:10]:
    print(f"{r[0]:<42} {int(r[1]):>8,} {int(r[2]):>7,} "
          f"{float(r[6]):>6.3f} {float(r[7]):>6.3f} {float(r[8]):>6.3f} {float(r[9]):>6.3f}")

print("\n10 largest detections by predicted area:")
for r in sorted(rows, key=lambda r: -int(r[1]))[:10]:
    print(f"{r[0]:<42} pred {r[10]:>10} m2 | gt {r[11]:>10} m2 | IoU {float(r[6]):.3f}")

results_path = PATHS.FINAL_RESULTS_CSV
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
