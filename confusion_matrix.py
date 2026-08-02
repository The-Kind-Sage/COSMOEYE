import os
import csv
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import CustomUNet
from dataset import LandslideDataset, NDVI_CHANNEL
from train import tta_probabilities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_img_dir = "./datasets/TrainData/img/"
train_mask_dir = "./datasets/TrainData/mask/"
split_path = "./datasets/TrainData/val_indices.json"

weights_path = "landslide_unet_weights.pth"
if not os.path.exists(weights_path):
    weights_path = "landslide_unet_weights.prev.pth"

gate_enabled = True
gate_threshold = 0.48
threshold = 0.5
threshold_path = "landslide_best_threshold.json"
if not os.path.exists(threshold_path):
    threshold_path = "landslide_best_threshold.prev.json"
if os.path.exists(threshold_path):
    with open(threshold_path, "r") as jf:
        saved = json.load(jf)
    threshold = saved.get("threshold", 0.5)
    gate_enabled = saved.get("ndvi_gate", True)
    gate_threshold = saved.get("ndvi_gate_threshold", 0.48)

full_dataset = LandslideDataset(img_dir=train_img_dir, mask_dir=train_mask_dir)

with open(split_path, "r") as jf:
    split_data = json.load(jf)
val_indices = split_data["val"]
print(f"Val tiles: {len(val_indices)}")

val_dataset = Subset(
    LandslideDataset(img_dir=train_img_dir, mask_dir=train_mask_dir),
    val_indices,
)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

ndvi_mean = float(full_dataset.norm_mean[NDVI_CHANNEL])
ndvi_std = float(full_dataset.norm_std[NDVI_CHANNEL])

model = CustomUNet(in_channels=42, out_channels=1).to(device)
model.load_state_dict(
    torch.load(weights_path, map_location=device, weights_only=True)
)
model.to(device).eval()

results = []
with torch.no_grad():
    for images, masks, weights in val_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        probs = tta_probabilities(model, images, device, n_views=4, scales=(1.0,))
        if gate_enabled:
            raw_ndvi = images[:, 27:28] * ndvi_std + ndvi_mean
            probs = probs * (raw_ndvi < gate_threshold).float()
        pred = (probs > threshold).float()
        target = (masks > 0.5).float()
        for b in range(images.size(0)):
            p = pred[b].reshape(-1).cpu().numpy()
            t = target[b].reshape(-1).cpu().numpy()
            tp = float(((p > 0.5) & (t > 0.5)).sum())
            fp = float(((p > 0.5) & (t <= 0.5)).sum())
            fn = float(((p <= 0.5) & (t > 0.5)).sum())
            tn = float(((p <= 0.5) & (t <= 0.5)).sum())
            iou = tp / (tp + fp + fn + 1e-8)
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            results.append({
                "tile": val_dataset.dataset.file_list[val_dataset.indices[b]],
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "iou": iou, "precision": prec, "recall": rec, "f1": f1,
            })

totals = {k: sum(r[k] for r in results) for k in ("tp", "fp", "fn", "tn")}
n = totals["tp"] + totals["fp"] + totals["fn"] + totals["tn"]
a_iou = totals["tp"] / (totals["tp"] + totals["fp"] + totals["fn"] + 1e-8)
a_prec = totals["tp"] / (totals["tp"] + totals["fp"] + 1e-8)
a_rec = totals["tp"] / (totals["tp"] + totals["fn"] + 1e-8)
a_f1 = 2 * a_prec * a_rec / (a_prec + a_rec + 1e-8)
a_acc = (totals["tp"] + totals["tn"]) / n

print("================ AGGREGATE CONFUSION MATRIX (val pixels) ================")
print(f"Threshold: {threshold:.2f} | NDVI gate: {gate_enabled} @ {gate_threshold:.2f}")
print(f"TN {int(totals['tn']):>9} | FP {int(totals['fp']):>9}")
print(f"FN {int(totals['fn']):>9} | TP {int(totals['tp']):>9}")
print(f"Accuracy {a_acc * 100:.1f}% | Precision {a_prec * 100:.1f}% | "
      f"Recall {a_rec * 100:.1f}% | F1 {a_f1 * 100:.1f}% | IoU {a_iou * 100:.1f}%")
print("========================================================================")

csv_path = "confusion_matrices_tiles.csv"
with open(csv_path, "w", newline="") as cf:
    writer = csv.writer(cf)
    writer.writerow(["tile", "tp", "fp", "fn", "tn", "iou", "precision", "recall", "f1"])
    for r in sorted(results, key=lambda x: x["fp"] + x["fn"], reverse=True):
        writer.writerow([r["tile"], int(r["tp"]), int(r["fp"]), int(r["fn"]),
                         int(r["tn"]), f"{r['iou']:.4f}", f"{r['precision']:.4f}",
                         f"{r['recall']:.4f}", f"{r['f1']:.4f}"])
print(f"Per-tile table written to {csv_path}")


def draw_matrix(ax, cm, title, totals_n=None):
    ax.imshow([[cm[0], cm[1]], [cm[2], cm[3]]], cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            v = cm[i * 2 + j]
            pct = 100.0 * v / totals_n if totals_n else 0.0
            ax.text(j, i, f"{int(v):,}\n({pct:.2f}%)", ha="center", va="center",
                    fontsize=11, color="white" if v > 0.5 * max(cm) else "black")
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=9)


agg_fig, agg_ax = plt.subplots(figsize=(6.5, 5.5))
draw_matrix(agg_ax, [totals["tn"], totals["fp"], totals["fn"], totals["tp"]], "", totals_n=n)
agg_ax.set_title(
    f"Aggregate Confusion Matrix - Validation Set ({len(results)} tiles)\n"
    f"Threshold {threshold:.2f} + NDVI gate | Acc {a_acc * 100:.1f}% | P {a_prec * 100:.1f}% | "
    f"R {a_rec * 100:.1f}% | F1 {a_f1 * 100:.1f}% | IoU {a_iou * 100:.1f}%",
    fontsize=11,
)
agg_fig.tight_layout()
agg_fig.savefig("confusion_matrix_aggregate.png", dpi=150)
print("Saved confusion_matrix_aggregate.png")

top_tiles = sorted(results, key=lambda x: x["fp"] + x["fn"], reverse=True)[:12]
n_cols, n_rows = 4, 3
grid_fig, grid_axes = plt.subplots(n_rows, n_cols, figsize=(19, 13))
for ax, r in zip(grid_axes.ravel(), top_tiles):
    draw_matrix(ax, [r["tn"], r["fp"], r["fn"], r["tp"]], "", totals_n=r["tp"] + r["fp"] + r["fn"] + r["tn"])
    ax.set_title(f"{os.path.basename(r['tile'])}\nIoU {r['iou'] * 100:.1f}% | P {r['precision'] * 100:.1f}% | "
                 f"R {r['recall'] * 100:.1f}% | errors {int(r['fp'] + r['fn'])}", fontsize=9)
for ax in grid_axes.ravel()[len(top_tiles):]:
    ax.axis("off")
grid_fig.suptitle("Per-Tile Confusion Matrices - Top 12 Tiles by Pixel Error (FP + FN)",
                  fontsize=13, y=0.995)
grid_fig.tight_layout()
grid_fig.savefig("confusion_matrix_tiles_grid.png", dpi=150)
print("Saved confusion_matrix_tiles_grid.png")

err_tiles = sorted(results, key=lambda x: x["fp"] + x["fn"], reverse=True)[:40]
fig, ax = plt.subplots(figsize=(13, max(6, 0.28 * len(err_tiles))))
idx = np.arange(len(err_tiles))
labels = [os.path.basename(r["tile"]) for r in err_tiles]
ax.barh(idx, [r["fp"] for r in err_tiles], color="#f4a582", label="FP (false alarm)")
ax.barh(idx, [r["fn"] for r in err_tiles], left=[r["fp"] for r in err_tiles],
        color="#d73027", label="FN (missed)")
ax.barh(idx, [r["tp"] for r in err_tiles], left=[r["fp"] + r["fn"] for r in err_tiles],
        color="#92c5de", label="TP (hit)")
ax.set_yticks(idx)
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Pixels")
ax.set_title(f"Per-Tile Pixel Breakdown (TP / FP / FN) - worst 40 tiles @ thr {threshold:.2f}", fontsize=11)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("confusion_matrix_tile_errors.png", dpi=150)
print("Saved confusion_matrix_tile_errors.png")
