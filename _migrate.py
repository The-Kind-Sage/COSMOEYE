"""File migration script — moves files into the new structure. Delete after running."""
import os
import shutil

ROOT = "d:/COSMOEYE"


def mv(src, dst):
    s = os.path.join(ROOT, src)
    d = os.path.join(ROOT, dst)
    if not os.path.exists(s):
        print(f"  SKIP (missing): {src}")
        return
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.move(s, d)
    print(f"  MOVED  {src}  ->  {dst}")


# ── Task 2: data/reference ────────────────────────────────────────────────
mv("local_highways.csv",                 "data/reference/local_highways.csv")

# ── Task 3: models/ ──────────────────────────────────────────────────────
mv("landslide_unet_weights.pth",         "models/landslide_unet_weights.pth")
mv("landslide_unet_weights.prev.pth",    "models/landslide_unet_weights.prev.pth")
mv("landslide_unet_weights.refine3ep.pth","models/landslide_unet_weights.refine3ep.pth")
mv("landslide_best_threshold.json",      "models/landslide_best_threshold.json")
mv("landslide_best_threshold.prev.json", "models/landslide_best_threshold.prev.json")

# ── Task 4: src/ ──────────────────────────────────────────────────────────
for module in [
    "model.py", "dataset.py", "augmentations.py", "spatial_math.py",
    "predict.py", "generate_prompt.py", "generate_reports.py",
    "convert_sen12.py", "geo_lookup.py", "custom_vocabulary.py",
    "confusion_matrix.py", "probe_weights.py", "test_gpu.py",
]:
    mv(module, f"src/{module}")

# ── Task 5: results/ ─────────────────────────────────────────────────────
# training_runs subtree
for item in os.listdir(os.path.join(ROOT, "training_runs")):
    src_path = f"training_runs/{item}"
    dst_path = f"results/training_runs/{item}"
    mv(src_path, dst_path)
try:
    os.rmdir(os.path.join(ROOT, "training_runs"))
    print("  RMDIR  training_runs")
except Exception as e:
    print(f"  RMDIR training_runs failed: {e}")

# per_image_val_results -> results/per_image_val
for item in os.listdir(os.path.join(ROOT, "per_image_val_results")):
    mv(f"per_image_val_results/{item}", f"results/per_image_val/{item}")
try:
    os.rmdir(os.path.join(ROOT, "per_image_val_results"))
    print("  RMDIR  per_image_val_results")
except Exception as e:
    print(f"  RMDIR per_image_val_results failed: {e}")

# prediction_overlays -> results/prediction_overlays
for item in os.listdir(os.path.join(ROOT, "prediction_overlays")):
    mv(f"prediction_overlays/{item}", f"results/prediction_overlays/{item}")
try:
    os.rmdir(os.path.join(ROOT, "prediction_overlays"))
    print("  RMDIR  prediction_overlays")
except Exception as e:
    print(f"  RMDIR prediction_overlays failed: {e}")

# metrics: CSVs, PNGs, NPY
for f in [
    "confusion_matrices_tiles.csv",
    "confusion_matrix_aggregate.png",
    "confusion_matrix_tiles_grid.png",
    "confusion_matrix_tile_errors.png",
    "final_results.csv",
    "per_image_predictions.csv",
    "train_history.csv",
    "predicted_mask_image_1.npy",
    "vocabulary.json",
]:
    mv(f, f"results/metrics/{f}")

# ── Task 6: docs/ ────────────────────────────────────────────────────────
mv("README.md",          "docs/README.md")
mv("emergency_bulletin.txt", "docs/emergency_bulletin.txt")

print("\nMigration complete.")
