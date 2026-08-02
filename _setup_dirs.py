"""One-shot directory scaffolding script — delete after running."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

dirs = [
    "data/raw/s2",
    "data/TrainData/img",
    "data/TrainData/mask",
    "data/TestData/img",
    "data/TestData/mask",
    "data/reference",
    "src",
    "models",
    "results/training_runs",
    "results/per_image_val",
    "results/prediction_overlays",
    "results/metrics",
    "notebooks",
    "docs",
]

for d in dirs:
    p = os.path.join(ROOT, d)
    os.makedirs(p, exist_ok=True)
    # Place .gitkeep so git tracks empty dirs
    gk = os.path.join(p, ".gitkeep")
    if not any(f for f in os.listdir(p) if f != ".gitkeep"):
        open(gk, "w").close()
    print(f"OK  {p}")

print("\nAll directories created.")
