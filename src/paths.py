"""Central path configuration for the COSMOEYE project.

Every module imports paths from here instead of hard-coding strings.
All paths are absolute, derived from the project root (the directory
that contains this src/ folder), so the project works regardless of
the current working directory when Python is invoked.

Usage
-----
    from src.paths import PATHS          # when running from project root
    # or, inside src/ modules:
    from paths import PATHS
"""
import os
import glob

# Project root = parent of this file's directory (i.e. d:/COSMOEYE)
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _p(*parts):
    """Join parts under ROOT and normalise separators."""
    return os.path.normpath(os.path.join(ROOT, *parts))


class PATHS:
    # ── Root ──────────────────────────────────────────────────────────────
    ROOT = ROOT

    # ── Data ──────────────────────────────────────────────────────────────
    DATA_DIR            = _p("data")
    RAW_S2_DIR          = _p("data", "raw", "s2")

    TRAIN_IMG_DIR       = _p("data", "TrainData", "img")
    TRAIN_MASK_DIR      = _p("data", "TrainData", "mask")
    TRAIN_DATA_DIR      = _p("data", "TrainData")

    TEST_IMG_DIR        = _p("data", "TestData", "img")
    TEST_MASK_DIR       = _p("data", "TestData", "mask")
    TEST_DATA_DIR       = _p("data", "TestData")

    REFERENCE_DIR       = _p("data", "reference")
    HIGHWAYS_CSV        = _p("data", "reference", "local_highways.csv")

    # Cached dataset artefacts (live inside TrainData alongside the tiles)
    NORM_STATS          = _p("data", "TrainData", "norm_stats.npz")
    VAL_INDICES         = _p("data", "TrainData", "val_indices.json")
    TILE_METADATA       = _p("data", "TrainData", "tile_metadata.json")
    TILE_COORDINATES    = _p("data", "TrainData", "tile_coordinates.csv")
    POS_FRAC_CACHE      = _p("data", "TrainData", "pos_frac_cache.npy")

    # ── Models ────────────────────────────────────────────────────────────
    MODELS_DIR          = _p("models")
    WEIGHTS             = _p("models", "landslide_unet_weights.pth")
    WEIGHTS_PREV        = _p("models", "landslide_unet_weights.prev.pth")
    THRESHOLD_JSON      = _p("models", "landslide_best_threshold.json")
    THRESHOLD_JSON_PREV = _p("models", "landslide_best_threshold.prev.json")

    # Optional offline NLP model (T5)
    T5_TOKENIZER        = _p("models", "t5_tokenizer")
    T5_WEIGHTS          = _p("models", "t5_weights")

    # ── Results ───────────────────────────────────────────────────────────
    RESULTS_DIR             = _p("results")
    TRAINING_RUNS_DIR       = _p("results", "training_runs")
    RUNS_SUMMARY            = _p("results", "training_runs", "runs_summary.csv")
    PER_IMAGE_VAL_DIR       = _p("results", "per_image_val")
    PER_IMAGE_VAL_CSV       = _p("results", "per_image_val", "per_image_val_results.csv")
    PREDICTION_OVERLAYS_DIR = _p("results", "prediction_overlays")
    METRICS_DIR             = _p("results", "metrics")
    FINAL_RESULTS_CSV       = _p("results", "metrics", "final_results.csv")
    PER_IMAGE_PREDS_CSV     = _p("results", "metrics", "per_image_predictions.csv")

    # ── Docs ──────────────────────────────────────────────────────────────
    DOCS_DIR            = _p("docs")
    EMERGENCY_BULLETIN  = _p("docs", "emergency_bulletin.txt")

    # ── Notebooks ─────────────────────────────────────────────────────────
    NOTEBOOKS_DIR       = _p("notebooks")

    # ── Source ────────────────────────────────────────────────────────────
    SRC_DIR             = _p("src")


# -- Latest-model resolution ------------------------------------------------
#
# Model training writes to the canonical PATHS.WEIGHTS file and archives any
# previous weights to PATHS.WEIGHTS_PREV.  Separate experimental checkpoints
# (e.g. landslide_unet_weights.refine3ep.pth) can also exist in models/.
# Inference, the dashboard, the confusion matrix and the evaluation script all
# want the LATEST usable checkpoint, so instead of hard-coding the canonical
# filename each consumer resolves the newest non-.prev .pth under models/.
# "Latest" is determined by file modification time: whichever checkpoint was
# written most recently is the one that should be live.  .prev.pth archives are
# always excluded so a rollback file is never mistaken for the live model.


def _candidate_weights():
    """All .pth checkpoint paths under models/, excluding .prev.pth archives."""
    return [p for p in glob.glob(os.path.join(PATHS.MODELS_DIR, "*.pth"))
            if not p.endswith(".prev.pth")]


def get_latest_weights_path():
    """Return the most recently written (by mtime) non-archive .pth in models/.

    Falls back to the canonical PATHS.WEIGHTS when no candidates exist, so the
    rest of the pipeline keeps working even if models/ is empty (predict.py
    raises its usual "run train.py first" error in that case).
    """
    candidates = _candidate_weights()
    if not candidates:
        return PATHS.WEIGHTS
    return max(candidates, key=os.path.getmtime)


def get_latest_threshold_path():
    """Return the threshold JSON that belongs to the latest weights file.

    The canonical threshold file (landslide_best_threshold.json) is rewritten
    whenever the canonical weights are retrained, so for the canonical weights
    it is always in sync.  For a non-canonical latest weights file (e.g. a
    refine3ep experiment) the canonical threshold is still the closest tuning
    available, so it is returned as the fallback.
    """
    latest_weights = get_latest_weights_path()
    base = os.path.splitext(latest_weights)[0]
    candidate = f"{base}.json"
    if os.path.exists(candidate):
        return candidate
    return PATHS.THRESHOLD_JSON


if __name__ == "__main__":
    import inspect
    print("COSMOEYE project paths")
    print(f"  Root: {PATHS.ROOT}\n")
    for name, val in inspect.getmembers(PATHS):
        if not name.startswith("_") and isinstance(val, str):
            exists = "OK " if os.path.exists(val) else "-- "
            print(f"  {exists} {name:<28} {val}")
    print(f"\nLatest weights : {get_latest_weights_path()}")
    print(f"Latest threshold: {get_latest_threshold_path()}")