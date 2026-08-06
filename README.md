# COSMOS-EYE

**Change Observation and Satellite Monitoring Of Slopes — Nepal**

COSMOS-EYE is a self-contained deep-learning pipeline for landslide detection from satellite imagery, built from scratch in PyTorch. It ingests multi-temporal Sentinel-2 and DEM data and produces per-pixel landslide segmentation masks, complete with a bilingual (English/Nepali) interactive dashboard for triage of detected landslide events.

No external cloud APIs are used — the full pipeline (data conversion, training, inference, evaluation, and visualization) runs locally.

---

## Project description

- **Input**: 3 event-aligned temporal windows (PRE / POST / LATE) × 14 bands (10 Sentinel-2 bands, DEM, slope, valid-observation fraction, NDVI) = **42-channel** tiles of shape `(3, 128, 128, 14)`.
- **Model**: custom PyTorch U-Net with SE channel attention, residual blocks, spatial-attention skip connections, deep supervision, and a 16×16 bottleneck.
- **Training**: event-disjoint train/validation split, self-paced label-trust sampling, EMA + SWA weight averaging, weighted BCE + Dice + Tversky loss, 12-view multi-scale test-time augmentation, and an NDVI inference gate that suppresses false positives on vegetated ground.
- **Output**: per-pixel segmentation masks, tuned decision thresholds, per-tile metrics, and a Streamlit dashboard (`app.py`) showing detection overlays, probability maps, and English/Nepali reports.

For full pipeline details (data layout, architecture, loss, sampling, augmentations), see [docs/README.md](docs/README.md).

---

## Team

| Member | Role |
|---|---|
| Rohit Chand | Leader |
| Alish Roka | Team member |
| Diwash Jung Thapa | Team member |
| Sandeep Thapa | Team member |

---

## How to run the code

### 1. Environment setup

Requires Python 3.13+ and (recommended) an NVIDIA GPU with CUDA 12.x.

```bash
# Create and activate a virtual environment
python -m venv cosmoeye_env
cosmoeye_env\Scripts\activate        # Windows
# source cosmoeye_env/bin/activate   # Linux/macOS

# Install PyTorch from the CUDA index first
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132

# Install the remaining dependencies
pip install -r requirements.txt
```

### 2. Quickstart (use the pretrained model — no GPU, no data, no training)

The trained weights and tuned threshold are already included in the repository at `models/landslide_unet_weights.pth` + `models/landslide_best_threshold.json` — no download needed.

Then run the demo immediately:

```bash
streamlit run app.py
```

Upload any Sen12Landslides `.h5` tile in the sidebar, or obtain the dataset as described in step 3.

> A GitHub release + `tools/download_weights.py` are also provided as a fallback if you want to fetch refreshed weights without cloning them (see [Pretrained model](#pretrained-model)).

### 3. Data preparation (only if retraining)

Download the **Sen12Landslides** dataset (see [Data sources](#data-sources)) and convert raw NetCDF tiles into the standardized h5 temporal stacks:

```bash
python src/convert_sen12.py
```

This creates `data/TrainData/img/` + `data/TrainData/mask/` and the sidecar metadata (`tile_metadata.json`, `norm_stats.npz`, `val_indices.json`).

### 4. Train (only if retraining)

```bash
python train.py 20                                  # 20 epochs (default)
python train.py --epochs 20 --batch 16 --lr 2e-4    # explicit hyperparameters
python train.py 20 --no-compile                     # skip torch.compile
```

The best checkpoint is saved to `models/landslide_unet_weights.pth` with its tuned threshold in `models/landslide_best_threshold.json`. Per-epoch history and summaries are written under `results/training_runs/`.

### 5. Run inference / the dashboard

```bash
# Streamlit dashboard (detection overlay, probability map, EN/NE reports)
streamlit run app.py

# CLI inference on a single tile
python src/predict.py <tile_name>

# Batch inference over all tiles
python src/predict.py batch
```

### 6. Evaluate

```bash
python eval_final.py          # reproduce final metrics on the saved checkpoint
python src/confusion_matrix.py
```

---

## Pretrained model

The trained U-Net weights and its tuned threshold are committed directly in the repository (gitignore no longer excludes them), so a fresh clone is immediately runnable:

| Asset | Location |
|---|---|
| `landslide_unet_weights.pth` | Trained model checkpoint (31 MB, base_filters 64, ~8.2M params) |
| `landslide_best_threshold.json` | Tuned detection threshold (0.23) + NDVI-gate settings |
| `models/Old Models/` | Archived checkpoints from earlier runs |

As a fallback, the same files can be fetched from a GitHub Release without cloning the repo:

```bash
python tools/download_weights.py
```

If no release exists yet, publish one from the repository page: **Releases → New release** (tag `v1.0.0`), attach the two files above, and publish.

---

## Data sources

- **Sen12Landslides** — the primary benchmark dataset: Sentinel-2 (10-band) + SRTM DEM over landslide regions, with per-event metadata and pixel-level landslide labels. Each tile carries up to 15 acquisition frames (PRE/POST/LATE event windows).
  - Dataset hub: <https://huggingface.co/datasets/paulhoehn/Sen12Landslides>
  - Paper: Hoehn, P., Heidler, K., Behling, R., Zhu, X. X. (2025). *A Spatio-Temporal Dataset for Satellite-Based Landslide Detection.* Scientific Data 12, 1772. <https://doi.org/10.1038/s41597-025-06167-2>
- **Sentinel-2** — ESA/Copernicus multispectral imagery (used via the Sen12Landslides dataset).
- **SRTM DEM** — digital elevation model (used via the Sen12Landslides dataset).
- **Local reference data** — `data/reference/local_highways.csv`, used by the dashboard for road-blockage proximity alerts (not part of the public dataset).

---

## Attribution

- The **Sen12Landslides dataset** was created by Paul Hoehn, Konrad Heidler, Robert Behling, and Xiao Xiang Zhu (Data Science in Earth Observation, Technical University of Munich / DLR). Please cite their paper when using this project:

  > Hoehn, P., Heidler, K., Behling, R., Zhu, X. X. (2025). *A Spatio-Temporal Dataset for Satellite-Based Landslide Detection.* Scientific Data 12, 1772. https://doi.org/10.1038/s41597-025-06167-2

- **Sentinel-2** data is © ESA/Copernicus, distributed under the Copernicus Programme license terms.
- This project's code and models are developed by the COSMOS-EYE team.

---

## License & disclaimer

This project is a research/triage aid only. Model outputs are a machine-learning screening tool and **must be verified by a qualified geohazard specialist** before any emergency response or infrastructure decisions.
