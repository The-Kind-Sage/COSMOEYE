# COSMOEYE

Self-contained landslide detection pipeline for the **Sen12Landslides** benchmark dataset, built from scratch in PyTorch (no external cloud APIs). Takes Sentinel-2 + DEM temporal satellite imagery and produces per-pixel landslide segmentation masks.

---

## Project status

- Dataset: **Sen12Landslides** (13,628 local tiles; Sentinel-2 10-band + DEM, 15 acquisition frames per tile)
- Model: custom U-Net, **18.3M params** — SE channel attention, residual blocks, spatial-attention skip connections, deep supervision, 3-level encoder (bottleneck at 16x16)
- Input: 3 event-aligned temporal windows x 14 bands = **42 channels**
- Training: event-disjoint split, self-paced label-trust sampling, EMA + SWA, Tversky + weighted BCE/Dice loss, 12-view multi-scale TTA, NDVI inference gate

---

## Data layout

Raw NetCDF tiles (`datasets/s2/`) are converted to standardized temporal-stack h5 files (`datasets/TrainData/img/` + `mask/`) with `convert_sen12.py`:

- Each tile is `(3, 128, 128, 14)`: three temporal windows
  - `W1 = PRE` (median of pre-event frames, aligned to per-tile event metadata)
  - `W2 = POST` (median of post-event frames)
  - `W3 = LATE` (frame 14, regrowth reference)
  - Fallback windows `(0-7), (8-13), (14,14)` for tiles without annotations
- Per-window 14-band layout: `0-9` = B02..B12, `10` = DEM (clipped 0-8800), `11` = slope (deg), `12` = reserved, `13` = NDVI
- Flattened to 42 network channels: `[PRE 0-13 | POST 14-27 | LATE 28-41]`
- Sidecar files in `datasets/TrainData/`:
  - `tile_metadata.json` — annotated flag, event date, annotation id, `pre_post` frame indices for all 13,628 tiles (6,739 annotated)
  - `norm_stats.npz` — per-channel z-score statistics (14,)
  - `pos_frac_cache.npy` — per-tile positive-pixel fractions
  - `val_indices.json` — persisted event-disjoint split (same val set every run)

---

## Architecture (`model.py`)

`CustomUNet(in_channels=42, out_channels=1, base_filters=96)`:

- 3 downsampling levels (keeps a 16x16 bottleneck so small landslide blobs survive)
- Every block: `Conv3x3(BN,ReLU) x2` with reflect padding, **SE channel attention**, 1x1 residual shortcut
- **Spatial attention** on skip connections (CBAM-style)
- Bottleneck dropout (0.1) against noisy labels
- **Deep supervision**: auxiliary 1x1 heads on decoder levels 2 and 3 (`model(x, return_aux=True)`)
- Output heads are zero-initialized (stable early training)

---

## Training (`train.py`)

```
python train.py 20              # positional epochs (default 20)
python train.py --epochs 20 --batch 16 --lr 1e-3 --pos-weight 6.0
python train.py 20 --no-compile # skip torch.compile attempt
```

### Split
- **Event-disjoint**: tiles of the same landslide event (region + event date) never straddle train/val; persisted to `val_indices.json`
- Validation positive tiles **capped at 65%** (background tiles included) so threshold tuning is honest

### Sampling
- `EventGroupedSampler`: draws whole events weighted by positive content; background tiles get a 15% draw-mass floor; 20% of draws forced from the richest events; per-tile self-paced trust (refreshed every 2 epochs) down-weights noisy labels

### Loss
`WeightedBCEDiceLoss`: label-trust-weighted BCE (`pos_weight` 6.0, ramping to ~3.5) + weighted Dice (1.5) + Tversky (0.3, alpha=0.3/beta=0.7, precision-leaning) + deep-supervision aux loss (0.4) + label smoothing (0.05)
- Per-pixel label-trust weights (from post-event NDVI): noisy positive (over-cover) 0.40, suspicious negative 0.65, boundary ring x1.5

### Augmentations (`augmentations.py`)
Flips/rot90, affine scale 0.9-1.1, coarse hole-dropout (cloud simulation), random spectral band dropout, per-window acquisition jitter, brightness/contrast, gauss noise/blur. Zeros = z-score mean ("no information").

### Optimizer / schedule
- AdamW (lr 1e-3, weight decay 1e-4), OneCycleLR (30% warmup), gradient clipping 1.0, AMP fp16 + TF32
- EMA of weights **and** BN running stats (decay ramps 0.99 -> 0.999 over 5 epochs)
- SWA: last 5 EMA snapshots averaged; kept as final weights if it beats the best EMA checkpoint

### Evaluation
- Per-epoch: EMA weights, 2-view flip TTA, threshold grid 0.02
- Final summary: best checkpoint, **12-view multi-scale TTA** (0.9/1.0/1.1 x 4 flips), threshold grid 0.01
- **NDVI gate** (primary metric): predictions suppressed where post-event NDVI >= 0.48 (landslides are bare ground; vegetated detections are cloud/water/regrowth false positives). Ungated numbers reported alongside
- `train_history.csv` logs every epoch; best checkpoint saved to `landslide_unet_weights.pth` + `landslide_best_threshold.json`

---

## Inference (`predict.py`)

Loads `landslide_unet_weights.pth`, normalizes with cached stats, applies the tuned threshold. Supports the temporal-stack (N,H,W,14) layout and legacy single-frame inputs.

---

## Utilities

- `convert_sen12.py` — raw .nc -> temporal-stack h5 conversion; builds `tile_metadata.json`
- `geo_lookup.py` — UTM tile coordinates -> WGS84 (pure math, no pyproj) with Google Maps links; exports `datasets/TrainData/tile_coordinates.csv`

---

## Environment

- Python 3.13.14, PyTorch 2.13.0+cu132, CUDA 12.x
- `albumentations 2.0.8`, `opencv-python 5.x`, `h5py`, `numpy`
- Active venv: `D:\COSMOEYE\cosmoeye_env`
- torch.compile is skipped automatically when Triton is unavailable (Python 3.13 Windows)

---

## Results so far

Latest pipeline, 1 epoch (honest 65/35 val split):

| Metric | NDVI-gated | Raw |
|---|---|---|
| Precision | 22.6% | 28.6% |
| Recall | 30.1% | 42.4% |
| F1 | 25.8% | 34.2% |
| IoU | 14.8% | 20.6% |
| Threshold | 0.45 | 0.69 |

A 20-epoch run is the current training target (~4 min/epoch).

---

## References

- **Sen12Landslides**: Hoehn, P., Heidler, K., Behling, R., Zhu, X. X. (2025). *A Spatio-Temporal Dataset for Satellite-Based Landslide Detection*. Scientific Data 12, 1772. https://doi.org/10.1038/s41597-025-06167-2
- Dataset hub: https://huggingface.co/datasets/paulhoehn/Sen12Landslides
- Official benchmark (raw labels, S2+DEM, `S12LS-LD` split): U-ConvLSTM F1 0.63 / IoU 0.46; with **refined labels** F1 > 0.83 (Sentinel-2)

### Roadmap

1. Finish 20-epoch baseline (in progress)
2. Label-refinement loop (train -> predict -> merge confident predictions + NDVI heuristics into masks -> retrain; largest documented lever, ~+25-30 IoU)
3. Full 15-frame temporal modeling (U-ConvLSTM-style) for comparability with the official benchmark
