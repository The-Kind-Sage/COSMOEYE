import os
import gzip
import tempfile
import h5py
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Available variables inside each Sen12Landslides .nc file:
# B02..B08, B8A, B11, B12, SCL, MASK, DEM  (15 time steps each for bands)
BAND_VARS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
DEM_VAR = "DEM"
# Temporal formulation: three robust temporal windows per tile, each the
# median of its frames (median = cloud-resistant). Annotated tiles carry
# per-tile event metadata (pre_post_dates = acquisition frame indices that
# bracket the event), so windows are aligned to the TRUE event timing:
#   W1 PRE  = median(frames 0..pre)          clean pre-event baseline
#   W2 POST = median(frames post..13)        sustained post-event bare state
#   W3 LATE = frame 14                       regrowth reference
# Unannotated tiles fall back to the population median event timing
# (event frame ~8): PRE=(0-7), POST=(8-13), LATE=(14).
# Verified on 71 annotated tiles: the mask region stays NDVI-depressed
# (bare) from the event frame through frame ~13, then regrows.
ANNOTATED_PRE_END = None      # per-tile from metadata
ANNOTATED_POST_START = None   # per-tile from metadata
FALLBACK_WINDOWS = [(0, 7), (8, 13), (14, 14)]
NDVI_CHANNEL = 13
NORM_STATS_PATH = "./datasets/TrainData/norm_stats.npz"
DEM_MIN, DEM_MAX = 0.0, 8800.0  # harmonized dataset convention
METADATA_JSON_PATH = "./datasets/TrainData/tile_metadata.json"


def parse_pre_post_dates(value):
    """Robustly parse the pre_post_dates attribute into (pre, post) indices."""
    try:
        import ast
        pp = ast.literal_eval(value)
        while not isinstance(pp, dict):
            pp = pp[0]
        return int(pp["pre"]), int(pp["post"])
    except Exception:
        return None


def resolve_windows(annotated, pre_post):
    """Return the three (f0, f1) frame windows for a tile."""
    if annotated and pre_post is not None:
        pre, post = pre_post
        pre = max(0, min(pre, 14))
        post = max(0, min(post, 14))
        return [(0, pre), (post, 13), (14, 14)]
    return FALLBACK_WINDOWS


def compute_slope(dem, pixel_size=10.0):
    """Terrain slope in degrees from the DEM (landslides favor steep slopes)."""
    if float(np.nanstd(dem)) < 1e-6:
        return np.zeros_like(dem, dtype=np.float32)
    dz_dx, dz_dy = np.gradient(dem, pixel_size)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
    return np.where(np.isfinite(slope), slope, 0.0).astype(np.float32)


def convert_one_file(full_path, img_out_dir, mask_out_dir):
    """Convert a single raw .nc tile into a standardized temporal-stack .h5 file.

    Output image layout: (3, 128, 128, 14)
      [0] = PRE window stack  (median of pre-event frames, per-tile aligned)
      [1] = POST window stack (median of post-event frames, per-tile aligned)
      [2] = LATE window stack (frame 14, regrowth reference)
    Channels: 0-9 spectral bands (B02..B12), 10 = DEM, 11 = slope (deg),
              12 = reserved, 13 = NDVI (per stack).
    Returns (file_name, ok, error, metadata_dict).
    """
    file_name = os.path.basename(full_path)
    temp_nc_path = os.path.join(
        tempfile.gettempdir(), f"cosmoeye_decompress_{os.getpid()}.nc"
    )
    is_temp_created = False

    try:
        # 1. Open the NetCDF container (try plain, fall back to gzip decompression)
        try:
            src = Dataset(full_path, "r")
        except Exception:
            with gzip.open(full_path, "rb") as gz_file:
                file_content = gz_file.read()
            with open(temp_nc_path, "wb") as tmp_out:
                tmp_out.write(file_content)
            is_temp_created = True
            src = Dataset(temp_nc_path, "r")

        post_event_mask = np.array(src.variables["MASK"][-1, :, :])

        # 1b. Per-tile event metadata for temporal alignment + event-disjoint splits
        annotated = str(src.getncattr("annotated")).lower() == "true"
        metadata = {
            "annotated": annotated,
            "event_date": str(src.getncattr("event_date")).strip(),
            "ann_id": str(src.getncattr("ann_id")).strip(),
            "pre_post": parse_pre_post_dates(src.getncattr("pre_post_dates")),
        }
        windows = resolve_windows(metadata["annotated"], metadata["pre_post"])

        # 2. Read all spectral bands across all 15 frames
        band_series = {}
        for band_var in BAND_VARS:
            band_series[band_var] = np.array(
                src.variables[band_var][:], dtype=np.float32
            )

        # 3. Build the standardized temporal-stack layout matrix
        n_stacks = len(windows)
        standardized = np.zeros((n_stacks, 128, 128, 14), dtype=np.float32)
        for stack_idx, (f0, f1) in enumerate(windows):
            frame_slice = slice(f0, f1 + 1)
            for channel_idx, band_var in enumerate(BAND_VARS):
                stacked = np.nanmedian(band_series[band_var][frame_slice], axis=0)
                standardized[stack_idx, :, :, channel_idx] = np.where(
                    np.isfinite(stacked), stacked, 0.0
                )

        # 4. Static terrain layers (identical in every stack)
        dem = np.array(src.variables[DEM_VAR][-1, :, :], dtype=np.float32)
        dem = np.clip(np.where(np.isfinite(dem), dem, 0.0), DEM_MIN, DEM_MAX)
        slope = compute_slope(dem)
        standardized[:, :, :, 10] = dem
        standardized[:, :, :, 11] = slope

        # 5. Per-stack NDVI from B08 (NIR) and B04 (Red)
        for stack_idx in range(n_stacks):
            red = standardized[stack_idx, :, :, 2]
            nir = standardized[stack_idx, :, :, 6]
            standardized[stack_idx, :, :, NDVI_CHANNEL] = (
                (nir - red) / (nir + red + 1e-8)
            )

        src.close()  # Safely release system file locks immediately

        # Clean up the temporary file from disk space
        if is_temp_created and os.path.exists(temp_nc_path):
            os.remove(temp_nc_path)

        # 6. Extension-agnostic base name extractor
        base_str, ext = os.path.splitext(file_name)
        while ext in ['.gz', '.nc']:
            base_str, ext = os.path.splitext(base_str)
        clean_h5_name = f"{base_str}.h5"

        # 7. Overwrite any stale file from a previous conversion first
        img_save_path = os.path.join(img_out_dir, clean_h5_name)
        mask_save_path = os.path.join(mask_out_dir, clean_h5_name)
        for stale in (img_save_path, mask_save_path):
            if os.path.exists(stale):
                os.remove(stale)

        with h5py.File(img_save_path, "w") as f_img:
            f_img.create_dataset("img", data=standardized)

        # 8. Binarize the landslide label map (MASK > 0 => landslide pixel)
        with h5py.File(mask_save_path, "w") as f_msk:
            f_msk.create_dataset(
                "mask", data=(post_event_mask > 0).astype(np.uint8)
            )

        return (file_name, True, "", metadata)
    except Exception as e:
        if is_temp_created and os.path.exists(temp_nc_path):
            os.remove(temp_nc_path)
        return (file_name, False, str(e), None)


def extract_and_convert_sen12_dataset(raw_s2_dir="D:/COSMOEYE/datasets/s2/",
                                      output_root="D:/COSMOEYE/datasets/TrainData/",
                                      max_workers=4):
    """
    Recursively processes all subfolders within 's2' (s2_part01 to s2_part28)
    and formats pre/post-event change-pair .h5 containers.
    """
    # Safety Check: Fallback to local workspace relative paths if the D:/ layout fails
    if not os.path.exists(raw_s2_dir):
        raw_s2_dir = "./datasets/s2/"
        output_root = "./datasets/TrainData/"

    img_out_dir = os.path.join(output_root, "img")
    mask_out_dir = os.path.join(output_root, "mask")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    if not os.path.exists(raw_s2_dir):
        print(f"Error: Missing target source root folder at {raw_s2_dir}")
        return

    print(f"Initiating recursive file tree scan across: {raw_s2_dir}")

    all_file_paths = []
    for root, dirs, files in os.walk(raw_s2_dir):
        for f in files:
            # Audit items to ignore hidden operating system tracker cache blocks
            if not f.startswith("."):
                all_file_paths.append(os.path.join(root, f))

    total_discovered = len(all_file_paths)
    print(f"Discovered {total_discovered} total regional data files across subdirectories.")
    print("================== MONITORING MATRIX INGESTION ==================\n")

    processed_count = 0
    failed_files = []
    tile_metadata = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(convert_one_file, p, img_out_dir, mask_out_dir)
            for p in all_file_paths
        ]
        with tqdm(total=total_discovered, desc="Standardizing Temporal-Stacks Tiles", unit="img") as pbar:
            for future in as_completed(futures):
                file_name, ok, error_msg, metadata = future.result()
                pbar.update(1)
                if ok:
                    processed_count += 1
                    if metadata is not None:
                        base = file_name
                        while base.endswith((".gz", ".nc")):
                            base = base[:-3]
                        tile_metadata[base] = metadata
                else:
                    failed_files.append((file_name, error_msg))

    # Sidecar metadata for event-disjoint splits and per-tile alignment
    with open(METADATA_JSON_PATH, "w") as jf:
        import json
        json.dump(tile_metadata, jf, indent=0)

    # The dataset layout changed: force a rebuild of the normalization stats
    if os.path.exists(NORM_STATS_PATH):
        os.remove(NORM_STATS_PATH)

    print(f"\n================== CONVERSION SUCCESS ==================")
    print(f"Subfolder data translation complete!")
    print(f"Standardized {processed_count} / {total_discovered} files into temporal-stack .h5 files.")
    print(f"Metadata sidecar written for {len(tile_metadata)} tiles: {METADATA_JSON_PATH}")
    if failed_files:
        print(f"\nWARNING: {len(failed_files)} files failed and were skipped:")
        for name, err in failed_files[:10]:
            print(f"  - {name}: {err}")
    print("========================================================\n")


if __name__ == "__main__":
    extract_and_convert_sen12_dataset()
