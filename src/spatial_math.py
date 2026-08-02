import csv
import os

import cv2
import numpy as np

# Path to the local road reference table (pixel-grid coordinates).
# One row per road segment; the pixel_x/pixel_y columns are in the same
# 128×128 coordinate space as the binary detection mask.
try:
    from paths import PATHS as _PATHS
    _HIGHWAYS_CSV = _PATHS.HIGHWAYS_CSV
except ImportError:
    _HIGHWAYS_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "reference", "local_highways.csv")


def _load_highway_segments(csv_path=_HIGHWAYS_CSV):
    """Parse local_highways.csv and return a list of road-segment dicts.

    Each dict has: road_id (int), road_name (str), road_type (str),
    pixel_x (int), pixel_y (int).  Missing or malformed rows are silently
    skipped so a corrupt CSV never aborts an inference pass.
    """
    segments = []
    if not os.path.exists(csv_path):
        return segments
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    segments.append({
                        "road_id":   int(row["road_id"]),
                        "road_name": row["road_name"].strip(),
                        "road_type": row["road_type"].strip(),
                        "pixel_x":   int(row["pixel_x"]),
                        "pixel_y":   int(row["pixel_y"]),
                    })
                except (KeyError, ValueError):
                    continue
    except Exception:
        pass
    return segments


def check_infrastructure_blockage(binary_mask_array, proximity_m=500.0,
                                   pixel_size_m=10.0, csv_path=_HIGHWAYS_CSV):
    """Check whether any detected landslide body falls within *proximity_m*
    metres of a road segment listed in local_highways.csv.

    The tile resolution is 10 m/px, so a 500 m proximity threshold equals
    50 pixels.  The function finds the road segment whose reference pixel is
    closest to the nearest detected landslide pixel and compares that
    Euclidean pixel distance (converted to metres) against the threshold.

    Parameters
    ----------
    binary_mask_array : np.ndarray, shape (H, W), dtype uint8
        Binary landslide detection mask (1 = landslide, 0 = background).
    proximity_m : float
        Distance in metres below which a road is considered blocked.
    pixel_size_m : float
        Ground resolution in metres per pixel (default 10 m for Sentinel-2).
    csv_path : str
        Override path to local_highways.csv (used by tests).

    Returns
    -------
    dict with keys:
        nearest_road (str | None)   — name of the closest road segment,
                                      or None if no roads loaded / no detections.
        nearest_road_dist_m (float) — Euclidean distance in metres to that road.
        road_blocked (bool)         — True when nearest_road_dist_m < proximity_m.
        road_type (str | None)      — OSM highway type of the nearest road.
    """
    null_result = {
        "nearest_road": None,
        "nearest_road_dist_m": float("inf"),
        "road_blocked": False,
        "road_type": None,
    }

    segments = _load_highway_segments(csv_path)
    if not segments:
        return null_result

    # No landslide detected → no blockage possible.
    detected_yx = np.argwhere(binary_mask_array > 0)   # shape (N, 2): [row, col]
    if detected_yx.size == 0:
        return null_result

    proximity_px = proximity_m / pixel_size_m

    best_dist_px = float("inf")
    best_segment = None

    for seg in segments:
        seg_col = seg["pixel_x"]   # x = column
        seg_row = seg["pixel_y"]   # y = row
        # Euclidean distance from each detected pixel to this road point.
        # detected_yx[:, 0] = row, detected_yx[:, 1] = col.
        dists = np.hypot(detected_yx[:, 1] - seg_col,
                         detected_yx[:, 0] - seg_row)
        min_dist = float(dists.min())
        if min_dist < best_dist_px:
            best_dist_px = min_dist
            best_segment = seg

    if best_segment is None:
        return null_result

    best_dist_m = best_dist_px * pixel_size_m
    return {
        "nearest_road":      best_segment["road_name"],
        "nearest_road_dist_m": best_dist_m,
        "road_blocked":      best_dist_m < proximity_m,
        "road_type":         best_segment["road_type"],
    }


def extract_spatial_metrics(binary_mask_array):
    """Contours the detected objects and calculates their areas and centroids,
    building an NLP-ready summary of each landslide body.
    """
    if binary_mask_array is None:
        raise ValueError("Matrix error: Input mask array cannot be empty.")

    # Topological Contour Extraction Pass
    contours, _ = cv2.findContours(binary_mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    structured_landslide_objects = []

    # Geometric Calculus Loop
    for idx, contour in enumerate(contours):
        moments = cv2.moments(contour)
        pixel_area = moments["m00"]

        # Keep the noise filter active to comply with robust CV guidelines
        if pixel_area < 3:
            continue

        centroid_x = int(moments["m10"] / (moments["m00"] + 1e-8))
        centroid_y = int(moments["m01"] / (moments["m00"] + 1e-8))
        centroid_pixel = (centroid_x, centroid_y)

        metric_surface_area_sqm = pixel_area * 100

        landslide_metadata = {
            "object_id": idx + 1,
            "centroid_pixel": centroid_pixel,
            "surface_area_sqm": int(metric_surface_area_sqm),
        }

        structured_landslide_objects.append(landslide_metadata)

    return structured_landslide_objects


# --- Script Local Operational Verification Loop ---
if __name__ == "__main__":
    print("Starting contour + area extraction check...")

    # Verification Simulation Matrix: Construct an artificial mock hazard
    sample_mask = np.zeros((128, 128), dtype=np.uint8)
    sample_mask[40:46, 50:56] = 1  # 6x6 pixel block = 36 pixels = 3600 sqm area

    extracted_hazards = extract_spatial_metrics(sample_mask)

    print("\n================== GEOSPATIAL ENGINE OUTPUT ==================")
    print(f"Total Valid Hazard Objects Logged: {len(extracted_hazards)}")
    print("--------------------------------------------------------------")

    for hazard in extracted_hazards:
        print(f"Anomaly ID: {hazard['object_id']}")
        print(f"  Center Coordinates (X, Y Grid)    : {hazard['centroid_pixel']}")
        print(f"  Calculated Surface Area Footprint : {hazard['surface_area_sqm']:,} sqm")
        print("--------------------------------------------------------------")
    print("==============================================================\n")
