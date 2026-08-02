import cv2
import numpy as np


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
