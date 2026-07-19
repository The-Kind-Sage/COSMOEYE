import os
import cv2
import numpy as np
import pandas as pd


def check_infrastructure_blockage(centroid_pixel, csv_path="local_highways.csv", threshold=5.0):
    """Processes Step 16: Calculates the Euclidean distance between a landslide center

    and local highways to check for blockages entirely offline.
    """
    if not os.path.exists(csv_path):
        # Fallback dictionary to keep the pipeline from crashing if the CSV file is missing
        return "Unknown_Road", False, 999.0

    # Load the offline coordinates database matrix using pandas
    df = pd.read_csv(csv_path)
    
    closest_highway = "None"
    min_distance = float("inf")
    is_blocked = False

    lx, ly = centroid_pixel

    # Loop through every row in the CSV matrix to perform geometry calculus
    for _, row in df.iterrows():
        hx = row["x_coordinate"]
        hy = row["y_coordinate"]
        name = row["highway_name"]

        # Apply the Euclidean Distance Formula from scratch
        distance = np.sqrt((lx - hx)**2 + (ly - hy)**2)

        # Track the closest highway asset encountered
        if distance < min_distance:
            min_distance = distance
            closest_highway = name

    # If the closest highway sits within our danger radius zone, flag a blockage
    if min_distance <= threshold:
        is_blocked = True

    return closest_highway, is_blocked, min_distance


def extract_spatial_metrics(binary_mask_array, csv_path="local_highways.csv"):
    """Processes Steps 14, 15, and 16: Contours objects, calculates areas, and

    cross-references infrastructural distances to build an NLP-ready summary.
    """
    if binary_mask_array is None:
        raise ValueError("Matrix error: Input mask array cannot be empty.")

    # Step 14: Topological Contour Extraction Pass
    contours, _ = cv2.findContours(binary_mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    structured_landslide_objects = []

    # Step 15: Geometric Calculus Loop
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

        # Step 16 Integration: Cross-reference coordinate locations against the road matrix
        nearest_road, road_blocked, exact_distance = check_infrastructure_blockage(
            centroid_pixel, csv_path=csv_path, threshold=5.0
        )

        landslide_metadata = {
            "object_id": idx + 1,
            "centroid_pixel": centroid_pixel,
            "surface_area_sqm": int(metric_surface_area_sqm),
            "nearest_infrastructure": nearest_road,
            "infrastructure_blocked": "Yes" if road_blocked else "No",
            "proximity_distance_units": round(float(exact_distance), 2)
        }

        structured_landslide_objects.append(landslide_metadata)

    return structured_landslide_objects


# --- Script Local Operational Verification Loop ---
if __name__ == "__main__":
    print("Starting Step 14, 15, and 16 Combined Vector Proximity Check...")

    # Verification Simulation Matrix: Construct an artificial mock hazard
    # Draw a mock landslide at coordinates (42, 53) to intentionally trigger
    # a blockage warning against the Arniko Highway coordinates (45, 55)
    sample_mask = np.zeros((128, 128), dtype=np.uint8)
    sample_mask[40:46, 50:56] = 1  # 6x6 pixel block = 36 pixels = 3600 sqm area

    # Create dummy local_highways.csv file automatically if missing during runtime testing
    csv_file = "local_highways.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            f.write("highway_name,x_coordinate,y_coordinate\n")
            f.write("Arniko_Highway,45,55\n")
            f.write("Prithvi_Highway,20,35\n")
            f.write("BP_Highway,80,95\n")

    # Run the integrated geospatial math pipelines
    extracted_hazards = extract_spatial_metrics(sample_mask, csv_path=csv_file)

    print("\n================== GEOSPATIAL PROXIMITY ENGINE OUTPUT ==================")
    print(f"Total Valid Hazard Objects Logged: {len(extracted_hazards)}")
    print("------------------------------------------------------------------------")
    
    for hazard in extracted_hazards:
        print(f"Anomaly ID: {hazard['object_id']}")
        print(f"  Center Coordinates (X, Y Grid)    : {hazard['centroid_pixel']}")
        print(f"  Calculated Surface Area Footprint : {hazard['surface_area_sqm']:,} sqm")
        print(f"  Nearest Monitored Infrastructure  : {hazard['nearest_infrastructure']}")
        print(f"  Proximity Distance on Pixel Grid  : {hazard['proximity_distance_units']} units")
        print(f"  Critical Blockage Status Flag     : {hazard['infrastructure_blocked']}")
        print("------------------------------------------------------------------------")
    print("========================================================================\n")
