def compile_nlp_input_vector(spatial_metadata_list, default_district="Sindhupalchok",
                              nearest_road=None, road_blocked=False):
    """Processes Step 17: Compresses raw geometric measurement vectors into a highly

    structured, sequential text prompt template for local NLP ingestion.

    Parameters
    ----------
    spatial_metadata_list : list[dict]
        Output of spatial_math.extract_spatial_metrics — one dict per
        detected landslide body.
    default_district : str
        Administrative district name for the tile.
    nearest_road : str | None
        Name of the road nearest to the primary detected landslide, as
        returned by spatial_math.check_infrastructure_blockage.
    road_blocked : bool
        True when the nearest road is within the blockage proximity threshold.
    """
    # If the vision engine detected zero active landslides, generate a clear control string
    if not spatial_metadata_list:
        return f"INPUT_VEC: District={default_district}; Anomaly=None; Status=Clear;"

    # For this operational phase, we target the most severe hazard event detected in the patch
    primary_hazard = max(spatial_metadata_list, key=lambda x: x["surface_area_sqm"])

    # Extract metrics out of the spatial math dictionary object
    area = primary_hazard["surface_area_sqm"]
    centroid = primary_hazard["centroid_pixel"]

    # Road proximity flags
    road_name = nearest_road if nearest_road else "Unknown"
    blocked_flag = "Yes" if road_blocked else "No"

    # Compile the final structured token stream sequence string layout
    structured_prompt_string = (
        f"INPUT_VEC: District={default_district}; "
        f"Area={area}sqm; "
        f"Centroid={centroid[0]},{centroid[1]}; "
        f"Nearest_Road={road_name}; "
        f"Blocked={blocked_flag};"
    )

    return structured_prompt_string


if __name__ == "__main__":
    print("Starting Step 17 Prompt Compilation Verification Check...")

    # Simulated output data list mimicking a true results pass from spatial_math.py
    mock_spatial_output = [{
        "object_id": 1,
        "centroid_pixel": (42, 53),
        "surface_area_sqm": 3600,
    }]

    # Process Step 17 compression loop (with road blockage flags)
    compiled_vector_string = compile_nlp_input_vector(
        mock_spatial_output,
        nearest_road="Araniko Highway",
        road_blocked=True,
    )

    print("\n================== STEP 17 PROMPT OUTPUT ==================")
    print("Prompt sequence text stream compiled successfully.")
    print(f"Resulting NLP Prompt String: {compiled_vector_string}")
    print("============================================================\n")