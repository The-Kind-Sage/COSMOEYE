def compile_nlp_input_vector(spatial_metadata_list, default_district="Sindhupalchok"):
    """Processes Step 17: Compresses raw geometric measurement vectors into a highly

    structured, sequential text prompt template for local NLP ingestion.
    """
    # If the vision engine detected zero active landslides, generate a clear control string
    if not spatial_metadata_list:
        return f"INPUT_VEC: District={default_district}; Anomaly=None; Status=Clear;"

    # For this operational phase, we target the most severe hazard event detected in the patch
    primary_hazard = max(spatial_metadata_list, key=lambda x: x["surface_area_sqm"])

    # Extract metrics out of the spatial math dictionary object
    area = primary_hazard["surface_area_sqm"]
    road = primary_hazard["nearest_infrastructure"]
    blocked_flag = primary_hazard["infrastructure_blocked"]

    # Compile the final structured token stream sequence string layout
    structured_prompt_string = (
        f"INPUT_VEC: District={default_district}; "
        f"Area={area}sqm; "
        f"Nearest_Road={road}; "
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
        "nearest_infrastructure": "Arniko_Highway",
        "infrastructure_blocked": "Yes",
        "proximity_distance_units": 3.61
    }]

    # Process Step 17 compression loop
    compiled_vector_string = compile_nlp_input_vector(mock_spatial_output)

    print("\n================== STEP 17 PROMPT OUTPUT ==================")
    print("Prompt sequence text stream compiled successfully.")
    print(f"Resulting NLP Prompt String: {compiled_vector_string}")
    print("============================================================\n")