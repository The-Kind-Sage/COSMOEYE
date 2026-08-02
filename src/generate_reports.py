import os
from paths import PATHS

# Import Step 17 prompt builder logic
from generate_prompt import compile_nlp_input_vector


def _rule_based_bulletin(structured_prompt_string):
    """Build bulletin from structured prompt without T5."""
    parts = {}
    for token in structured_prompt_string.replace("INPUT_VEC: ", "").strip(";").split("; "):
        if "=" in token:
            key, _, val = token.partition("=")
            parts[key.strip()] = val.strip()

    if parts.get("Anomaly") == "None":
        return (
            f"CRITICAL INCIDENT BRIEF: Disaster risk parameters are currently within "
            f"normal baseline thresholds across {parts.get('District', 'Unknown')} District. "
            f"Satellite imagery reveals no active slope displacement anomalies. "
            f"Transport corridors remain fully operational."
        )

    district = parts.get("District", "Unknown")
    area = parts.get("Area", "unknown area")
    centroid = parts.get("Centroid", "unknown")
    nearest_road = parts.get("Nearest_Road", "Unknown")
    blocked = parts.get("Blocked", "No")

    bulletin = (
        f"CRITICAL DISASTER BULLETIN - GOVERNMENT OF NEPAL (NDRRMA PROTOCOL)\n"
        f"Location Focus: {district} District, Nepal\n"
        f"Satellite observation metrics reveal an extreme mass-wasting event covering "
        f"approximately {area} of land surface slope, "
        f"centred on tile grid position {centroid}. "
        f"Local disaster coordinate teams are advised to deploy immediate response "
        f"units to this sector."
    )

    # Road blockage sentence — appended only when the model flagged a blockage.
    if blocked == "Yes":
        bulletin += (
            f"\nROAD BLOCKAGE ALERT: {nearest_road} is potentially obstructed by "
            f"the detected landslide deposit. Access route verification and emergency "
            f"detour planning are strongly recommended before dispatching field teams."
        )
    else:
        bulletin += (
            f"\nNearest transport corridor: {nearest_road} — no blockage detected "
            f"within the current proximity threshold."
        )

    return bulletin


def execute_offline_report_generator(structured_prompt_string,
                                     output_txt_path=None):
    """Step 20: Ingests prompt, optionally runs offline T5 inference, or falls back to rule-based report."""
    if output_txt_path is None:
        output_txt_path = PATHS.EMERGENCY_BULLETIN

    decoded_natural_language_report = None

    try:
        import torch
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer_path = PATHS.T5_TOKENIZER
        model_weights_path = PATHS.T5_WEIGHTS

        if os.path.exists(tokenizer_path) and os.path.exists(model_weights_path):
            local_tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            local_model = T5ForConditionalGeneration.from_pretrained(model_weights_path, local_files_only=True)
            local_model.to(device)
            local_model.eval()

            input_tensors = local_tokenizer(
                structured_prompt_string,
                return_tensors="pt",
                max_length=128,
                truncation=True
            ).to(device)

            with torch.no_grad():
                output_token_ids = local_model.generate(
                    input_ids=input_tensors["input_ids"],
                    attention_mask=input_tensors["attention_mask"],
                    max_length=100,
                    num_beams=3,
                    early_stopping=True
                )

            decoded_natural_language_report = local_tokenizer.decode(
                output_token_ids[0],
                skip_special_tokens=True
            )

            if len(decoded_natural_language_report.strip()) < 5 or "vec" in decoded_natural_language_report.lower():
                decoded_natural_language_report = None
    except Exception:
        decoded_natural_language_report = None

    if decoded_natural_language_report is None:
        decoded_natural_language_report = _rule_based_bulletin(structured_prompt_string)

    # Write report to file
    with open(output_txt_path, "w") as file_out:
        file_out.write(decoded_natural_language_report)

    return decoded_natural_language_report, output_txt_path


if __name__ == "__main__":
    print("Starting Step 20 Master System Integration and Report Publishing Pass...")

    # Mock input data from upstream detection pipeline
    simulated_metadata = [{
        "object_id": 1,
        "centroid_pixel": (52, 42),
        "surface_area_sqm": 2500,
    }]

    # Build prompt (Step 17) — include road blockage flags
    prompt_string = compile_nlp_input_vector(
        simulated_metadata,
        default_district="Sindhupalchok",
        nearest_road="Araniko Highway",
        road_blocked=True,
    )
    print(f"Compiled Input Data Vector Stream: {prompt_string}")

    # Run generation
    final_text, saved_path = execute_offline_report_generator(prompt_string)

    print("\n================== STEP 20 AUTOMATED BULLETIN OUTPUT ==================")
    print(f"Natural language emergency report successfully compiled and saved.")
    print(f"Target Output Storage File Location: {saved_path}\n")
    print(final_text)
    print("========================================================================\n")