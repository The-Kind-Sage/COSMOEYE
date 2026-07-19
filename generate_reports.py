import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Import Step 17 prompt builder logic
from generate_prompt import compile_nlp_input_vector


def execute_offline_report_generator(structured_prompt_string, output_txt_path="emergency_bulletin.txt"):
    """Step 20: Ingests prompt, tokenizes, runs offline T5 inference, and writes report."""
    
    # 1. Set compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Local model paths
    tokenizer_path = "./local_models/t5_tokenizer/"
    model_weights_path = "./local_models/t5_weights/"

    if not os.path.exists(tokenizer_path) or not os.path.exists(model_weights_path):
        raise FileNotFoundError("Error: Offline T5 directories missing. Run Steps 18 & 19 first!")

    # 3. Load model and tokenizer offline
    local_tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    local_model = T5ForConditionalGeneration.from_pretrained(model_weights_path, local_files_only=True)
    local_model.to(device)
    local_model.eval()

    # 4. Tokenize input string
    input_tensors = local_tokenizer(
        structured_prompt_string, 
        return_tensors="pt", 
        max_length=128, 
        truncation=True
    ).to(device)

    # 5. Generate text via beam search
    with torch.no_grad():
        output_token_ids = local_model.generate(
            input_ids=input_tensors["input_ids"],
            attention_mask=input_tensors["attention_mask"],
            max_length=100,
            num_beams=3,
            early_stopping=True
        )

    # 6. Decode tokens back to string
    decoded_natural_language_report = local_tokenizer.decode(
        output_token_ids[0], 
        skip_special_tokens=True
    )

    # 7. Rule-based fallback parser if model output is empty/invalid
    if len(decoded_natural_language_report.strip()) < 5 or "vec" in decoded_natural_language_report.lower():
        parts = {k.split("=")[0]: k.split("=")[1] for k in structured_prompt_string.replace("INPUT_VEC: ", "").strip(";").split("; ")}
        
        if parts.get("Anomaly") == "None":
            decoded_natural_language_report = f"CRITICAL INCIDENT BRIEF: Disaster risk parameters are currently within normal baseline thresholds across {parts['District']} District. Satellite imagery reveals no active slope displacement anomalies. Transport corridors remain fully operational."
        else:
            block_msg = f"The critical trade network route {parts['Nearest_Road']} is physically BLOCKED by debris flow." if parts['Blocked'] == 'Yes' else f"The nearest transport route {parts['Nearest_Road']} remains open, but emergency teams should monitor adjacent slopes."
            decoded_natural_language_report = (
                f"CRITICAL DISASTER BULLETIN - GOVERNMENT OF NEPAL (NDRRMA PROTOCOL)\n"
                f"Location Focus: {parts['District']} District, Nepal\n"
                f"Satellite observation metrics reveal an extreme mass-wasting event covering approximately {parts['Area']} of land surface slope. "
                f"{block_msg} Local disaster coordinate teams are advised to deploy immediate response units to this sector."
            )

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
        "nearest_infrastructure": "Arniko_Highway",
        "infrastructure_blocked": "Yes"
    }]

    # Build prompt (Step 17)
    prompt_string = compile_nlp_input_vector(simulated_metadata, default_district="Sindhupalchok")
    print(f"Compiled Input Data Vector Stream: {prompt_string}")

    # Run generation
    final_text, saved_path = execute_offline_report_generator(prompt_string)

    print("\n================== STEP 20 AUTOMATED BULLETIN OUTPUT ==================")
    print(f"Natural language emergency report successfully compiled and saved.")
    print(f"Target Output Storage File Location: {saved_path}\n")
    print(final_text)
    print("========================================================================\n")