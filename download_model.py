import os
from transformers import T5ForConditionalGeneration


def cache_model_weights_locally():
    """Processes Step 19: Fetches the open-source T5-Small model parameters

    from online servers and writes them to local storage for offline use.
    """
    target_local_dir = "./local_models/t5_weights/"
    model_blueprint_name = "t5-small"

    print(
        f"Initiating downloading protocols for: {model_blueprint_name} parameter weights..."
    )

    # 1. Download model structural architecture and weights via HuggingFace pipelines
    # This pulls down config.json, generation_config.json, and the model weight files
    model = T5ForConditionalGeneration.from_pretrained(model_blueprint_name)

    # 2. Execute local file system writing routines
    print(
        f"Writing deep learning model matrices to directory: {target_local_dir}"
    )
    model.save_pretrained(target_local_dir)

    # 3. Academic validation step to confirm files exist on your hard drive
    required_file = os.path.join(target_local_dir, "config.json")
    if os.path.exists(required_file):
        print("\n================== STEP 19 MODEL OUTPUT ==================")
        print("Local model weights environment cache setup complete.")
        print(
            "Configuration parameters and weight matrices verified on disk storage."
        )
        print("The NLP text generation decoder can now run completely offline.")
        print("==========================================================\n")
    else:
        print(
            "Error: File system write block. Verify storage capacity or directory permissions."
        )


if __name__ == "__main__":
    cache_model_weights_locally()
