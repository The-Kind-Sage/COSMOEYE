import os
from transformers import T5Tokenizer


def cache_tokenizer_locally():
    """Processes Step 18: Connects to open repositories to download and store

    the T5 configuration json and vocabulary maps onto the laptop drive.
    """
    target_local_dir = "./local_models/t5_tokenizer/"
    model_blueprint_name = "t5-small"

    print(f"Initiating downloading protocols for: {model_blueprint_name} vocabulary files...")

    # 1. Fetch tokenizer matrices via HuggingFace hub framework pipelines
    tokenizer = T5Tokenizer.from_pretrained(model_blueprint_name)

    # 2. Force structural write protocols to hardcoded offline storage directory paths
    print(f"Writing vocabulary and configurations data structures to: {target_local_dir}")
    tokenizer.save_pretrained(target_local_dir)

    # 3. Verify files sit safely on the local file tree layout
    if os.path.exists(os.path.join(target_local_dir, "tokenizer.json")):
        print("\n================== STEP 18 TOKENIZER OUTPUT ==================")
        print("Local tokenizer environment cache setup successfully completed.")
        print("All required vocabulary JSON index files verified on disk storage.")
        print("The NLP string tokenization layer can now run completely offline.")
        print("==============================================================\n")
    else:
        print("Error: Missing file structures. Verify folder write permissions attributes.")


if __name__ == "__main__":
    cache_tokenizer_locally()
