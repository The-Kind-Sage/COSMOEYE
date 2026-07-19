import os
import h5py
import torch
import numpy as np

# Import your custom modules directly from your workspace folder
from model import CustomUNet


def run_inference_pipeline(sample_file_name, threshold=0.5):
    """Loads trained weights, processes a multi-spectral image tensor patch,

    and saves the resulting binary prediction array to disk.
    """
    # 1. Select Available System Computing Resource Node
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting System Computing Node: {device}")

    # 2. Local File Path Definitions
    test_img_path = f"./datasets/TestData/img/{sample_file_name}"
    weights_path = "landslide_unet_weights.pth"

    # Path safety check to ensure files exist on your drive
    if not os.path.exists(test_img_path):
        print(f"Error: Missing targeted image file element at {test_img_path}")
        return
    if not os.path.exists(weights_path):
        print(
            f"Error: Trained model weights file not found at {weights_path}. Run train.py first!"
        )
        return

    # 3. Read Raw Multi-Spectral .h5 Matrix File from Local Storage
    print(f"Ingesting raw imagery container: {sample_file_name}")
    with h5py.File(test_img_path, "r") as f:
        # Data shape format inside .h5 is (128, 128, 14) -> (Height, Width, Channels)
        image_array = np.array(f["img"]).astype(np.float32)

    # 4. Prepare Tensor Transformations manually for PyTorch Ingestion
    # Convert numpy array to torch tensor, permute axes to match shape format (14, 128, 128)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    # Unsqueeze to add a batch dimension matching model input requirements -> (1, 14, 128, 128)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 5. Initialize Model Architecture and Load Calibrated Parameters
    print("Loading optimized parameters into network layers...")
    model = CustomUNet(in_channels=14, out_channels=1)
    # map_location ensures weights load safely on either CPU or GPU without errors
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()  # Freeze BatchNorm and Dropout layers to establish stable inference mode

    # 6. Process Forward Inference Pass
    print("Executing neural segmentation matrix operations...")
    with torch.no_grad():  # Turn off gradient engine track loops to maximize speed
        probability_mask = model(image_tensor)

    # 7. Post-Processing Matrix Reductions
    # Move tensor back to CPU memory space, remove batch/channel dimensions, convert to numpy
    probability_mask = probability_mask.squeeze().cpu().numpy()

    # Apply manual threshold logic: values over 0.5 become 1 (landslide), else 0 (normal ground)
    binary_prediction_mask = (probability_mask > threshold).astype(np.uint8)

    # 8. Save the Computed Mask Local Output Array File
    output_filename = f"predicted_mask_{sample_file_name.replace('.h5', '.npy')}"
    np.save(output_filename, binary_prediction_mask)

    print("\n================== STEP 13 INFERENCE OUTPUT ==================")
    print("Inference pipeline executed successfully.")
    print(f"Input Satellite Image Array Dimensions : {image_array.shape}")
    print(f"Predicted Probability Output Resolution: {probability_mask.shape}")
    print(f"Resulting Binary Output Mask Resolution: {binary_prediction_mask.shape}")
    print(
        f"Total Pixels Categorized as Landslide   : {np.sum(binary_prediction_mask)}"
    )
    print(f"Saved binary array output file path   : {output_filename}")
    print("==============================================================\n")

    return binary_prediction_mask


if __name__ == "__main__":
    # Test execution routine running on the first sample of your TestData folder
    test_directory = "./datasets/TestData/img/"

    if not os.path.exists(test_directory):
        print(f"Error: Missing folder tree at {test_directory}")
    else:
        test_files = sorted(os.listdir(test_directory))
        if len(test_files) == 0:
            print("Error: TestData image folder is empty.")
        else:
            # target the very first available unseeen file patch element
            first_test_file = test_files[0]
            run_inference_pipeline(first_test_file)
