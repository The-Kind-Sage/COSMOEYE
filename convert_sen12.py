import os
import gzip
import h5py
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm  

def extract_and_convert_sen12_dataset(raw_s2_dir="D:/COSMOEYE/datasets/s2/", output_root="D:/COSMOEYE/datasets/TrainData/"):
    """
    Recursively processes all subfolders within 's2' (s2_part01 to s2_part28),
    extracts pre-unpacked spectral bands, and formats output filenames as standard .h5 containers.
    """
    # Safety Check: Fallback to local workspace relative paths if the D:/ directory layout fails
    if not os.path.exists(raw_s2_dir):
        raw_s2_dir = "./datasets/s2/"
        output_root = "./datasets/TrainData/"

    img_out_dir = os.path.join(output_root, "img")
    mask_out_dir = os.path.join(output_root, "mask")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    
    if not os.path.exists(raw_s2_dir):
        print(f"Error: Missing target source root folder at {raw_s2_dir}")
        return

    print(f"Initiating recursive file tree scan across: {raw_s2_dir}")

    all_file_paths = []
    for root, dirs, files in os.walk(raw_s2_dir):
        for f in files:
            # Audit items to ignore hidden operating system tracker cache blocks
            if not f.startswith("."):
                all_file_paths.append(os.path.join(root, f))

    total_discovered = len(all_file_paths)
    print(f"🚀 Discovered {total_discovered} total regional data files across subdirectories.")
    print("================== MONITORING MATRIX INGESTION ==================\n")

    processed_count = 0
    temp_nc_path = "./temp_decompress_block.nc"

    # Initialize the live-updating visual progress loader bar interface
    for full_path in tqdm(all_file_paths, desc="Standardizing Satellite Tiles", unit="img"):
        file_name = os.path.basename(full_path)
        is_temp_created = False
        
        try:
            # 1. Open the NetCDF file container (Try uncompressed first, then fallback to Gzip disk decompression)
            try:
                src = Dataset(full_path, "r")
            except Exception:
                with gzip.open(full_path, 'rb') as gz_file:
                    file_content = gz_file.read()
                with open(temp_nc_path, 'wb') as tmp_out:
                    tmp_out.write(file_content)
                is_temp_created = True
                src = Dataset(temp_nc_path, "r")
                
            # 2. Extract specific Red (B04) and NIR (B08) metrics directly using individual key identifiers
            # Index -1 selects the final post-event temporal scene state
            red_channel = src.variables["B04"][-1, :, :].astype(np.float32)
            nir_channel = src.variables["B08"][-1, :, :].astype(np.float32)
            post_event_mask = src.variables["MASK"][-1, :, :]      
            src.close()  # Safely release system file locks immediately

            # Clean up the temporary file from disk space
            if is_temp_created and os.path.exists(temp_nc_path):
                os.remove(temp_nc_path)
            
            # 3. Build Option A 14-Channel Layout Matrix
            standardized_14_bands = np.zeros((128, 128, 14), dtype=np.float32)
            standardized_14_bands[:, :, 3] = red_channel  
            standardized_14_bands[:, :, 7] = nir_channel  
            
            # Compute manual NDVI from scratch to protect the 14th band channel slot
            numerator = nir_channel - red_channel
            denominator = nir_channel + red_channel + 1e-8  
            computed_ndvi = numerator / denominator
            standardized_14_bands[:, :, 13] = computed_ndvi
            
            # 4. Extension-agnostic base name extractor
            # Recursively drops trailing extensions to isolate clean base names like "chimanimani_s2_101"
            base_str, ext = os.path.splitext(file_name)
            while ext in ['.gz', '.nc']:
                base_str, ext = os.path.splitext(base_str)
            clean_h5_name = f"{base_str}.h5"
            
            # 5. Save standardized files to disk storage paths
            img_save_path = os.path.join(img_out_dir, clean_h5_name)
            mask_save_path = os.path.join(mask_out_dir, clean_h5_name)

            with h5py.File(img_save_path, "w") as f_img:
                f_img.create_dataset("img", data=standardized_14_bands)
                
            with h5py.File(mask_save_path, "w") as f_msk:
                f_msk.create_dataset("mask", data=post_event_mask.astype(np.uint8))
                
            processed_count += 1
                
        except Exception as e:
            # Safely catch error trails and clean leftover file handles
            if is_temp_created and os.path.exists(temp_nc_path):
                os.remove(temp_nc_path)
            continue

    print(f"\n================== CONVERSION SUCCESS ==================")
    print(f"✅ Subfolder data translation complete!")
    print(f"Standardized {processed_count} / {total_discovered} files into .h5 formats inside: {output_root}")
    print("========================================================\n")

if __name__ == "__main__":
    extract_and_convert_sen12_dataset()
