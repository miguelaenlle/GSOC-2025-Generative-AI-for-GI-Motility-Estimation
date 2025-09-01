# Author: Miguel Aenlle

import os
import subprocess
import glob

def synthetic_data_generation():
    input_dir = f"../../../0 - Data Exploration & Analysis/GI-Roberta/gi-roberta-dataset/full_dataset_rgba"
    main_train_py = f"../../../1.5 - Synthetic Data Generation/Singan-Seg/main_train.py"

    images = glob.glob(rf"{input_dir}/*.png")  # Adjust this path as needed
    if not images:
        print("No images found. Please check the directory and image file paths.")
        return
    
    for img in images:
        img_name = os.path.basename(img)

        print(f"Generating synthetic data for: {img_name}")

        python_command = f"python \"{main_train_py}\" --input_name \"{img_name}\" --input_dir=\"{input_dir}\" --nc_z 4 --nc_im 4 --gpu_id 0"

        try:
            print(f"Running command: {python_command}")
            subprocess.run(python_command, shell=True, check=True)
            print(f"Command completed successfully for {img_name}\n")
        
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {img_name}: {e}")
            breakpoint()
            continue 

if __name__ == "__main__":
    synthetic_data_generation()