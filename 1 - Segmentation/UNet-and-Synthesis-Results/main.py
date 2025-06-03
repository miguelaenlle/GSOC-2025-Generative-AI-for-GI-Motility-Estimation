import os
import shutil

import random
import torch
import wandb
import subprocess
import pandas as pd
from tqdm import tqdm

# full_dataset_folder = '/home/miguel/GI/0 - Data Exploration & Analysis/UW-Madison/stomach_data_and_masks_preparation/stomach_data_and_masks'
full_dataset_folder = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg/Input/data-RGBA'
train_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/train'
train_augmented_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/train_augmented'
test_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/test'
main_train_py = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg/main_train.py'

train_test_split_ratio = 0.8  # 80% for training, 20% for testing

def prepare_dataset():
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(train_augmented_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    if len(os.listdir(train_folder)) > 0 or len(os.listdir(test_folder)) > 0:
        print("Training and testing folders are not empty. Please clear them before running the script.")
        return

    files = os.listdir(full_dataset_folder)
    file_case_and_day = [file[:file.find('_slice')] for file in files]
    unique_case_and_days = list(set(file_case_and_day))

    # With the uniques, randomly select the training/testing subsets
    random.shuffle(unique_case_and_days)

    split_idx = int(len(unique_case_and_days) * train_test_split_ratio)

    train_cases = unique_case_and_days[:split_idx]
    test_cases = unique_case_and_days[split_idx:]
    
    # Copy the training files
    for case in tqdm(train_cases):
        case_files = [f for f in files if f.startswith(case)]
        for file in case_files:
            if 'stomach_mask' not in file:
                file_base = file[:-4]
                new_image_file = f"{file_base}_image.png"
                shutil.copy(os.path.join(full_dataset_folder, file), os.path.join(train_folder, new_image_file))
                mask_file = f"{file_base}_stomach_mask.png"
                shutil.copy(os.path.join(full_dataset_folder, mask_file), os.path.join(train_folder, mask_file))
    
    # Copy the testing files
    for case in tqdm(test_cases):
        case_files = [f for f in files if f.startswith(case)]
        for file in case_files:
            if 'stomach_mask' not in file:
                file_base = file[:-4]
                new_image_file = f"{file_base}_image.png"
                shutil.copy(os.path.join(full_dataset_folder, file), os.path.join(test_folder, new_image_file))
                mask_file = f"{file_base}_stomach_mask.png"
                shutil.copy(os.path.join(full_dataset_folder, mask_file), os.path.join(test_folder, mask_file))

def synthesize_with_singan():
    train_dataset = os.listdir(train_folder)
    if len(train_dataset) < 170:
        print("Not enough images in the training set for Singan-Seg synthesis. Need at least 170 images.")
        return
   
    selected_file = os.path.join(train_folder, 'selected_images.csv')
    if os.path.exists(selected_file):
        selected_image_df = pd.read_csv(selected_file)
        selected_images = selected_image_df['0'].unique().tolist()
    else:
        selected_images = random.sample(train_dataset, 170)
        pd.DataFrame(selected_images).to_csv(os.path.join(train_folder, 'selected_images.csv'), index=False)

    for img_name in selected_images:
        python_command = f"python '{main_train_py}' --input_dir '{train_folder}' --input_name {img_name} --nc_z 4 --nc_im 4 --gpu_id 0"
        try:
            print(f"Running command: {python_command}")
            result = subprocess.run(python_command, shell=True, check=True)
            print(f"Command completed successfully for {img_name}\n")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {img_name}: {e}")
            continue  # Continue processing the next image if an error occurs
    
    # Create augmented training dataset folder
    # Move the samples into here



def main():
    # Acquire the Madison dataset, and split it into training and testing sets (80/20)
    prepare_dataset()
    synthesize_with_singan()
    
    # Select a pretrained generative model or no generative model, and synthesize images
    # Place this in an augmented dataset folder, and use it to train a UNet model
    generative_models = [
        'none',
        'singan-seg',
        'ddim',
        'vae'
    ]
    for model in generative_models:
        if model == 'none':
            print("No generative model selected, skipping synthesis.")
            continue
        elif model == 'singan-seg':
            print("Using Singan-Seg for synthesis.")
            # Randomly select 170 images from the training set as bases for synthesis



        elif model == 'ddim':
            print("DDIM synthesis is not supported yet.")
        elif model == 'vae':
            print("VAE synthesis is not supported yet.")
        else:
            print(f"Unknown generative model: {model}. Skipping synthesis.")
            continue
            

    # Evaluate segmentation model performance using the Dice coefficient on the testing dataset


if __name__ == "__main__":
    main()