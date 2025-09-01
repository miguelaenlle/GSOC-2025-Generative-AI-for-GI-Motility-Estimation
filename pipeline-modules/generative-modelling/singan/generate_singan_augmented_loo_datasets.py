# Author: Miguel Aenlle

import os
import sys
import shutil
import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm

TRAINED_MODELS_PATH = '../../1.5 - Synthetic Data Generation/Singan-Seg/TrainedModels'
RANDOM_SAMPLES_PY = '../../1.5 - Synthetic Data Generation/Singan-Seg/random_samples.py'
FULL_DATASET_FOLDER_PATH = '../../0 - Data Exploration & Analysis/GI-Roberta/gi-roberta-dataset/full_dataset'
FULL_DATRASET_RGBA_FOLDER_PATH = '../../0 - Data Exploration & Analysis/GI-Roberta/gi-roberta-dataset/full_dataset_rgba'
OUTPUT_FOLDER_PATH = 'Output_benchmark'
RANDOM_SAMPLES = '../../1.5 - Synthetic Data Generation/Singan-Seg/Output_benchmark/RandomSamples'

AUGMENTED_DATASET_FOLDER_PATH = '../../1.5 - Synthetic Data Generation/Singan-Seg/unet_singan_augmented_datasets'

def prepare_dataset(
    expansion_factor: float,
    reset_folders = True
):
    if reset_folders:
        if os.path.exists(RANDOM_SAMPLES):
            shutil.rmtree(RANDOM_SAMPLES)
        os.makedirs(RANDOM_SAMPLES)

        if os.path.exists(AUGMENTED_DATASET_FOLDER_PATH):
            shutil.rmtree(AUGMENTED_DATASET_FOLDER_PATH)
        os.makedirs(AUGMENTED_DATASET_FOLDER_PATH)

    trained_models = [f for f in os.listdir(TRAINED_MODELS_PATH) if os.path.isdir(os.path.join(TRAINED_MODELS_PATH, f)) and 'FD' in f]
    model_names = np.unique([f.split('-slice')[0] for f in trained_models])

    singan_ssim_results = pd.read_csv('../../1.5 - Synthetic Data Generation/Singan-Seg/singan_ssim_results.csv')
    singan_ssim_results['model_name'] = [f.split('-slice')[0] for f in singan_ssim_results['image_name']]

    singan_ssim_results = singan_ssim_results[singan_ssim_results['num_samples_gt_0_5'] > 3]
    trained_models_and_model_names = singan_ssim_results[['image_name', 'model_name']]
    trained_models_and_model_names.columns = ['trained_model', 'model_name']

    images = os.listdir(FULL_DATRASET_RGBA_FOLDER_PATH)
    image_datas = pd.DataFrame({
        'file_name': images,
        'model_name': [f.split('-slice')[0] for f in images]
    })

    factor_augmented_dataset_path = os.path.join(AUGMENTED_DATASET_FOLDER_PATH, f'augmented_dataset_expansion_factor_{expansion_factor}')
    if not os.path.exists(factor_augmented_dataset_path):
        os.makedirs(factor_augmented_dataset_path)

    for cv_model_name in tqdm(model_names): 
        cv_model_dataset_path = os.path.join(factor_augmented_dataset_path, cv_model_name)
        if not os.path.exists(cv_model_dataset_path):
            os.makedirs(cv_model_dataset_path)

        train_dataset_path = os.path.join(cv_model_dataset_path, 'train')
        val_dataset_path = os.path.join(cv_model_dataset_path, 'val')
        if not os.path.exists(train_dataset_path):
            os.makedirs(train_dataset_path)
        if not os.path.exists(val_dataset_path):
            os.makedirs(val_dataset_path)

        # cv_model_name is the one that will be for cross-validation

        # Train samples used for synthesis
        trained_models_and_model_names_df = trained_models_and_model_names[trained_models_and_model_names['model_name'] != cv_model_name]

        # Move the standard training samples into the folder
        training_samples = image_datas[image_datas['model_name'] != cv_model_name]
        cv_val_samples = image_datas[image_datas['model_name'] == cv_model_name]

        for i in training_samples.index:
            file_name = training_samples.loc[i, 'file_name']
            original_path = os.path.join(FULL_DATASET_FOLDER_PATH, file_name)
            destination_path = os.path.join(train_dataset_path, file_name)

            mask_file_name = file_name.replace('-image.png', '-mask.png')
            original_mask_path = os.path.join(FULL_DATASET_FOLDER_PATH, mask_file_name)
            destination_mask_path = os.path.join(train_dataset_path, mask_file_name)
            
            shutil.copy(original_path, destination_path)
            shutil.copy(original_mask_path, destination_mask_path)

        
        # Move the cross-validation validation samples into the folder
        for i in cv_val_samples.index:
            file_name = cv_val_samples.loc[i, 'file_name']
            original_path = os.path.join(FULL_DATASET_FOLDER_PATH, file_name)
            destination_path = os.path.join(val_dataset_path, file_name)

            mask_file_name = file_name.replace('-image.png', '-mask.png')
            original_mask_path = os.path.join(FULL_DATASET_FOLDER_PATH, mask_file_name)
            destination_mask_path = os.path.join(val_dataset_path, mask_file_name)
            
            shutil.copy(original_mask_path, destination_mask_path)
            shutil.copy(original_path, destination_path)

        # Determine the total number of samples to generate per model
        training_samples = image_datas[image_datas['model_name'] != cv_model_name].shape[0]
        total_samples = int(training_samples * expansion_factor)

        # Synthesize 2x more than needed to account for low-quality synthetic samples./
        num_samples_per_synthesis_sample = max(total_samples // trained_models_and_model_names_df.shape[0], 10) * 2

        # We will reduce the dataset size as needed.

        for i in tqdm(trained_models_and_model_names_df.index):
            trained_model = trained_models_and_model_names_df.loc[i, 'trained_model']
            python_command = f"python '{RANDOM_SAMPLES_PY}' --input_name '{trained_model + '.png'}' --input_dir='{FULL_DATASET_FOLDER_PATH}' --mode random_samples --gen_start_scale 0 --nc_z 4 --nc_im 4 --gpu_id 0 --num_samples {num_samples_per_synthesis_sample} --out '{OUTPUT_FOLDER_PATH}'"

            try:
                print(f"Running command: {python_command}")
                result = subprocess.run(python_command, shell=True, check=True)
                print(f"Command completed successfully for {cv_model_name}\n")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while processing {cv_model_name}: {e}")
                continue  # Continue processing the next image if an error occurs

            # Move the generated samples to the appropriate folder
            folder_path = os.path.join(RANDOM_SAMPLES, trained_model, 'gen_start_scale=0')
            for synthetic_image_name in os.listdir(folder_path):
                synthetic_image_path = os.path.join(folder_path, synthetic_image_name)

                # To the filename, prepend the trained_model
                new_image_name = trained_model + '-synthetic-' + synthetic_image_name

                destination_path = os.path.join(train_dataset_path, new_image_name)

                # Process the synthetic image
                shutil.copy(synthetic_image_path, destination_path)

def main():
    singanseg_folder = os.path.abspath('../../../1.5 - Synthetic Data Generation/Singan-Seg')
    os.chdir(singanseg_folder)
    for expansion_factor in [0.5, 1.0, 2.0, 4.0, 9.0, 16.0]:
        prepare_dataset(
            expansion_factor,
            reset_folders=False
        )

if __name__ == "__main__":
    main()