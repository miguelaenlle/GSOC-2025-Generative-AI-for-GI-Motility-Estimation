# Author: Miguel Aenlle

# Retrieve finetuning images directly, placing them into the dataset

import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

def main():
    # The new directory we will create.
    NUM_IMAGES = 10

    SELF_SUPERVISED = False
    FINETUNING_AUGMENTATION = False

    BASE_DIR = '../../../../1.5 - Synthetic Data Generation/Singan-Seg'

    parent_directory = f'{BASE_DIR}/unet_singan_augmented_datasets_with_finetuning{'_num_images_' + str(NUM_IMAGES) if NUM_IMAGES else ''}{'_self_supervised' if SELF_SUPERVISED else ''}{'_no_finetuning_augmentation' if not FINETUNING_AUGMENTATION else ''}'
    original_directory = f'{BASE_DIR}/unet_singan_augmented_datasets'

    # Recursive clone the original directory to parent directory
    shutil.copytree(original_directory, parent_directory, dirs_exist_ok=True)

    for folder in tqdm(os.listdir(parent_directory)):
        dataset_folder = os.path.join(parent_directory, folder)
        for subject in os.listdir(dataset_folder):
            validation_folder = os.path.join(dataset_folder, subject, 'val')
            finetuning_folder = os.path.join(dataset_folder, subject, 'finetuning')

            os.makedirs(finetuning_folder, exist_ok=True)

            if not FINETUNING_AUGMENTATION:
                # We move everything in the finetuning folder into this folder for consistency
                finetuning_augmented_folder = os.path.join(dataset_folder, subject, 'finetuning_augmented')
                os.makedirs(finetuning_augmented_folder, exist_ok=True)

            num_samples_found = 0
            i = 1
            while (num_samples_found < NUM_IMAGES):
                filename = f'{subject}-slice-{str(i).zfill(2)}-image.png'
                mask_filename = f'{subject}-slice-{str(i).zfill(2)}-mask.png'
                i += 1
                sample_path = os.path.join(validation_folder, filename)
                mask_path = os.path.join(validation_folder, mask_filename)

                # Ensure the mask is not empty
                mask_data = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                mask_sum = np.array(mask_data).flatten().sum()

                if mask_sum < 10:
                    continue
                    
                if os.path.exists(sample_path):
                    shutil.copy(sample_path, finetuning_folder)
                    shutil.copy(mask_path, finetuning_folder)

                    if not FINETUNING_AUGMENTATION:
                        # Move the files to the finetuning_augmented folder
                        shutil.move(os.path.join(finetuning_folder, filename), finetuning_augmented_folder)
                        shutil.move(os.path.join(finetuning_folder, mask_filename), finetuning_augmented_folder)

                    # Remove the files from the validation folder
                    os.remove(sample_path)
                    os.remove(mask_path)

                    num_samples_found += 1
                
            print(f"Created finetuning folder at {finetuning_folder} with first {num_samples_found} samples from {validation_folder}.")
            print(f"Finetuning folder created at: {finetuning_folder}")

if __name__ == "__main__":
    main()