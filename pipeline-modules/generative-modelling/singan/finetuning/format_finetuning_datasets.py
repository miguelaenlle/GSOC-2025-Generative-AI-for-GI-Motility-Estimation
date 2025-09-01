# Author: Miguel Aenlle

import os
import shutil
import random
import shutil
from collections import deque

def main():
    singan_datasets_folder = '../../../../GI/1.5 - Synthetic Data Generation/Singan-Seg/unet_singan_augmented_datasets_2'
    SELF_SUPERVISED = True
    RANDOM_SAMPLES_FOLDER = '../../../../GI/1.5 - Synthetic Data Generation/Singan-Seg/Output/RandomSamples' if not SELF_SUPERVISED else '../../../../GI/1.5 - Synthetic Data Generation/Singan-Seg/Output/FinetuningSamples/RandomSamples'

    mask_image_counts = []
    for dataset in os.listdir(singan_datasets_folder):
        expansion_factor = float(dataset.split('expansion_factor_')[-1])
        dataset_path = os.path.join(singan_datasets_folder, dataset)
        for cv_subject in os.listdir(dataset_path):
            finetuning_folder_path = os.path.join(dataset_path, cv_subject, 'finetuning')
            files = os.listdir(finetuning_folder_path)
            num_image_files = 0
            num_mask_files = 0
            for f in files:
                if 'mask' in f:
                    num_mask_files += 1
                else:
                    num_image_files += 1
            mask_image_counts.append({
                'dataset': dataset,
                'cv_subject': cv_subject,
                'num_image_files': num_image_files,
                'num_mask_files': num_mask_files
            })
            print(f"Dataset: {dataset}, CV Subject: {cv_subject}, Images: {num_image_files}, Masks: {num_mask_files}")
            assert num_image_files == num_mask_files, f"Mismatch in number of image and mask files in {finetuning_folder_path}"
    
    for dataset in os.listdir(singan_datasets_folder):
        expansion_factor = float(dataset.split('expansion_factor_')[-1])
        dataset_path = os.path.join(singan_datasets_folder, dataset)
        for cv_subject in os.listdir(dataset_path):
            finetuning_augmented_folder_path = os.path.join(dataset_path, cv_subject, 'finetuning_augmented')
            # Clear finetuning augmented folder if it exists
            if os.path.exists(finetuning_augmented_folder_path):
                shutil.rmtree(finetuning_augmented_folder_path)
            os.makedirs(finetuning_augmented_folder_path, exist_ok=True)

    CONSTANT_EXPANSION_FACTOR = 10 # Set to None if you want to use the expansion factor from the dataset name

    file_transfers = []
    images_missing_samples = []
    for dataset in os.listdir(singan_datasets_folder):
        expansion_factor = CONSTANT_EXPANSION_FACTOR if CONSTANT_EXPANSION_FACTOR else float(dataset.split('expansion_factor_')[-1])
        dataset_path = os.path.join(singan_datasets_folder, dataset)
        for cv_subject in os.listdir(dataset_path):
            finetuning_folder_path = os.path.join(dataset_path, cv_subject, 'finetuning')
            finetuning_augmented_folder_path = os.path.join(dataset_path, cv_subject, 'finetuning_augmented')
            # Clear finetuning augmented folder if it exists
            if os.path.exists(finetuning_augmented_folder_path):
                shutil.rmtree(finetuning_augmented_folder_path)
            os.makedirs(finetuning_augmented_folder_path, exist_ok=True)

            # 1) Gather and shuffle generated images per original file
            finetuning_image_paths_per_file = {}
            for fn in os.listdir(finetuning_folder_path):
                if 'image' not in fn:
                    continue
                base_name = os.path.splitext(fn)[0]
                matching = [d for d in os.listdir(RANDOM_SAMPLES_FOLDER) if base_name in d]
                if not matching:
                    print(f"No matching random samples found for {fn} in {RANDOM_SAMPLES_FOLDER}. Skipping.")
                    images_missing_samples.append({
                        'file_name': fn,
                        'basename': base_name,
                        'cv_subject': cv_subject,
                        'dataset': dataset
                    })
                    continue
                image_dir = os.path.join(RANDOM_SAMPLES_FOLDER, matching[0], 'gen_start_scale=0')
                paths = [
                    os.path.join(image_dir, f)
                    for f in os.listdir(image_dir)
                    if '_img' in f
                ]
                if paths:
                    random.shuffle(paths)
                    finetuning_image_paths_per_file[fn] = paths

            # 2) Compute how many new samples we need total
            dataset_expansion_amount = (len(os.listdir(finetuning_folder_path)) / 2) * expansion_factor

            # 3) Cycle through files, picking one image at a time from each
            selected_images = []
            queue = deque(finetuning_image_paths_per_file.keys())

            while len(selected_images) < dataset_expansion_amount and queue:
                fn = queue.popleft()
                paths = finetuning_image_paths_per_file[fn]
                if paths:
                    # take the next image
                    selected_images.append(paths.pop(0))
                    # if there are more for this file, re‑enqueue it
                    if paths:
                        queue.append(fn)

            # 4) Copy exactly the selected images
            for image_path in selected_images:
                mask_path = image_path.replace('_img.png', '_mask.png')

                original_image_name = image_path.split('/')[-3]
                image_basename = original_image_name + '-' + os.path.basename(image_path).replace('_img.png', '_image.png')

                mask_basename = image_basename.replace('_image.png', '_mask.png')

                print(image_basename, mask_basename)

                # Copy the image path
                shutil.copy(image_path, os.path.join(finetuning_augmented_folder_path, image_basename))
                file_transfers.append({
                    'original_path': image_path,
                    'new_path': os.path.join(finetuning_augmented_folder_path, image_basename   )
                })
                # Copy the mask path
                shutil.copy(mask_path, os.path.join(finetuning_augmented_folder_path, mask_basename))
                file_transfers.append({
                    'original_path': mask_path,
                    'new_path': os.path.join(finetuning_augmented_folder_path, mask_basename)
                })

            # 5) Copy all images in finetuning into finetuning_augmented
            for fn in os.listdir(finetuning_folder_path):
                if fn.endswith('image.png') or fn.endswith('mask.png'):
                    shutil.copy(os.path.join(finetuning_folder_path, fn), os.path.join(finetuning_augmented_folder_path, fn))
                    file_transfers.append({
                        'original_path': os.path.join(finetuning_folder_path, fn),
                        'new_path': os.path.join(finetuning_augmented_folder_path, fn)
                    })

if __name__ == "__main__":
    main()