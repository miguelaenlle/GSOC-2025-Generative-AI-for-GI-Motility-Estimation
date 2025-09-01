# Author: Miguel Aenlle

import os
import shutil
import ntpath
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from PIL import Image
from tqdm import tqdm


import cv2
import numpy as np
from typing import Tuple

def is_cohesive_mask(mask_gray: np.ndarray,
                     thresh_method: str = 'otsu',
                     min_area: int = 50,
                     main_frac_thresh: float = 0.6,
                     max_extra_components: int = 1
                    ) -> Tuple[bool, np.ndarray]:
    """
    Decide whether a grayscale mask is coherent (one big region) or incohesive
    (many small regions), *and* return a mask image that contains only the
    dominant region(s).

    Returns
    -------
    coherent : bool
        True if one blob covers ≥ `main_frac_thresh` of all (filtered) mask pixels
        and there are at most `max_extra_components` others.
    mask_filtered : np.ndarray
        8‑bit binary image (0 or 255) where only the largest region and up to
        `max_extra_components` next‑largest regions are kept.
    """
    # ——— ensure uint8 [0,255] ———
    if mask_gray.dtype in (np.float32, np.float64):
        img = (mask_gray * 255).astype(np.uint8)
    else:
        img = mask_gray.astype(np.uint8)

    # ——— binarize ———
    if thresh_method == 'otsu':
        _, bw = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresh_method == 'adaptive':
        bw = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   blockSize=11, C=2)
    else:  # 'fixed'
        _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # ——— connected components ———
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw,
                                                                  connectivity=8)
    # stats[i, cv2.CC_STAT_AREA] is area of label i
    # skip stats[0] (background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    labels_list = np.arange(1, n_labels)

    # ——— filter out tiny noise blobs ———
    keep = areas >= min_area
    if not np.any(keep):
        # no component big enough
        empty_mask = np.zeros_like(bw)
        return False, empty_mask

    filtered_labels = labels_list[keep]
    filtered_areas = areas[keep]
    total_area = filtered_areas.sum()

    # ——— sort by descending area ———
    idx_sorted = np.argsort(-filtered_areas)
    sorted_labels = filtered_labels[idx_sorted]

    # ——— select top regions ———
    keep_labels = sorted_labels[: 1 + max_extra_components]

    # ——— build filtered mask ———
    mask_filtered = np.isin(labels, keep_labels).astype(np.uint8) * 255

    # ——— decision ———
    largest_area = filtered_areas[idx_sorted[0]]
    num_extras = len(filtered_labels) - 1
    coherent = (largest_area / total_area) >= main_frac_thresh and \
               num_extras <= max_extra_components

    return coherent, mask_filtered

def main():
    # Replace with paths to PKL files of synthesized images
    # Synthesize images with train_diffusion_models_and_synthesize_images.py

    os.chdir(os.path.abspath('../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/testing_notebooks'))

    original_dataset_folder_path = '../../../../1.5 - Synthetic Data Generation/Singan-Seg/unet_singan_augmented_datasets_2/augmented_dataset_expansion_factor_0'

    paths = {
        'FD-027': ['../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-027/diffgen-2025-07-13-00-54/styled_samples_400x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-027/diffgen-2025-07-23-01-29/styled_samples_260x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-027/diffgen-2025-07-12-00-38/styled_samples_183x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-029/diffgen-2025-07-12-03-48/styled_samples_180x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-027/diffgen-2025-07-09-13-50/styled_samples_183x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-027/diffgen-2025-06-29-03-16/styled_samples_288x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-027/diffgen-2025-06-24-19-17/styled_samples_288x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-027/diffgen-2025-07-23-10-39/styled_samples_480x256x256x4.pkl'],
        'FD-029': ['../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-029/diffgen-2025-07-13-07-47/styled_samples_560x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-029/diffgen-2025-07-12-03-48/styled_samples_185x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-029/diffgen-2025-06-29-18-09/styled_samples_288x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-029/diffgen-2025-06-25-09-26/styled_samples_288x256x256x4.pkl'],
        'FD-030': ['../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-030/diffgen-2025-07-22-01-14/styled_samples_1701x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-030/diffgen-2025-06-25-04-45/styled_samples_288x256x256x4.pkl'],
        'FD-031': ['../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-031/diffgen-2025-07-13-17-25/styled_samples_860x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-031/diffgen-2025-06-29-08-13/styled_samples_288x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-031/diffgen-2025-06-24-23-57/styled_samples_288x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-031/diffgen-2025-07-23-10-37/styled_samples_258x256x256x4.pkl'],
        'FD-032': ['../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-032/diffgen-2025-07-15-02-07/styled_samples_1503x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-032/diffgen-2025-07-12-07-05/styled_samples_183x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-032/diffgen-2025-07-11-13-57/styled_samples_183x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-032/diffgen-2025-07-11-13-57/styled_samples_183x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-032/diffgen-2025-07-08-21-16/styled_samples_183x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-032/diffgen-2025-06-28-21-00/styled_samples_288x256x256x4.pkl', '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-032/diffgen-2025-06-24-14-34/styled_samples_288x256x256x4.pkl']
    }


    dataset_export_paths = {
        'mask': '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/augmented_datasets_varied_expansion_factor',
        'ssim': '../../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/augmented_datasets_varied_expansion_factor_ssim'
    }

    dataset_folder = '../augmented_datasets_varied_expansion_factor_mask'
    
    shutil.rmtree(dataset_folder, ignore_errors=True)

    os.makedirs(dataset_folder, exist_ok=True)
    masked_images_folder_path = '../../../../0 - Data Exploration & Analysis/GI-Roberta/gi-roberta-dataset/full_dataset'

    original_filename_images = {}
    original_filename_masks = {}
    masked_images = os.listdir(masked_images_folder_path)

    for filename in masked_images:
        if not filename.endswith('.png') and not filename.endswith('.jpg'):
            continue
            
        img_path = os.path.join(masked_images_folder_path, filename)
        mask_path = os.path.join(masked_images_folder_path, filename.replace('-image', '-mask'))

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        original_filename_images[filename] = image
        original_filename_masks[filename] = mask

    for mode, folder in dataset_export_paths.items():
        for expansion_factor_folder in os.listdir(folder):
            print(mode)
            print(expansion_factor_folder)
            expansion_factor_path = os.path.join(folder, expansion_factor_folder)
            for cv_subject in os.listdir(expansion_factor_path):
                cv_subject_folder = os.path.join(expansion_factor_path, cv_subject, 'train')
                num_in_folder = len([file for file in os.listdir(cv_subject_folder) if 'image' in file])
                num_synthetic_images_in_folder = len([file for file in os.listdir(cv_subject_folder) if 'synthetic-image' in file])
                print(f"Number of images in {cv_subject}: {num_in_folder}")
                print(f"Number of synthetic images in {cv_subject}: {num_synthetic_images_in_folder}")

    MODE = 'mask' # 'ssim' | 'mask'

    expansion_factors_mode = {
        'ssim': [0.5, 0.75],
        'mask': [0.5]
    }

    for expansion_factor in expansion_factors_mode[MODE]:
        expansion_factor_folder = os.path.join(dataset_folder, f'augmented_dataset_expansion_factor_{expansion_factor}')
        augmentation_amount = {}
        subject_number = {}
        for subject in tqdm(list(paths.keys())):
            subject_paths = paths[subject]

            selected_images = []
            selected_images_per_subject = {}
            
            cv_subject_folder = os.path.join(expansion_factor_folder, subject)
            train_folder = os.path.join(cv_subject_folder, 'train')
            os.makedirs(train_folder, exist_ok=True)
            val_folder = os.path.join(cv_subject_folder, 'val')
            os.makedirs(val_folder, exist_ok=True)
            
            # Copy over all the images for the CV-subject, originally. 
            original_cv_subject_folder = os.path.join(original_dataset_folder_path, subject)
            original_train_folder = os.path.join(original_cv_subject_folder, 'train')
            original_val_folder = os.path.join(original_cv_subject_folder, 'val')

            # Copy all the original images to the new folder
            for original_img_path in os.listdir(original_train_folder):
                original_img_full_path = os.path.join(original_train_folder, original_img_path)
                if original_img_path.endswith('.png'):
                    shutil.copy(original_img_full_path, train_folder)
            
            for original_img_path in os.listdir(original_val_folder):
                original_img_full_path = os.path.join(original_val_folder, original_img_path)
                if original_img_path.endswith('.png'):
                    shutil.copy(original_img_full_path, val_folder)

            for path in subject_paths:
                # Load the augmented data
                subject_image_data = pickle.load(open(path, 'rb'))

                # Select at most expansion factor random images to use for the augmentation, selecting evenly from the subject image data.

                for original_img_path in subject_image_data.keys():
                    subject_img_data_list = subject_image_data[original_img_path]
                    for img in subject_img_data_list:
                        original_img_name = os.path.basename(original_img_path)
                        original_image = original_filename_images.get(original_img_name, None)
                        
                        original_img_name_count = subject_number.get(original_img_name, 1)

                        synthetic_image_path = os.path.join(train_folder, f"{original_img_name.replace('-image.png', f'-{original_img_name_count}-synthetic-image.png')}")
                        synthetic_mask_path = os.path.join(train_folder, f"{original_img_name.replace('-image.png', f'-{original_img_name_count}-synthetic-mask.png')}")

                        subject_number[original_img_name] = original_img_name_count + 1

                        # Ensure the augmented data is of a high quality. 
                        synthetic_mask = img[:, :, 3]
                        synthetic_image = img[:, :, :3]

                        synthetic_gray = cv2.cvtColor(np.asarray(synthetic_image), cv2.COLOR_RGB2GRAY)

                        # black out everything under 200px and make everything else white
                        mask_bool = synthetic_mask > 200
                        percent_mask_highlighted = np.sum(mask_bool) / mask_bool.size 

                        criteria_met = False
                        if MODE == 'ssim':
                            ssim_value = ssim(
                                np.array(original_image),
                                synthetic_gray,
                                full=True
                            )[0]
                            criteria_met = ssim_value > 0.5
                        elif MODE == 'mask':
                            is_cohesive_mask_image, synthetic_mask = is_cohesive_mask(synthetic_mask)
                            criteria_met = percent_mask_highlighted < 0.5 and is_cohesive_mask_image

                        if criteria_met:
                            selected_images_per_subject.setdefault(original_img_name, []).append((synthetic_image, synthetic_mask, synthetic_image_path, synthetic_mask_path))
                            selected_images.append((synthetic_image, synthetic_mask, synthetic_image_path, synthetic_mask_path))
                    
            num_images = len([file for file in os.listdir(train_folder) if '-image.png' in file])
            target_num_images = int(expansion_factor * num_images)

            subjects = list(selected_images_per_subject.items())

            print(len(subjects), len(selected_images))

            n_subjects = len(subjects)
            if n_subjects == 0 or target_num_images <= 0:
                final_selected = []
            elif abs(len(selected_images) - target_num_images) < 50:
                # If we are close enough to the target number of images, just use what we have.
                if (len(selected_images) < target_num_images):
                    final_selected = selected_images
                else:
                    # Randomly select the target number of images from the selected images
                    final_selected = random.sample(selected_images, target_num_images)
            else:
                # base quota per subject
                base_quota = target_num_images // n_subjects
                # leftover slots to distribute one‐by‐one
                leftover   = target_num_images - base_quota * n_subjects

                final_selected = []
                for original_img_name, items in subjects:
                    # how many to take from this subject
                    take = base_quota + (1 if leftover > 0 else 0)
                    take = min(take, len(items))     # don’t exceed available
                    if leftover > 0:
                        leftover -= 1
                    # random.sample will fail if take == 0, so guard it
                    if take:
                        final_selected.extend(random.sample(items, take))

                # in case we somehow overshot (due to subjects with fewer samples),
                # just trim to target_num_images
                final_selected = final_selected[:target_num_images]
            for synthetic_image, synthetic_mask, synthetic_image_path, synthetic_mask_path in final_selected:
                # Export the mask to the location
                cv2.imwrite(synthetic_mask_path, synthetic_mask)

                # Export the image to the location
                cv2.imwrite(synthetic_image_path, synthetic_image)

            augmentation_amount[subject] = augmentation_amount.get(subject, 0) + len(final_selected) # Num augmented images used
        print('augmentation_amount', augmentation_amount)
    breakpoint()

if __name__ == "__main__":
    main()

