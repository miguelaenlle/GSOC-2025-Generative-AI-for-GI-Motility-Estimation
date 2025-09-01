# Author: Miguel Aenlle

import os
import re
import cv2
import json
import pickle
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


from sklearn.metrics import mean_absolute_error, average_precision_score
from scipy.stats import entropy
import numpy as np

masked_images_folder_path = '../../../0 - Data Exploration & Analysis/GI-Roberta/gi-roberta-dataset/full_dataset'

shape = (256, 256)

styled_img_folders = [
    '../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-027',
    '../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-029',
    '../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-030',
    '../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-031',
    '../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_FD-032'
]

# Specify styled_img_paths manually if you would like to acquire performance statistics across more than 1 
# pickle file per subject.

# You can find the generated images in 1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_<Subject Name>. 
# We recommend using the one with the largest number of synthesized samples.

styled_img_paths = [
]

def find_largest_samples_file(folder_path):
    max_num = -1
    max_file = None

    for fname in os.listdir(folder_path):
        if "samples_" not in fname:
            continue

        # Split off everything before 'samples_'
        tail = fname.split("samples_", 1)[1]

        # Now collect leading digits
        num_str = ""
        for ch in tail:
            if ch.isdigit():
                num_str += ch
            else:
                break

        if not num_str:
            continue

        num = int(num_str)
        if num > max_num:
            max_num, max_file = num, fname

    return max_file

if len(styled_img_paths) == 0:
    for folder in styled_img_folders:
        subfolders = sorted(os.listdir(folder), reverse = True)
        assert len(subfolders) > 0
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder, subfolder)
            largest = find_largest_samples_file(subfolder_path)
            if not largest:
                continue
            styled_img_paths.append(os.path.join(subfolder_path, largest))
            break

export_path_nonzero_masks = './tmp/nonzero_masks'
export_path_zero_masks = './tmp/zero_masks'
export_path_low_ssim = './tmp/low_ssim'
export_path_high_ssim = './tmp/high_ssim'
export_path_limited_mask = './tmp/limited_masks'

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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from scipy.linalg import sqrtm
import numpy as np

# 1) Preprocessing: resize to 299×299, convert to tensor, normalize like Inception expects
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 2) Load Inception v3 (pretrained) and strip off the final fc so it returns 2048‐d features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inception = models.inception_v3(pretrained=True, transform_input=False)
inception.fc = nn.Identity()
inception.eval().to(device)


def distribution_alignment(orig_gray: np.ndarray, synth_gray: np.ndarray,
                           bins: int = 256) -> float:
    """
    Compute the (symmetrized) KL-divergence between the intensity histograms
    of original and synthetic images as a measure of distribution alignment.
    Smaller is better.
    """
    h1, _ = np.histogram(orig_gray.ravel(), bins=bins, range=(0,255), density=True)
    h2, _ = np.histogram(synth_gray.ravel(), bins=bins, range=(0,255), density=True)
    # add small epsilon to avoid zeros
    h1 += 1e-8;  h2 += 1e-8
    kl12 = entropy(h1, h2)
    kl21 = entropy(h2, h1)
    return 0.5 * (kl12 + kl21)

def single_image_fid(orig_img: np.ndarray,
                     synth_img: np.ndarray,
                     model=inception,
                     preprocess=preprocess) -> float:
    """
    Compute FID between one original and one synthetic image.
    Falls back to zero-covariance if only one sample is available.
    """
    # get 1×D activations for each image
    act1 = get_activations([orig_img], model)   # shape (1, 2048)
    act2 = get_activations([synth_img], model)  # shape (1, 2048)

    # means
    mu1 = act1.mean(axis=0)
    mu2 = act2.mean(axis=0)

    # covariances: if only one sample, force to zero matrix
    dim = mu1.shape[0]
    if act1.shape[0] < 2:
        sigma1 = np.zeros((dim, dim))
    else:
        sigma1 = np.cov(act1, rowvar=False)

    if act2.shape[0] < 2:
        sigma2 = np.zeros((dim, dim))
    else:
        sigma2 = np.cov(act2, rowvar=False)

    # now safe to take sqrtm
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

def map_mae(orig_mask: np.ndarray, synth_mask: np.ndarray) -> Tuple[float,float]:
    """
    Treat each pixel as a sample to compute:
     - MAP: average precision treating orig_mask>threshold as “ground truth”
     - MAE: mean absolute error between mask intensities.
    """
    # flatten and binarize ground truth
    y_true = (orig_mask.ravel() > 127).astype(int)
    y_score = synth_mask.ravel() / 255.0
    # if all y_true are zeros or ones, average_precision_score will warn; handle fallback
    try:
        ap = average_precision_score(y_true, y_score)
    except ValueError:
        ap = float('nan')
    mae = mean_absolute_error(orig_mask.ravel()/255.0, y_score)
    return ap, mae

def miou(orig_mask: np.ndarray, synth_mask: np.ndarray) -> float:
    """
    Compute the mean Intersection-over-Union between two binary masks.
    """
    m1 = (orig_mask > 127).astype(int)
    m2 = (synth_mask > 127).astype(int)
    intersection = np.logical_and(m1, m2).sum()
    union        = np.logical_or(m1, m2).sum()
    if union == 0:
        return float('nan')
    return intersection / union


def get_activations(images, model, batch_size=32):
    """
    images: list of H×W×3 uint8 RGB NumPy arrays
    returns: N×2048 array of features
    """
    acts = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            # preprocess each image
            batch_t = torch.stack([preprocess(img) for img in batch]).to(device)
            feat = model(batch_t)                 # shape [B,2048]
            acts.append(feat.cpu().numpy())
    return np.concatenate(acts, axis=0)

def calculate_fid(orig_imgs, synth_imgs):
    # 1) Extract activations
    act1 = get_activations(orig_imgs, inception)
    act2 = get_activations(synth_imgs, inception)

    # 2) Compute mean & covariance
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # 3) Compute Frechet distance
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # numerical fix

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def main():
    shutil.rmtree(export_path_low_ssim, ignore_errors=True)
    os.makedirs(export_path_low_ssim, exist_ok=True)

    shutil.rmtree(export_path_nonzero_masks, ignore_errors=True)
    os.makedirs(export_path_nonzero_masks, exist_ok=True)

    shutil.rmtree(export_path_high_ssim, ignore_errors=True)
    os.makedirs(export_path_high_ssim, exist_ok=True)

    shutil.rmtree(export_path_limited_mask, ignore_errors=True)
    os.makedirs(export_path_limited_mask, exist_ok=True)

    filename_images = {}
    filename_masks = {}
    masked_images = os.listdir(masked_images_folder_path)

    for filename in masked_images:
        if not filename.endswith('.png') and not filename.endswith('.jpg'):
            continue
            
        img_path = os.path.join(masked_images_folder_path, filename)
        mask_path = os.path.join(masked_images_folder_path, filename.replace('-image', '-mask'))

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        filename_images[filename] = image
        filename_masks[filename] = mask

    # Clear the export path directories if they exists
    if os.path.exists(export_path_nonzero_masks):
        shutil.rmtree(export_path_nonzero_masks)

    os.makedirs(export_path_nonzero_masks, exist_ok=True)

    if os.path.exists(export_path_zero_masks):
        shutil.rmtree(export_path_zero_masks)
        

    os.makedirs(export_path_zero_masks, exist_ok=True)

    shutil.rmtree(export_path_high_ssim)
    os.makedirs(export_path_high_ssim, exist_ok=True)

    shutil.rmtree(export_path_low_ssim)
    os.makedirs(export_path_low_ssim, exist_ok=True)

    shutil.rmtree(export_path_limited_mask)
    os.makedirs(export_path_limited_mask, exist_ok=True)

    num_nonzero_masks = 0
    num_masks = 0

    mask_min = 0
    mask_max = 0

    num_masks_high_ssim = 0
    num_masks_cohesive = 0

    ssim_values = []
    psnr_values = []

    original_images = []
    synthetic_images = []

    dist_values = []
    sifid_values = []
    map_values = []
    iou_values = []

    num_images_for_each_cv_subject = {}

    HIGH_SSIM_THRESHOLD = 0.5
    LOW_SSIM_THRESHOLD = 0.1


    LIMIT_PER_SUBJECT = 400

    images_per_subject = {}

    for styled_img_path in tqdm(styled_img_paths):
        with open(styled_img_path, "rb") as f:
            styled_images = pickle.load(f)

        for key, styled_list in tqdm(styled_images.items()):
            # styled_image = styled_list[0]
            key_split=key.split('/')
            for _ in tqdm(styled_list):
                # Split synthetic image and mask
                cv_subject = key_split[-3]
                images_per_subject[cv_subject] = images_per_subject.get(cv_subject, 0) + 1

    timestamp = pd.Timestamp.now().isoformat()
    index = 0

    statistics_all = []

    for styled_img_path in tqdm(styled_img_paths):
        with open(styled_img_path, "rb") as f:
            styled_images = pickle.load(f)
        # Randomly select 1500 styled images
        
        for key, styled_list in tqdm(styled_images.items()):
            # styled_image = styled_list[0]
            for styled_image in tqdm(styled_list):
                # Split synthetic image and mask
                synthetic_img = styled_image[:, :, :3].astype(np.uint8)
                synthetic_gray = cv2.cvtColor(np.asarray(synthetic_img), cv2.COLOR_RGB2GRAY)

                mask = styled_image[:, :, 3]

                mask_min = min(mask_min, mask.flatten().min())
                mask_max = max(mask_max, mask.flatten().max())

                mask_normalized = mask / 255.0  # Normalize to [0,1] for transparency
                
                filename = key.split('/')[-1]  # Extract filename

                original_image = filename_images.get(filename, None)
                if original_image is None:
                    print(f"Warning: No original image found for {filename}. Skipping.")
                    continue

                original_mask = filename_masks.get(filename, None)
                if original_mask is None:
                    print(f"Warning: No original mask found for {filename}. Skipping.")
                    continue
                
                has_mask = mask_normalized.flatten().sum() > 5
                if has_mask:
                    num_nonzero_masks += 1
                num_masks += 1

                export_path = export_path_nonzero_masks if has_mask else export_path_zero_masks

                ssim_value = ssim(
                    np.array(original_image),
                    synthetic_gray,
                    full=True
                )[0]

                cv_subject = '-'.join(filename.split('-')[:2])

                num_images_for_each_cv_subject[cv_subject] = num_images_for_each_cv_subject.get(cv_subject, 0) + 1

                ssim_values.append({'ssim_value': ssim_value, 'cv_subject': cv_subject})

                psnr = cv2.PSNR(
                    np.array(original_image),
                    synthetic_gray
                )

                psnr_values.append({'psnr_value': psnr, 'cv_subject': cv_subject})

                orig_gray = np.array(original_image)
                synth_gray = synthetic_gray  # already gray


                # 1. Distribution alignment (KL divergence)
                dist_align = distribution_alignment(orig_gray, synth_gray)

                # 2. Single-image FID
                # convert both to RGB uint8 arrays
                orig_rgb = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2RGB)
                synth_rgb = cv2.cvtColor(synth_gray, cv2.COLOR_GRAY2RGB)
                sifid = single_image_fid(orig_rgb, synth_rgb)

                original_mask = np.array(original_mask)

                # 3. MAP and MAE on masks
                orig_mask_arr = cv2.cvtColor(np.array(original_mask), cv2.COLOR_BGR2GRAY) if original_mask.ndim==3 else original_mask
                synth_mask_arr = (mask_normalized * 255).astype(np.uint8)
                map_score, mae_score = map_mae(orig_mask_arr, synth_mask_arr)

                # 4. mIoU on masks
                iou_score = miou(orig_mask_arr, synth_mask_arr)

                dist_values.append({'dist_align': dist_align, 'cv_subject': cv_subject})
                sifid_values.append({'sifid': sifid, 'cv_subject': cv_subject})
                map_values.append({'map': map_score, 'mae': mae_score, 'cv_subject': cv_subject})
                iou_values.append({'miou': iou_score, 'cv_subject': cv_subject})

                export_paths = [export_path]

                if (ssim_value > HIGH_SSIM_THRESHOLD):
                    num_masks_high_ssim += 1
                    export_paths.append(export_path_high_ssim)

                elif (ssim_value < LOW_SSIM_THRESHOLD):
                    export_paths.append(export_path_low_ssim)
                
                mask_bool = mask > 200
                percent_mask_highlighted = np.sum(mask_bool) / mask_bool.size 
                is_cohesive_mask_image, cohesive_mask = is_cohesive_mask(mask_normalized, thresh_method='otsu', min_area=50, main_frac_thresh=0.6, max_extra_components=1)
                if percent_mask_highlighted < 0.5 and is_cohesive_mask_image:
                    num_masks_cohesive += 1
                    export_paths.append(export_path_limited_mask)

                statistics_all.append({
                    'cv_subject': cv_subject,
                    'key': key,
                    'styled_img_path': styled_img_path,
                    'ssim': ssim_value,
                    'psnr': psnr,
                    'dist_align': dist_align,
                    'sifid': sifid,
                    'map': map_score,
                    'mae': mae_score,
                    'miou': iou_score,
                    'is_cohesive_mask_image': is_cohesive_mask_image,
                })

                for path in export_paths:
                    # Save original synthetic image
                    orig_filename = os.path.join(path, f"{filename}_synthetic.png")
                    cv2.imwrite(orig_filename, cv2.cvtColor(synthetic_img, cv2.COLOR_RGB2BGR))

                    # Save the raw mask
                    mask_filename = os.path.join(path, f"{filename}_synthetic_mask.png")
                    cv2.imwrite(mask_filename, (mask_normalized * 255).astype(np.uint8))
                    # num_limited_masks+= 1

                # Create and save the overlay
                fig, axes = plt.subplots(1, 4, figsize=(24, 12))
                axes[0].imshow(synthetic_img, cmap='gray')
                axes[0].set_title(f"Synthetic: {filename} - SSIM {ssim_value:.4f}", fontsize=10)
                axes[0].axis('off')

                axes[1].imshow(synthetic_img, cmap='gray')
                axes[1].imshow(mask_normalized * 255, cmap='jet', alpha=0.5)
                axes[1].set_title(f"Synthetic + Mask Overlay: {filename}", fontsize=10)
                axes[1].axis('off')
                
                axes[2].imshow(original_image, cmap='gray')
                axes[2].set_title(f"Original: {filename}", fontsize=10)
                axes[2].axis('off')

                axes[3].imshow(original_image, cmap='gray')
                axes[3].imshow(original_mask, cmap='jet', alpha=0.5)
                axes[3].set_title(f"Original + Mask Overlay: {filename}", fontsize=10)
                axes[3].axis('off')

                overlay_filename = os.path.join(export_path, f"{filename}_overlay.png")
                plt.tight_layout()
                plt.savefig(overlay_filename, bbox_inches='tight')

                for path in export_paths:
                    overlay_filename = os.path.join(path, f"{filename}_overlay.png")
                    plt.savefig(overlay_filename, bbox_inches='tight')

                plt.close(fig)

                # Convert original and synthetic gray images to RGB
                original_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_GRAY2RGB)
                synthetic_gray = cv2.cvtColor(synthetic_gray, cv2.COLOR_GRAY2RGB)

                original_images.append({'original_image': original_image, 'cv_subject': cv_subject})
                synthetic_images.append({'synthetic_image': synthetic_gray, 'cv_subject': cv_subject})
                index += 1
                if index % 10 == 0:
                    pd.DataFrame(statistics_all).to_csv(f'tmp/diffusion_statistics_{timestamp}.csv')

                    ssim_values_df = pd.DataFrame(ssim_values)
                    psnr_values_df = pd.DataFrame(psnr_values)

                    # Acquire means per cv_subject and in aggregate for both DFs
                    ssim_means = ssim_values_df.groupby('cv_subject').mean().reset_index()
                    ssim_means = ssim_means.set_index('cv_subject')['ssim_value'].to_dict()
                    
                    psnr_means = psnr_values_df.groupby('cv_subject').mean().reset_index()
                    psnr_means = psnr_means.set_index('cv_subject')['psnr_value'].to_dict()

                    ssim_means_agg = ssim_values_df['ssim_value'].mean()
                    psnr_means_agg = psnr_values_df['psnr_value'].mean()

                    num_ssim_values_above_threshold = ssim_values_df[ssim_values_df['ssim_value'] > HIGH_SSIM_THRESHOLD].shape[0]
                    num_ssim_values_below_threshold = ssim_values_df[ssim_values_df['ssim_value'] < LOW_SSIM_THRESHOLD].shape[0]

                    statistics = {
                        'num_synthetic_images': len(synthetic_images),
                        'num_raw_images': len(original_images),
                        'num_nonzero_masks': num_nonzero_masks,
                        'num_masks': num_masks,
                        'num_cohesive_masks': num_masks_cohesive,
                        'num_masks_high_ssim': num_masks_high_ssim,
                        'ssim_means': ssim_means,
                        'psnr_means': psnr_means,
                        'ssim_means_agg': float(ssim_means_agg),
                        'psnr_means_agg': float(psnr_means_agg),
                        'num_ssim_values_above_threshold': num_ssim_values_above_threshold,
                        'num_ssim_values_below_threshold': num_ssim_values_below_threshold,
                        'num_images_for_each_cv_subject': num_images_for_each_cv_subject
                    }


                    dist_df   = pd.DataFrame(dist_values)
                    sifid_df  = pd.DataFrame(sifid_values)
                    map_df    = pd.DataFrame(map_values)
                    iou_df    = pd.DataFrame(iou_values)

                    # 1) Distribution alignment
                    dist_means       = dist_df.groupby('cv_subject')['dist_align'].mean().to_dict()
                    dist_mean_agg    = float(dist_df['dist_align'].mean())

                    # 2) Single-image FID
                    sifid_means      = sifid_df.groupby('cv_subject')['sifid'].mean().to_dict()
                    sifid_mean_agg   = float(sifid_df['sifid'].mean())

                    # 3) MAP & MAE
                    map_means        = map_df.groupby('cv_subject')['map'].mean().to_dict()
                    mae_means        = map_df.groupby('cv_subject')['mae'].mean().to_dict()
                    map_mean_agg     = float(map_df['map'].mean())
                    mae_mean_agg     = float(map_df['mae'].mean())

                    # 4) mIoU
                    iou_means        = iou_df.groupby('cv_subject')['miou'].mean().to_dict()
                    iou_mean_agg     = float(iou_df['miou'].mean())

                    # ——— extend your statistics dict ———
                    statistics.update({
                        'dist_means':            dist_means,
                        'dist_mean_agg':         dist_mean_agg,
                        'sifid_means':           sifid_means,
                        'sifid_mean_agg':        sifid_mean_agg,
                        'map_means':             map_means,
                        'mae_means':             mae_means,
                        'map_mean_agg':          map_mean_agg,
                        'mae_mean_agg':          mae_mean_agg,
                        'miou_means':            iou_means,
                        'miou_mean_agg':         iou_mean_agg,
                    })

                        # Save to tmp/diffusion_statistics.json
                    with open(f'./tmp/diffusion_statistics_{timestamp}.json', 'w') as f:
                        json.dump(statistics, f, indent=4)
if __name__ == "__main__":
    main()
