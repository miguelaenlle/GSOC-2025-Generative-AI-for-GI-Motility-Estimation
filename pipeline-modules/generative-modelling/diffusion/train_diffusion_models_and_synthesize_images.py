# Author: Miguel Aenlle

import os
import cv2
import json
import torch
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import io
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

def min_max_normalize(tensor, min_val=0.0, max_val=1.0):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * (max_val - min_val) + min_val
    return normalized_tensor

def resize_torch_tensor(tensor, w=256, h=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((w, h)),
    ])
    tensor = transform(tensor)
    tensor = tensor.float()

    tensor = min_max_normalize(tensor)

    return tensor

class RealAndSyntheticImageDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, synthetic_images_folder_path, real_images_folder_path):
        synthetic_images = os.listdir(synthetic_images_folder_path)

        real_images = os.listdir(real_images_folder_path)

        if len(synthetic_images) < num_samples:
            raise ValueError(f"Not enough synthetic images. Found {len(synthetic_images)}, but requested {num_samples}.")
        
        if len(real_images) < num_samples:
            raise ValueError(f"Not enough real images. Found {len(real_images)}, but requested {num_samples}.")
        
        synthetic_images_paths = [os.path.join(synthetic_images_folder_path, img) for img in synthetic_images[:(num_samples)]]
        real_images_paths = [os.path.join(real_images_folder_path, img) for img in real_images[:(num_samples)]]

        self.paths = []
        self.labels = []
        # Interleave the paths
        for i in range(min(len(synthetic_images_paths), len(real_images_paths))):
            self.paths.append(synthetic_images_paths[i])
            self.paths.append(real_images_paths[i])
            self.labels.append(1)
            self.labels.append(0)
    
        self.length = len(self.paths)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        # Return a random 1×266×266 tensor and a random binary label
        image_path = self.paths[idx]
        image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Resize and normalize the loaded data
        image_data = resize_torch_tensor(image_data)

        label = self.labels[idx]
        
        return image_data, label

from torchvision.ops import masks_to_boxes
from torchvision.io import read_image, ImageReadMode

def get_bbox_from_mask(mask: torch.Tensor, threshold: int = 0):
    """
    Computes the tightest bounding box around non-zero pixels in a 1-channel mask.

    Args:
        mask (Tensor or str):
            Either:
            - a torch Tensor of shape [H, W] or [1, H, W] (dtype uint8 or bool), or
            - a file path to a grayscale mask image.
        threshold (int):
            Pixel values > threshold are considered “foreground”.

    Returns:
        Tensor of shape [4]:
            (x_min, y_min, x_max, y_max), or
        None if no foreground pixels are found.
    """
    # 1) Load from disk if needed
    if isinstance(mask, str):
        mask = read_image(mask, mode=ImageReadMode.UNCHANGED)  # -> [1, H, W]

    # 2) Normalize shape -> [1, H, W]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 3 and mask.shape[0] != 1:
        # if you somehow loaded RGB, convert to single channel
        mask = torch.mean(mask.float(), dim=0, keepdim=True).to(torch.uint8)

    # 3) Binarize
    binary_mask = mask > threshold  # bool tensor [1, H, W]

    # 4) Compute boxes: returns Tensor[N,4] in (x1, y1, x2, y2)
    boxes = masks_to_boxes(binary_mask)

    if boxes.numel() == 0:
        return None
    
    return boxes

from PIL import Image
from torchvision.transforms import functional as TF

def shorten_key(key):
    return f".{key.split('diffusion-gen')[1]}"

def load_images_from_path(img_path, mask=False):
    conversion = "RGB" if not mask else "L"
    src_img_pil = Image.open(img_path).convert(conversion)
    src_img_pil = src_img_pil.resize(shape, Image.LANCZOS)  # type: ignore
    src_img = (
        TF.to_tensor(src_img_pil).unsqueeze(0)
    )
    return src_img
def prep_image_for_view(images, n=None):
    paths = []
    images_arr = []
    iters = images.items() if n is None else list(images.items())[:n]
    for path, imgs in iters:
        paths.append(path)
        images_arr.append(imgs)
        batch_size = len(imgs)
    images = np.concatenate(images_arr, axis=0)
    # images = np.load(img_path)['arr_0']
    print(images.shape) # (N, 256, 256, C)
    images, masks = images[..., :3], images[..., 3]
    # Erosion of mask
    threshold = (masks.min() + masks.max()) / 2
    print("Threshold", threshold)
    masks = np.where(masks > threshold, 255, 0).astype(np.uint8)

    return batch_size, images, masks, paths

def compute_ssim_mse(real_roi, synthetic_roi):
    """Compute SSIM and MSE between two ROIs."""
    real_gray = cv2.cvtColor(real_roi.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.uint8)
    synthetic_gray = cv2.cvtColor(synthetic_roi.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.uint8)

    ssim_value = ssim(real_gray, synthetic_gray)
    mse_value = np.mean((real_gray - synthetic_gray) ** 2)
    return ssim_value, mse_value

def synthesis_performance_benchmark_diffusion(
    custom_cv_subjects: list[str] = None,
    hq_samples_only: bool = True,
    datestamp: str = None,
    num_samples: int = 10, # per image
    skip_training: bool = True,
    gpu_id: int = 0,
    singan_augmented_dataset: bool = False,
    skip_validation: bool = False
):
    stomach_data_and_masks_dir_path = '../../../0 - Data Exploration & Analysis/GI-Roberta/gi-roberta-dataset/' + ('full_dataset' if not singan_augmented_dataset else 'full_dataset_augmented_hq') 
    
    available_filenames = [filename for filename in os.listdir(stomach_data_and_masks_dir_path) if 'mask' not in filename]

    # Associate the subjects of the filenames
    filenames_and_subjects = pd.DataFrame({
        'image_filename': available_filenames,
        'subject': ["-".join(filename.split("-")[:2]) for filename in available_filenames]
    })

    if not skip_training:
        cv_subjects = custom_cv_subjects if custom_cv_subjects else filenames_and_subjects['subject'].unique()
        for cv_val_subject in cv_subjects:
            filenames_and_subjects_train = filenames_and_subjects[filenames_and_subjects['subject'] != cv_val_subject]
            filenames_and_subjects_val_subject_train = filenames_and_subjects[filenames_and_subjects['subject'] == cv_val_subject]

            base_path = f'../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/guided_diffusion/segmented-images-roberta{'-aug' if singan_augmented_dataset else ''}/{cv_val_subject}'
            base_path_val = f"{base_path}_val"

            # Create the base path if it doesn't exist
            os.makedirs(base_path, exist_ok=True)

            masked_images_path = f'{base_path}/masked-images'
            masks_path = f'{base_path}/masks'

            masked_images_val_path = f'{base_path_val}/masked-images'
            masks_val_path = f'{base_path_val}/masks'

            if os.path.exists(masked_images_path):
                shutil.rmtree(masked_images_path)

            if os.path.exists(masks_path):
                shutil.rmtree(masks_path)

            if not skip_validation:
                if os.path.exists(masked_images_val_path):
                    shutil.rmtree(masked_images_val_path)

                if os.path.exists(masks_val_path):
                    shutil.rmtree(masks_val_path)

            os.makedirs(masked_images_path, exist_ok=True)
            os.makedirs(masks_path, exist_ok=True)

            os.makedirs(masked_images_val_path, exist_ok=True)
            os.makedirs(masks_val_path, exist_ok=True)

            # Filter out any empty mask training dataset images. 
            excluded_files = []
            for file in tqdm(filenames_and_subjects_train['image_filename'].unique()):
                mask_filename = file.replace('image.png', 'mask.png')

                mask_path = os.path.join(stomach_data_and_masks_dir_path, mask_filename)

                mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if (mask_image.flatten().max() == 0):
                    excluded_files.append(file)
                    continue
            
            for file in excluded_files:
                filenames_and_subjects_train = filenames_and_subjects_train[filenames_and_subjects_train['image_filename'] != file]
 
            # Train the diffusion model on the fold's training set.
            for file in tqdm(filenames_and_subjects_train['image_filename'].unique()):
                if '_img.png' not in file:
                    mask_filename = file.replace('image.png', 'mask.png')
                else:
                    mask_filename = file.replace('_img.png', '_mask.png')

                old_filepath = os.path.join(stomach_data_and_masks_dir_path, mask_filename)
                new_filepath = os.path.join(masks_path, file)
                if not os.path.exists(new_filepath):
                    shutil.copy(old_filepath, new_filepath)

                old_filepath = os.path.join(stomach_data_and_masks_dir_path, file)
                new_filepath = os.path.join(masked_images_path, file)
                image = io.read_image(old_filepath, mode=io.ImageReadMode.RGB).to(torch.uint8)
                if not os.path.exists(new_filepath):
                    io.write_png(image, new_filepath)

            if not skip_validation:
                num_images = 0
                for file in tqdm(filenames_and_subjects_val_subject_train['image_filename'].unique()):
                    is_synthetic = file.endswith('_img.png')
                    if is_synthetic:
                        continue # We will only perform validation with non-synthetic images.
                    mask_filename = file.replace('image.png', 'mask.png')

                    old_filepath = os.path.join(stomach_data_and_masks_dir_path, mask_filename)
                    new_filepath = os.path.join(masks_val_path, file)
                    if not os.path.exists(new_filepath):
                        shutil.copy(old_filepath, new_filepath)
                    old_filepath = os.path.join(stomach_data_and_masks_dir_path, file)
                    new_filepath = os.path.join(masked_images_val_path, file)
                    image = io.read_image(old_filepath, mode=io.ImageReadMode.RGB).to(torch.uint8)
                    if not os.path.exists(new_filepath):
                        io.write_png(image, new_filepath)
                        num_images += 1
                print('Number of images selected for validation',num_images)

            # Generate bounding boxes.
            for curr_set in ['train', 'val']:
                bounding_boxes_json = {}

                lengths = []
                excluded_files_set = set(excluded_files)

                curr_masks_path = masks_path if curr_set == 'train' else masks_val_path

                for mask_file in tqdm(os.listdir(curr_masks_path)):
                    if mask_file in excluded_files_set:
                        print(f"Skipping excluded file: {mask_file}")
                        continue
                    mask_path = os.path.join(curr_masks_path, mask_file)
                    mask = io.read_image(mask_path, mode=io.ImageReadMode.UNCHANGED)

                    try:
                        bboxes = get_bbox_from_mask(mask)
                    except:
                        bboxes = []

                    formatted_bboxes = []

                    for bbox in bboxes:
                        x_min, y_min, x_max, y_max = bbox.tolist()
                        formatted_bboxes.append({
                            'label': 'stomach',
                            'x_min': int(x_min),
                            'y_min': int(y_min),
                            'x_max': int(x_max),
                            'y_max': int(y_max)
                        })

                    bounding_boxes_json[mask_file] = {
                        'height': mask.shape[1],
                        'width': mask.shape[2],
                        'bbox': formatted_bboxes
                    }
                    lengths.append(len(formatted_bboxes))

                bounding_boxes_path = f'../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/guided_diffusion/segmented-images-roberta/{cv_val_subject}/bounding-boxes{'-val' if curr_set == 'val' else ''}.json'

                with open(bounding_boxes_path, 'w') as f:
                    json.dump(bounding_boxes_json, f, indent=4)


            image_train_py = '../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/guided_diffusion/image_train.py'  

            python_command = f"""
        python '{image_train_py}' --cv_subject '{cv_val_subject}' --data_dir '{base_path}' --image_size 256 --out_dir checkpoints-cv-val-subj-{cv_val_subject} --batch_size 1 --gpu_id {gpu_id} --skip_validation {skip_validation} {f"--validation_dir '{base_path_val}'" if not skip_validation else ""} --singan_augmented_dataset {'1' if singan_augmented_dataset else '0'}
            """

            try:
                print(f"Running command: {python_command}")
                result = subprocess.run(python_command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while processing: {e}")

    cv_subjects = custom_cv_subjects if custom_cv_subjects else filenames_and_subjects['subject'].unique()
    for cv_val_subject in cv_subjects:
        # Synthesize the samples
        checkpoint_folder_name = f'checkpoints-cv-val-subj-{cv_val_subject}'

        # Check if the folder exists
        if not os.path.exists(checkpoint_folder_name):
            print(f"Checkpoint folder {checkpoint_folder_name} does not exist. Skipping synthesis for subject {cv_val_subject}.")
            continue

        diffuse_gen_folder = max(
            (entry for entry in os.scandir(checkpoint_folder_name) if entry.is_dir()),
            key=lambda e: e.stat().st_mtime
        ).name


        folder = f'{checkpoint_folder_name}/{diffuse_gen_folder}'

        latest_model_file = max(
            (entry for entry in os.scandir(folder) if entry.is_file() and 'model' in entry.name and entry.name.endswith('.pt')),
            key=lambda e: e.stat().st_ctime  # Use st_mtime if you want "last modified" instead
        ).name

        model_path = f'{checkpoint_folder_name}/{diffuse_gen_folder}/{latest_model_file}'

        output_path = f'../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_{cv_val_subject}'
        os.makedirs(output_path, exist_ok=True)
        main_py = '../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/main.py'

        data_dir = f'../../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/guided_diffusion/segmented-images-roberta{"-hq-masks" if hq_samples_only else ""}/{cv_val_subject}/masked-images'

        synthesize_samples_py = f"""
    python '{main_py}' \
    --data_dir '{data_dir}' \
    --output_path '{output_path}' \
    --model_path '{model_path}' \
    --cluster_model_dir 'clustering' \
    --diff_iter 100 \
    --timestep_respacing 200 \
    --skip_timesteps 80 \
    --model_output_size 256 \
    --num_samples {num_samples} \
    --batch_size 1 \
    --use_noise_aug_all \
    --use_colormatch \
    --gpu_id {gpu_id} \
    -fti -inp -spi -sty
        """

        try:
            print(f"Running command: {synthesize_samples_py}")
            result = subprocess.run(synthesize_samples_py, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Run diffusion synthesis performance benchmark"
    )
    parser.add_argument(
        '--custom-cv-subjects', '-c',
        type=str,
        default="FD-027,FD-029,FD-030,FD-031,FD-032",
        help="Comma-separated list of CV subject identifiers (e.g. ‘FD-001,FD-002’)."
    )
    parser.add_argument(
        '--skip-training', '-s',
        action='store_true',
        help="If set, skips the training phase."
    )
    parser.add_argument(
        '--datestamp', '-d',
        type=str,
        default=None,
        help="Use a fixed datestamp (format YYYY-MM-DD_HH-MM-SS)."
    )
    parser.add_argument(
        '--gpu-id', '-g',
        type=int,
        default=1,
        help="GPU device ID to use."
    )
    parser.add_argument(
        '--hq-samples-only', '-hq',
        action='store_true',
        help="If set, only uses high-quality samples for synthesis."
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=50,
        help="Number of samples to synthesize per image."
    )
    parser.add_argument(
        '--singan-augmented-dataset', '-sa',
        action='store_true',
        help="If set, uses the SiGAN augmented dataset."
    )
    parser.add_argument(
        '--skip-validation', '-sv ',
        action='store_true',
        help="If set, skips the validation phase."
    )

    args = parser.parse_args()

    exp_datestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')

    synthesis_performance_benchmark_diffusion(custom_cv_subjects=args.custom_cv_subjects.split(','), skip_training=args.skip_training, datestamp=exp_datestamp, gpu_id=args.gpu_id, hq_samples_only=args.hq_samples_only, num_samples=args.num_samples, singan_augmented_dataset=args.singan_augmented_dataset, skip_validation=args.skip_validation)

if __name__ == "__main__":
    main()