import os
import cv2
import sys
import glob
import json
import torch
import random
import shutil
import pickle
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import io
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from calculate_similarity import process_folders, evaluate_folder

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    ),
)

from discriminator import CNNClassifier, train_discriminator

full_dataset_folder = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg/Input/data-RGBA'
trained_models_folder = '/home/miguel/GI/1 - Segmentation/Synthesis-performance-n-samples/TrainedModels'

main_train_py = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg/main_train.py'
random_samples_py = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg/random_samples.py'

random_samples_folder = '/home/miguel/GI/1 - Segmentation/Synthesis-performance-n-samples/Output/RandomSamples'
singan_output_from_similarity_folder = '/home/miguel/GI/1 - Segmentation/Synthesis-performance-n-samples/singan_output_from_similarity'
all_eval_folder = '/home/miguel/GI/1 - Segmentation/Synthesis-performance-n-samples/all_eval_results'

real_training_dataset_folder = '/home/miguel/GI/1 - Segmentation/Synthesis-performance-n-samples/real_training_dataset'

num_samples_to_generate = 10

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
    datestamp: str = None,
    skip_training: bool = True,
    gpu_id: int = 0
):
    stomach_data_and_masks_dir_path = '/home/miguel/GI/0 - Data Exploration & Analysis/GI-Roberta/gi-roberta-dataset/full_dataset'
    
    available_filenames = [filename for filename in os.listdir(stomach_data_and_masks_dir_path) if 'mask' not in filename]

    # Associate the subjects of the filenames
    filenames_and_subjects = pd.DataFrame({
        'image_filename': available_filenames,
        'subject': ["-".join(filename.split("-")[:2]) for filename in available_filenames]
    })

    if not skip_training:
        for cv_val_subject in filenames_and_subjects['subject'].unique():
            filenames_and_subjects_train = filenames_and_subjects[filenames_and_subjects['subject'] != cv_val_subject]

            base_path = f'/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/guided_diffusion/segmented-images-roberta/{cv_val_subject}'

            # Create the base path if it doesn't exist
            os.makedirs(base_path, exist_ok=True)

            masked_images_path = f'{base_path}/masked-images'
            masks_path = f'{base_path}/masks'

            if os.path.exists(masked_images_path):
                shutil.rmtree(masked_images_path)

            if os.path.exists(masks_path):
                shutil.rmtree(masks_path)

            os.makedirs(masked_images_path, exist_ok=True)
            os.makedirs(masks_path, exist_ok=True)

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
                mask_filename = file.replace('image.png', 'mask.png')
                    
                old_filepath = os.path.join(stomach_data_and_masks_dir_path, mask_filename)
                new_filepath = os.path.join(masks_path, file)
                if not os.path.exists(new_filepath):
                    shutil.copy(old_filepath, new_filepath)
                old_filepath = os.path.join(stomach_data_and_masks_dir_path, file)
                new_filepath = os.path.join(masked_images_path, file)
                image = io.read_image(old_filepath, mode=io.ImageReadMode.RGB).to(torch.uint8)
                if not os.path.exists(new_filepath):
                    io.write_png(image, new_filepath)
        
            # Generate bounding boxes.
            bounding_boxes_json = {}

            lengths = []
            excluded_files_set = set(excluded_files)
            for mask_file in tqdm(os.listdir(masks_path)):
                if mask_file in excluded_files_set:
                    print(f"Skipping excluded file: {mask_file}")
                    continue
                mask_path = os.path.join(masks_path, mask_file)
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

            bounding_boxes_path = f'/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/guided_diffusion/segmented-images-roberta/{cv_val_subject}/bounding-boxes.json'

            with open(bounding_boxes_path, 'w') as f:
                json.dump(bounding_boxes_json, f, indent=4)
                image_train_py = '/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/guided_diffusion/image_train.py'  
                python_command = f"""
            python '{image_train_py}' --data_dir '{base_path}' --image_size 256 --out_dir checkpoints-cv-val-subj-{cv_val_subject} --batch_size 1
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

        print(folder)

        latest_model_file = max(
            (entry for entry in os.scandir(folder) if entry.is_file() and 'model' in entry.name and entry.name.endswith('.pt')),
            key=lambda e: e.stat().st_ctime  # Use st_mtime if you want "last modified" instead
        ).name

        model_path = f'{checkpoint_folder_name}/{diffuse_gen_folder}/{latest_model_file}'

        output_path = f'/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_{cv_val_subject}'
        os.makedirs(output_path, exist_ok=True)
        main_py = '/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/main.py'

        synthesize_samples_py = f"""
    python '{main_py}' \
    --data_dir '/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/guided_diffusion/segmented-images-roberta/{cv_val_subject}/masked-images' \
    --output_path '{output_path}' \
    --model_path '{model_path}' \
    --cluster_model_dir 'clustering' \
    --diff_iter 100 \
    --timestep_respacing 200 \
    --skip_timesteps 80 \
    --model_output_size 256 \
    --num_samples 1 \
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
    return
    # Evaluate the SSIM of the generated samples
    original_images_folder_path = '/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/guided_diffusion/segmented-images-roberta/masked-images'

    filename_images = {}
    original_sample_filenames = os.listdir(original_images_folder_path)

    for filename in original_sample_filenames:
        if filename.endswith('.png'):
            img_path = os.path.join(original_images_folder_path, filename)
            image = Image.open(img_path)
            filename_images[filename] = image
        else:
            continue


    base = '/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples'

    latest_dir = max(
        (entry for entry in os.scandir(base) if entry.is_dir()),
        key=lambda e: e.stat().st_mtime      # use st_ctime on macOS / Windows if you want “created” time
    ).path

    img_path = max(
        (entry for entry in os.scandir(latest_dir) if entry.is_file() and entry.name.endswith('.pkl') and 'styled' not in entry.name),
        key=lambda e: e.stat().st_mtime
    ).path

    styled_img_path = img_path.replace('samples_', 'styled_samples_')

    with open(img_path, "rb") as f:
        images = pickle.load(f)
    with open(styled_img_path, "rb") as f:
        styled_images = pickle.load(f)

    total_ssim = 0

    for key in styled_images.keys():
        styled_image = styled_images[key]
        styled_image = styled_image[0]

        # Split image and mask
        image = styled_image[:, :, :3]  # RGB image (values 0-255)
        mask = styled_image[:, :, 3]    # Single-channel mask (values 0-255)

        # Evaluate SSIM compared to its original base image
        filename = key.split('/')[-1]

        original_image = filename_images[filename]
        original_image = np.array(original_image)
        original_image = cv2.resize(original_image, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        original_image = original_image[:, :, :3]
        original_image = original_image.astype(np.uint8)

        # Convert images to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

        # Calculate SSIM
        image_gray = image_gray.astype(np.uint8)
        original_image_gray = original_image_gray.astype(np.uint8)

        ssim_value = ssim(image_gray, original_image_gray)
        mse_value = np.mean((image_gray - original_image_gray) ** 2)

        total_ssim = total_ssim + ssim_value

    mean_ssim = total_ssim / len(styled_images)

    if datestamp:
        performance_df_filename = f'synthesis_performance_benchmark_diffusion_{sample_size_name}_{datestamp}.csv'
        if os.path.exists(performance_df_filename):
            performance_df = pd.read_csv(performance_df_filename)
            performance_df = performance_df.append({
                'sample_size': sample_size,
                'mean_ssim': mean_ssim,
                'num_samples': num_samples_to_generate
            }, ignore_index=True)
        else:
            performance_df = pd.DataFrame({
                'sample_size': [sample_size],
                'mean_ssim': [mean_ssim],
                'num_samples': [num_samples_to_generate]
            })
        performance_df.to_csv(performance_df_filename, index=False)

def synthesis_performance_benchmark_trained(
    sample_size: int
):
    case_slice_images = os.listdir('TrainedModels')
    for model in case_slice_images:
        if not os.path.exists(os.path.join('TrainedModels', model, 'scale_factor=0.750000,alpha=10', 'Gs.pth')):
            case_slice_images.remove(model)
    
    # Randomly select sample_size images
    sampled_images = random.sample(case_slice_images, sample_size)

    synthetic_output_folder = f"{random_samples_folder}_sample_size_{sample_size}".replace('.', '_')

    num_successes = 0

    for case_slice_image_filename_random in tqdm(sampled_images):
        case_slice_image_filename_random = case_slice_image_filename_random + '.png'
        # Copy the selected image to the real training dataset folder
        shutil.copy(
            os.path.join(full_dataset_folder, case_slice_image_filename_random),
            os.path.join(real_training_dataset_folder, case_slice_image_filename_random)
        )


        model_folder = os.path.join(trained_models_folder, case_slice_image_filename_random[:-4])

        if os.path.exists(model_folder) and os.path.exists(os.path.join(model_folder, 'scale_factor=0.750000,alpha=10', 'Gs.pth')):
            python_command = f"python '{random_samples_py}' --input_name '{case_slice_image_filename_random}' --input_dir='{full_dataset_folder}' --mode random_samples --gen_start_scale 0 --nc_z 4 --nc_im 4 --gpu_id 0 --num_samples {num_samples_to_generate} --out '{synthetic_output_folder}'"
        else:
            python_command = f"python '{main_train_py}' --input_name '{case_slice_image_filename_random}'  --input_dir='{full_dataset_folder}' --nc_z 4 --nc_im 4 --gpu_id 0 --num_samples {num_samples_to_generate} --out '{synthetic_output_folder}'"

        try:
            print(f"Running command: {python_command}")
            result = subprocess.run(python_command, shell=True, check=True)
            num_successes += 1
            print(f"Command completed successfully for {case_slice_image_filename_random}\n")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {case_slice_image_filename_random}: {e}")
            continue  # Continue processing the next image if an error occurs

    
    print(f"Successfully processed {num_successes} out of {len(sampled_images)} images.")

    real_folder = full_dataset_folder
    output_folder = singan_output_from_similarity_folder

    synthetic_folder = os.path.join(singan_output_from_similarity_folder, 'synthetic_images')
    masks_folder = os.path.join(singan_output_from_similarity_folder, 'masks')

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    if os.path.exists(synthetic_folder):
        shutil.rmtree(synthetic_folder)
    if os.path.exists(masks_folder):
        shutil.rmtree(masks_folder)

    os.makedirs(synthetic_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)

    image_files = glob.glob(os.path.join(synthetic_output_folder, '**', '*_img.png'), recursive=True)
    mask_files = glob.glob(os.path.join(synthetic_output_folder, '**', '*_mask.png'), recursive=True)
    for img_file in image_files:
        case_name = img_file.split('/')[-3]
        img_name = img_file.split('/')[-1][:-8]  # Remove '_img.png' suffix
        new_name = f"{case_name}_{img_name}.png"
        
        shutil.copy(img_file, f'{synthetic_folder}/{new_name}')

    for mask_file in mask_files:
        case_name = mask_file.split('/')[-3]
        mask_name = mask_file.split('/')[-1][:-9]  # Remove '_mask.png' suffix
        new_name = f"{case_name}_{mask_name}.png"
        shutil.copy(mask_file, f'{masks_folder}/{new_name}')

    process_folders(real_folder, synthetic_folder, masks_folder, output_folder)
    
    evaluation_results = evaluate_folder(real_folder, output_folder)

    evaluation_results.to_csv(f'{all_eval_folder}/evaluation_results_sample_size_{sample_size}.csv', index=False)

    # TODO: Adjust batch sizes

    discriminator_dataset = RealAndSyntheticImageDataset(
        sample_size, synthetic_folder, real_training_dataset_folder
    )

    discriminator_train_loader = DataLoader(discriminator_dataset, batch_size=min(max(1, sample_size // 10), 10), shuffle=True)

    discriminator_model = CNNClassifier()

    discriminator_training_results, discriminator_cv_val_results, discriminator_testing_results =  train_discriminator(
        discriminator_model,
        discriminator_dataset,
        None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_epochs=100,
        lr=0.001,
    )

    pd.DataFrame(discriminator_training_results).to_csv(f"discriminator_training_results/discriminator_performance_{sample_size}_samples.csv")
    pd.DataFrame(discriminator_cv_val_results).to_csv(f"discriminator_training_results/discriminator_cv_val_performance_{sample_size}_samples.csv")
    pd.DataFrame(discriminator_testing_results).to_csv(f"discriminator_training_results/discriminator_testing_performance_{sample_size}_samples.csv")


def main():
    exp_datestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
    synthesis_performance_benchmark_diffusion(custom_cv_subjects=[], skip_training = True, datestamp = exp_datestamp, gpu_id = 1)
    
if __name__ == "__main__":
    main()

