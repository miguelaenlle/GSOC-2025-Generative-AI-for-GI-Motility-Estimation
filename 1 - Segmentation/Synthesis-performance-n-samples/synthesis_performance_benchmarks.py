import os
import cv2
import sys
import glob
import json
import torch
import random
import shutil
import pickle
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



def synthesis_performance_benchmark(
    sample_size: int = 10,
    use_trained_singans: bool = False
):
    synthetic_output_folder = f"{random_samples_folder}_sample_size_{sample_size}".replace('.', '_')

    # Acquire sample_size synthetic samples.
    high_quality_subjects = pd.read_csv('high_quality_subjects_train.csv', index_col = 0)
    if sample_size > len(high_quality_subjects):
        raise ValueError(f"Sample size {sample_size} exceeds available subjects {len(high_quality_subjects)}.")
    
    num_filenames = []
    high_quality_subjects['num_filenames'] = 0
    for i in high_quality_subjects.index:
        case_id = high_quality_subjects.loc[i, 'case_id']
        case_slice_image_filenames = glob.glob(
            os.path.join(full_dataset_folder, f'{case_id}*.png')
        )
        num_filenames.append(len(case_slice_image_filenames))

    
    high_quality_subjects['num_filenames'] = num_filenames
    high_quality_subjects = high_quality_subjects[high_quality_subjects['num_filenames'] > 0]

    if use_trained_singans:
        case_ids = os.listdir('TrainedModels')
        case_days = np.unique([case_id.split('_slice')[0] for case_id in case_ids])

        high_quality_subjects['trained_model_available'] = False
        for i in high_quality_subjects.index:
            case_id = high_quality_subjects.loc[i, 'case_id']
            if case_id in case_days:
                if os.path.exists(os.path.join('TrainedModels', case_id, 'scale_factor=0.750000,alpha=10', 'Gs.pth')):
                    high_quality_subjects.loc[i, 'trained_model_available'] = True

        high_quality_subjects = high_quality_subjects[high_quality_subjects['trained_model_available']]

        

    sampled_subjects = high_quality_subjects.sample(n=sample_size, random_state=42)


    print(f"Sampled {sample_size} subjects for synthesis performance benchmarking.")
    num_successes = 0
    # Select reference images for each sample. These will train the SinGANs.


    if os.path.exists(real_training_dataset_folder):
        shutil.rmtree(real_training_dataset_folder)
    os.makedirs(real_training_dataset_folder, exist_ok=True)


    case_ids = []
    case_ids = sampled_subjects['case_id'].unique()

    for case_id in tqdm(case_ids):
        print(case_id)
        try:
            case_slice_image_filenames = glob.glob(
                os.path.join(full_dataset_folder, f'{case_id}*.png')
            )
            case_slice_image_filename_random = random.choice(case_slice_image_filenames)
            case_slice_image_filename_random = case_slice_image_filename_random[case_slice_image_filename_random.rfind('/')+1:]


            # Copy the selected image to the real training dataset folder
            shutil.copy(
                os.path.join(full_dataset_folder, case_slice_image_filename_random),
                os.path.join(real_training_dataset_folder, case_slice_image_filename_random)
            )

            # OVERRIDE TEST
            # case_slice_image_filename_random = 'case18_day0_slice_0068.png'

            model_folder = os.path.join(trained_models_folder, case_slice_image_filename_random[:-4])

            if os.path.exists(model_folder) and os.path.exists(os.path.join(model_folder, 'scale_factor=0.750000,alpha=10', 'Gs.pth')):
                python_command = f"python '{random_samples_py}' --input_name '{case_slice_image_filename_random}' --input_dir='{full_dataset_folder}' --mode random_samples --gen_start_scale 0 --nc_z 4 --nc_im 4 --gpu_id 0 --num_samples {num_samples_to_generate} --out '{synthetic_output_folder}'"
            else:
                python_command = f"python '{main_train_py}' --input_name '{case_slice_image_filename_random}'  --input_dir='{full_dataset_folder}' --nc_z 4 --nc_im 4 --gpu_id 0 --num_samples {num_samples_to_generate} --out '{synthetic_output_folder}'"

            try:
                print(f"Running command: {python_command}")
                result = subprocess.run(python_command, shell=True, check=True)
                num_successes += 1
                print(f"Command completed successfully for {case_id}\n")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while processing {case_id}: {e}")
                continue  # Continue processing the next image if an error occurs
        except:
            continue
    print(f"Successfully processed {num_successes} out of {len(sampled_subjects)} subjects.")

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

    breakpoint()

    discriminator_dataset = RealAndSyntheticImageDataset(
        sample_size, synthetic_folder, real_training_dataset_folder
    )
    discriminator_dataset = discriminator_dataset.to('cuda' if torch.cuda.is_available() else 'cpu')

    discriminator_model = CNNClassifier()
    discriminator_model = discriminator_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    discriminator_training_results, discriminator_cv_val_results, discriminator_testing_results = train_discriminator(
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
    datestamp: str = None,
    sample_size = 10,
    skip_training: bool = True
):
    stomach_data_and_masks_dir_path = '/home/miguel/GI/0 - Data Exploration & Analysis/UW-Madison/stomach_data_and_masks_preparation/stomach_data_and_masks'
    
    available_filenames = [filename for filename in os.listdir(stomach_data_and_masks_dir_path) if 'stomach_mask' not in filename]

    sample_size_name = sample_size if sample_size else 'all'

    # Locate available sampled_images files
    sampled_image_files = [file for file in os.listdir() if file.startswith(f'sampled_images_{sample_size_name}') and file.endswith('.csv')]
    sampled_images = []
    if sampled_image_files:
        sampled_image_file = sampled_image_files[0]
        print('Sampled image file found:', sampled_image_file)
        sampled_images_df = pd.read_csv(sampled_image_file)
        sampled_images = sampled_images_df['image_filename'].tolist()[:100]
    else:
        # Randomly select sample_size images
        if sample_size:
            sampled_images = random.sample(available_filenames, sample_size)
        else:
            sampled_images = random.sample(available_filenames, 500)

        datestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Export the sampled images
        pd.DataFrame(sampled_images, columns=['image_filename']).to_csv(
            f'sampled_images_{sample_size}_{datestamp}.csv', index=False
        )

    base_path = '/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/guided_diffusion/segmented-images-madison'

    masked_images_path = f'{base_path}/masked-images'
    masks_path = f'{base_path}/masks'

    if os.path.exists(masked_images_path):
        shutil.rmtree(masked_images_path)

    if os.path.exists(masks_path):
        shutil.rmtree(masks_path)

    os.makedirs(masked_images_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)

    # Train the diffusion model on sample_size samples.
    for file in sampled_images:
        mask_filename = file.replace('.png', '_stomach_mask.png')
            
        old_filepath = os.path.join(stomach_data_and_masks_dir_path, mask_filename)
        new_filepath = os.path.join(masks_path, file)
        if not os.path.exists(new_filepath):
            shutil.copy(old_filepath, new_filepath)
            print(f"Copied {mask_filename} to {file}")
        else:
            print(f"File {file} already exists, skipping copy.")

        old_filepath = os.path.join(stomach_data_and_masks_dir_path, file)
        new_filepath = os.path.join(masked_images_path, file)
        image = io.read_image(old_filepath, mode=io.ImageReadMode.RGB).to(torch.uint8)
        if not os.path.exists(new_filepath):
            io.write_png(image, new_filepath)
            print(f"Copied {file} to {new_filepath}")
        else:
            print(f"File {new_filepath} already exists, skipping copy.")
    
    # Generate bounding boxes.
    bounding_boxes_json = {}

    lengths = []

    breakpoint()
    for mask_file in tqdm(os.listdir(masks_path)):
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


        if (len(formatted_bboxes) > 0):
            print(f"Found {len(formatted_bboxes)} bounding boxes in {mask_file}")
            breakpoint()
        lengths.append(len(formatted_bboxes))

    bounding_boxes_path = '/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/guided_diffusion/segmented-images-madison/bounding-boxes.json'

    with open(bounding_boxes_path, 'w') as f:
        json.dump(bounding_boxes_json, f, indent=4)


    if not skip_training:
        image_train_py = '/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/guided_diffusion/image_train.py'  
        python_command = f"""
    python '{image_train_py}' --data_dir '{base_path}' --image_size 256 --out_dir checkpoints-{sample_size_name} --batch_size 1
        """

        try:
            print(f"Running command: {python_command}")
            result = subprocess.run(python_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing: {e}")

    # Synthesize the samples
    if sample_size:
        checkpoint_folder_name = f'checkpoints-{sample_size_name}'
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
    else:
        model_path = '/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/checkpoints/diffgen-2025-06-11-23-11/model310000.pt'

    output_path = f'/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/image_samples_{sample_size_name}'
    os.makedirs(output_path, exist_ok=True)
    main_py = '/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/main.py'

    synthesize_samples_py = f"""
python '{main_py}' \
--data_dir '/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/guided_diffusion/segmented-images-madison/masked-images' \
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
-fti -sty -inp -spi
    """

    try:
        print(f"Running command: {synthesize_samples_py}")
        result = subprocess.run(synthesize_samples_py, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while processing: {e}")
    
    # Evaluate the SSIM of the generated samples
    return
    original_images_folder_path = '/home/miguel/GI/1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/guided_diffusion/segmented-images-madison/masked-images'

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
    # sample_sizes = [1, 10, 25, 50, 100]
    sample_sizes = [
        # None,
        110, 
        300,
        200,
        # 500,
        # 400,


        # 10,
        # 100,
        # 50,
        # 30,
        # 70,
        # 60,
        # 20,
        # 80,
        # 90
        # 2,
        # 15,
        # 30,
        # 40,
        # 50,
        # 60,
        # 70,
        # 80,
        # 90,
        # 100
    ]

    exp_datestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')

    for sample_size in sample_sizes:
        # synthesis_performance_benchmark(sample_size=sample_size, use_trained_singans=True)
        # synthesis_performance_benchmark_trained(sample_size=sample_size)
        synthesis_performance_benchmark_diffusion(sample_size=sample_size, skip_training = True, datestamp = exp_datestamp)
    
if __name__ == "__main__":
    main()

