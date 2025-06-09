import os
import cv2
import sys
import glob
import torch
import random
import shutil
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
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

    discriminator_dataset = RealAndSyntheticImageDataset(
        sample_size, synthetic_folder, real_training_dataset_folder
    )

    discriminator_train_loader = DataLoader(discriminator_dataset, batch_size=min(max(1, sample_size // 10), 10), shuffle=True)

    discriminator_model = CNNClassifier()

    discriminator_training_results = train_discriminator(
        discriminator_model,
        discriminator_train_loader,
        None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_epochs=100,
        lr=0.001,
    )

    discriminator_training_results.to_csv(f"discriminator_training_results/discriminator_performance_{sample_size}_samples.csv")


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

    discriminator_training_results = train_discriminator(
        discriminator_model,
        discriminator_train_loader,
        None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_epochs=100,
        lr=0.001,
    )

    discriminator_training_results.to_csv(f"discriminator_training_results/discriminator_performance_{sample_size}_samples.csv")


def main():
    # sample_sizes = [1, 10, 25, 50, 100]
    sample_sizes = [
        2,
        15,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100
    ]

    for sample_size in sample_sizes:
        # synthesis_performance_benchmark(sample_size=sample_size, use_trained_singans=True)
        synthesis_performance_benchmark_trained(sample_size=sample_size)
    
if __name__ == "__main__":
    main()

