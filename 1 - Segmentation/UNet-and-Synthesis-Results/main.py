import os
import shutil

import random
import torch
import wandb
import subprocess
import pandas as pd
from tqdm import tqdm

import os
from PIL import Image
import numpy as np
import glob

from calculate_similarity import process_folders, evaluate_folder

import torchvision.io as io
import torchvision.transforms.functional as TF

# full_dataset_folder = '/home/miguel/GI/0 - Data Exploration & Analysis/UW-Madison/stomach_data_and_masks_preparation/stomach_data_and_masks'

SKIP_SINGAN_TRAINING = True
SKIP_UNTRAINED_MODELS = True
SHORT_SINGAN_TRAINING = True

full_dataset_folder = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg/Input/data-RGBA'
full_dataset_folder_image_and_masks = '/home/miguel/GI/0 - Data Exploration & Analysis/UW-Madison/stomach_data_and_masks_preparation/stomach_data_and_masks'

train_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/train'

unet_and_synthesis_results_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results'
train_augmented_singan_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/train_augmented_singan'
test_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/test'
singan_similarity_output_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/singan_output_from_similarity'

split_singan_output_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/split_singan_output'
random_samples_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/Output/RandomSamples'

main_train_py = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg/main_train.py'
random_samples_py = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg/random_samples.py'
train_unet_py = '/home/miguel/GI/1 - Segmentation/Run-UNET/scripts/train_unet.py'

trained_models_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/TrainedModels'

unet_training_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/unet_training'
unet_validation_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/unet_validation'


train_test_split_ratio = 0.8  # 80% for training, 20% for testing

def prepare_dataset():
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(train_augmented_singan_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    os.makedirs(unet_training_folder, exist_ok=True)
    os.makedirs(unet_validation_folder, exist_ok=True)

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
            shutil.copy(os.path.join(full_dataset_folder, file), os.path.join(train_folder, file))
            base_name = file[:-4]  # Remove the '.png' extension

            shutil.copy(
                os.path.join(full_dataset_folder_image_and_masks, file), 
                os.path.join(unet_training_folder, base_name + '_image.png')
            )
            shutil.copy(
                os.path.join(full_dataset_folder_image_and_masks, file), 
                os.path.join(unet_training_folder, base_name + '_stomach_mask.png')
            )

    # Copy the testing files
    for case in tqdm(test_cases):
        case_files = [f for f in files if f.startswith(case)]
        for file in case_files:
            shutil.copy(os.path.join(full_dataset_folder, file), os.path.join(test_folder, file))
            base_name = file[:-4]  # Remove the '.png' extension

            shutil.copy(
                os.path.join(full_dataset_folder_image_and_masks, file), 
                os.path.join(unet_validation_folder, base_name + '_image.png')
            )
            shutil.copy(
                os.path.join(full_dataset_folder_image_and_masks, file), 
                os.path.join(unet_validation_folder, base_name + '_stomach_mask.png')
            )

def process_to_rgba(input_dir, output_dir):
    """
    Converts pairs of images and masks in an input directory to RGBA format and saves them to an output directory.
    
    Parameters:
        - input_dir (str): Path to the directory containing image and mask pairs in .png format in "Preprocess".
        - output_dir (str): Path to the directory where RGBA images will be saved in "Input".
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    
    # Retrieve all .png files and sort them to ensure image-mask pairs are processed together
    paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    
    # Loop over image and mask pairs, assuming each image is followed by its mask
    for i in tqdm(range(0, len(paths), 2)):
        image_path = paths[i]       # Path to the MRI image
        mask_path = paths[i + 1]    # Path to the corresponding mask
        
        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Check dimensions of the image and the mask match
        if image.size != mask.size:
            raise ValueError(f"Image and mask sizes do not match for {image_path} and {mask_path}: {image.size} vs {mask.size}")
        
        # Convert images to numpy arrays
        image_array = np.array(image)
        mask_array = np.array(mask)
        
        # Stack RGB image and grayscale mask to create an RGBA image
        rgba_image = np.dstack((image_array, mask_array))
        
        # Convert back to PIL RGBA image
        rgba_image_pil = Image.fromarray(rgba_image, 'RGBA')
        
        # Save the RGBA image
        output_filename = os.path.basename(image_path).replace('_image.png', '_RGBA.png')
        output_path = os.path.join(output_dir, output_filename)
        rgba_image_pil.save(output_path)
        
        print(f"Saved 4D image to {output_path}")


def move_singan_results_to_preprocessing(
    source_dir,
    dest_dir,
    high_quality_images
):
    for subdir in os.listdir(source_dir):
        if subdir + ".png" not in high_quality_images:
            continue
        subdir_path = os.path.join(source_dir, subdir)
        if os.path.isdir(subdir_path):
            print(f"Processing subdirectory: {subdir_path}")
            gen_scale_dir = os.path.join(subdir_path, 'gen_start_scale=0')  # img/mask data stored here
            if os.path.isdir(gen_scale_dir):
                print(f"  Found 'gen_start_scale=0': {gen_scale_dir}")
                files = os.listdir(gen_scale_dir)
                if files:
                    print(f"  Files in directory: {files}")
                else:
                    print(f"  No files found in {gen_scale_dir}")
                for filename in files:
                    file_path = os.path.join(gen_scale_dir, filename)
                    if os.path.isfile(file_path):
                        print(f"  Processing file: {file_path}")
                        if 'mask' in filename.lower():
                            new_filename = os.path.splitext(filename)[0] + '_mask.png'
                        else:
                            new_filename = os.path.splitext(filename)[0] + '_image.png'

                        dest_file_path = os.path.join(dest_dir, new_filename)

                        counter = 1
                        original_new_filename = new_filename
                        while os.path.exists(dest_file_path):
                            base, ext = os.path.splitext(original_new_filename)
                            new_filename = f"{base}_{counter}{ext}"
                            dest_file_path = os.path.join(dest_dir, new_filename)
                            counter += 1

                        try:
                            with Image.open(file_path) as img:
                                grayscale_img = img.convert('L')
                                grayscale_img.save(dest_file_path)
                                print(f"Converted, moved, and renamed: {file_path} -> {dest_file_path}")
                        except Exception as e:
                            print(f"Error processing file {file_path}: {e}")
    
def synthesize_with_singan(
    output_train_folder,
    synthetic_output_folder,
    ratio_synthesized_to_real=1,
    skip_untrained_models = False
):
    train_dataset = os.listdir(train_folder)
    # Specifically picked to ensure that training the SinGANS takes about 24 hours
    num_base_images = 200
    if len(train_dataset) < num_base_images:
        print(f"Not enough images in the training set for Singan-Seg synthesis. Need at least {num_base_images} images.")
        return
   
    if not SKIP_SINGAN_TRAINING:
        selected_file = os.path.join(train_folder, 'selected_images.csv')
        if os.path.exists(selected_file):
            selected_image_df = pd.read_csv(selected_file)
            selected_images = selected_image_df['0'].unique().tolist()
        else:
            selected_images = random.sample(train_dataset, num_base_images)
            pd.DataFrame(selected_images).to_csv(os.path.join(train_folder, 'selected_images.csv'), index=False)
        # Perform set union between the selected images and existing available base image models
        existing_models = set(np.array(os.listdir(trained_models_folder)) + '.png')
        selected_images = [str(text) for text in (list(set(selected_images) | existing_models))]

        num_successes = 0

        for img_name in tqdm(selected_images):
            if SHORT_SINGAN_TRAINING and num_successes >= 5:
                print("Short Singan training mode enabled, stopping after 5 successful syntheses.")
                break
            num_samples_to_generate = int((len(train_dataset) * ratio_synthesized_to_real * 4) // num_base_images)
            if os.path.exists(os.path.join(trained_models_folder, img_name[:-4])):
                python_command = f"python '{random_samples_py}' --input_name {img_name} --input_dir='{train_folder}' --mode random_samples --gen_start_scale 0 --nc_z 4 --nc_im 4 --gpu_id 0 --num_samples {num_samples_to_generate} --out '{synthetic_output_folder}'"
            elif skip_untrained_models == False:
                python_command = f"python '{main_train_py}' --input_name {img_name}  --input_dir='{train_folder}' --nc_z 4 --nc_im 4 --gpu_id 0 --num_samples {num_samples_to_generate} --out '{synthetic_output_folder}'"
            else:
                print(f"Skipping {img_name} as it is not trained yet.")
                continue
            try:
                print(f"Running command: {python_command}")
                result = subprocess.run(python_command, shell=True, check=True)
                num_successes += 1
                print(f"Command completed successfully for {img_name}\n")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while processing {img_name}: {e}")
                continue  # Continue processing the next image if an error occurs
    
    # Preprocess the synthetic images

    # Move them into synthetic and masks folders
    real_folder = train_folder

    output_folder = singan_similarity_output_folder
    synthetic_folder = os.path.join(split_singan_output_folder, 'synthetic_images')
    masks_folder = os.path.join(split_singan_output_folder, 'masks')

    # Clear the output, synthetic, and masks folders if they exist
    # It is safe to remove these because their data is outputted to the RandomSamples folder
    # These are basically temporary folders
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    if os.path.exists(synthetic_folder):
        shutil.rmtree(synthetic_folder)
    if os.path.exists(masks_folder):
        shutil.rmtree(masks_folder)

    os.makedirs(synthetic_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(output_train_folder, exist_ok=True)

    # Clone synthetic images and masks into their respective folders
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

    # Generate similarity scores; filter by usability
    process_folders(real_folder, synthetic_folder, masks_folder, output_folder)
    
    evaluation_results = evaluate_folder(real_folder, output_folder)
    # evaluation_results = pd.read_csv('evaluation_results.csv')
    high_quality_images = set(list(evaluation_results[evaluation_results['SSIM'] > 0.5]['Image'].unique()))
    print(f"High quality synthetic images: {len(high_quality_images)}")

    # Move them into the ready folder -> Transfer into singan augmented dataset folder
    move_singan_results_to_preprocessing(synthetic_output_folder, output_train_folder, high_quality_images)

    # Transfer the training images and masks into the augmented training folder
    for file in os.listdir(unet_training_folder):
        if file.endswith('_image.png') or file.endswith('_stomach_mask.png'):
            shutil.copy(os.path.join(unet_training_folder, file), os.path.join(output_train_folder, file))

    # Transfer the synthetic images and masks into the augmented training folder
    masks_folder_set = set(os.listdir(masks_folder))
    num_synthetic_images_added = 0

    for file in os.listdir(synthetic_folder):
        if file in masks_folder_set:
            last_underscore_index = file.rfind('_')
            if last_underscore_index == -1:
                last_underscore_index = len(file)
            file_no_end_number = file[:last_underscore_index]
            matching_file = file_no_end_number + '.png'
            if matching_file in high_quality_images:
                new_filename = file[:-4] + '_fake_image.png'
                img_path = os.path.join(synthetic_folder, file)
                img = io.read_image(img_path)[:3]
                gray = TF.rgb_to_grayscale(img, num_output_channels=1)

                dest = os.path.join(output_train_folder, new_filename)
                io.write_png(gray, dest)

                # Transfer the mask
                mask_path = os.path.join(masks_folder, file)
                mask_img = io.read_image(mask_path)[:3]
                gray =  TF.rgb_to_grayscale(mask_img, num_output_channels=1)

                new_mask_filename = file[:-4] + '_fake_stomach_mask.png'
                mask_dest = os.path.join(output_train_folder, new_mask_filename)
                
                io.write_png(gray, mask_dest)

                num_synthetic_images_added += 1

    print(f"Added {num_synthetic_images_added} synthetic images to the augmented training folder.")
    save_synthetic_images_count('singan-seg', ratio_synthesized_to_real, num_synthetic_images_added)

# Save the number of synthetic images added for the given generative model and ratio in a pandas DF
def save_synthetic_images_count(gen_model, synthetic_real_ratio, count):
    """Saves the count of synthetic images added to a CSV file."""
    df = pd.DataFrame({
        'Generative Model': [gen_model],
        'Synthetic to Real Ratio': [synthetic_real_ratio],
        'Count of Synthetic Images Added': [count]
    })
    output_file = os.path.join(unet_and_synthesis_results_folder, 'synthetic_images_count.csv')
    
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, index=False)


def benchmark_unet(
    training_folder,
    validation_folder=unet_validation_folder,
    gen_model = "",
    synthetic_real_ratio = 0
):
    timestamp = np.datetime64('now', 's').astype(str).replace(':', '-')
    python_command = f"python '{train_unet_py}' --train_dir '{training_folder}' --val_dir '{validation_folder}' --gen_model {gen_model} --synthetic_real_ratio {synthetic_real_ratio} --exp_id 'model-{gen_model}-synthetic-real-ratio-{synthetic_real_ratio}-{timestamp}'"
    try:
        print(f"Running command: {python_command}")
        result = subprocess.run(python_command, shell=True, check=True)
        print("UNet training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during UNet training: {e}")
    
def main():
    # Acquire the Madison dataset, and split it into training and testing sets (80/20)
    prepare_dataset()
    
    # Select a pretrained generative model or no generative model, and synthesize images
    # Place this in an augmented dataset folder, and use it to train a UNet model
    generative_models = [
        'none',
        'singan-seg',
        'ddim',
        'vae'
    ]

    exp_timestamp = np.datetime64('now', 's').astype(str).replace(':', '-')

    for model in generative_models:
        if model == 'none':
            print("No generative model selected, skipping synthesis.")
            continue
        elif model == 'singan-seg':
            print("Using Singan-Seg for synthesis.")
            # Randomly select 170 images from the training set as bases for synthesis
            for synthetic_real_ratio in [0.5, 1.0, 5.0, 9.0][::-1]:
                folder_name = f"{unet_and_synthesis_results_folder}/train_augmented_singan_synthetic_real_ratio_{synthetic_real_ratio}".replace('.', '_')
                synthetic_output_folder = f"{random_samples_folder}_synthetic_real_ratio_{synthetic_real_ratio}".replace('.', '_')
                os.makedirs(train_augmented_singan_folder, exist_ok=True)
                synthesize_with_singan(
                    output_train_folder=folder_name,
                    synthetic_output_folder=synthetic_output_folder,
                    ratio_synthesized_to_real=synthetic_real_ratio,
                    skip_untrained_models=SKIP_UNTRAINED_MODELS 
                )
                benchmark_unet(
                    folder_name,
                    gen_model="singan-seg",
                    synthetic_real_ratio=synthetic_real_ratio  # Ratio of synthetic to real data in training    
                )
        elif model == 'ddim':
            print("DDIM synthesis is not supported yet.")
        elif model == 'vae':
            print("VAE synthesis is not supported yet.")
        else:
            print(f"Unknown generative model: {model}. Skipping synthesis.")
            continue
            
    # Evaluate segmentation model on the validation dataset

if __name__ == "__main__":
    main()