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

SKIP_SINGAN_TRAINING = False
SKIP_UNTRAINED_MODELS = False
SHORT_SINGAN_TRAINING = False

full_dataset_folder = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg/Input/data-RGBA'
full_dataset_folder_image_and_masks = '/home/miguel/GI/0 - Data Exploration & Analysis/UW-Madison/stomach_data_and_masks_preparation/stomach_data_and_masks'

train_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/train'

unet_and_synthesis_results_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results'
train_augmented_singan_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/train_augmented_singan'
test_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/test'
singan_similarity_output_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/singan_output_from_similarity'

split_singan_output_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/split_singan_output'
random_samples_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/Output/RandomSamples2'

main_train_py = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg/main_train.py'
random_samples_py = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg/random_samples.py'
train_unet_py = '/home/miguel/GI/1 - Segmentation/Run-UNET/scripts/train_unet.py'

trained_models_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/TrainedModels'

unet_training_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/unet_training'
unet_validation_folder = '/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/unet_validation'




train_test_split_ratio = 0.8  # 80% for training, 20% for testing

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
    synthetic_real_ratio = 0,
    num_epochs = 30
):
    timestamp = np.datetime64('now', 's').astype(str).replace(':', '-')
    python_command = f"python '{train_unet_py}' --train_dir '{training_folder}' --val_dir '{validation_folder}' --gen_model {gen_model} --synthetic_real_ratio {synthetic_real_ratio} --exp_id 'model-{gen_model}-synthetic-real-ratio-{synthetic_real_ratio}-{timestamp}' --epoch {num_epochs}"
    try:
        print(f"Running command: {python_command}")
        result = subprocess.run(python_command, shell=True, check=True)
        print("UNet training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during UNet training: {e}")
    return f'model-{gen_model}-synthetic-real-ratio-{synthetic_real_ratio}-{timestamp}'

import ast
def parse_metrics(filepath):
    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, val = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            # history fields are Python lists—use ast.literal_eval for safety
            if key in ("train_loss_history", "val_loss_history"):
                metrics[key] = ast.literal_eval(val)
            else:
                # convert numeric strings to int or float
                if val.isdigit():
                    metrics[key] = int(val)
                else:
                    try:
                        metrics[key] = float(val)
                    except ValueError:
                        metrics[key] = val
    return metrics

def clear_output_folders():
    # Clear the random samples folders
    for folder in os.listdir('Output/'):
        if 'RandomSamples2' in folder:
            folder_path = os.path.join('Output/', folder)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)

    # Clear the training folders
    for folder in os.listdir('train-folders/'):
        if 'train-' in folder:
            folder_path = os.path.join('train-folders/', folder)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)

    # Clear the validation folder
    if os.path.exists('validation'):
        shutil.rmtree('validation')

def main():
    clear_output_folders()
    
    SKIP_UNAUGMENTED_TRAINING = False
    synthesis_multiplier = 2

    sample_sizes = [
        10, 20, 30, 40, 50, 
        60, 70, 80, 90, 100,
        150, 200
    ] # In reference to the number of subjects to use

    evaluation_results_original = pd.read_csv('evaluation_results_original.csv')
    high_quality_images = evaluation_results_original[evaluation_results_original['SSIM'] > 0.5]
    
    high_quality_image_case_day = [image for image in high_quality_images['Image'].values.tolist()]

    train_set_case_days = pd.read_csv('train_set_case_days.csv')['0'].values.tolist()

    if (not os.path.exists('validation')) or (not os.listdir('validation')):
        os.makedirs('validation', exist_ok=True)
        # Designate and clone the validation subjects
        # Validation is the same for all experiments
        val_set_case_days = pd.read_csv('val_set_case_days.csv')['0'].values.tolist()

        for case_day in tqdm(val_set_case_days):
            # Find the files that begin with the case day
            matching_files = glob.glob(os.path.join(full_dataset_folder_image_and_masks, f"{case_day}*"))

            for file in matching_files:
                file = file.split('/')[-1]
                base_name = file[:-4]  # Remove the '.png' extension
                file = base_name + '.png'
                if 'mask' in file:
                    shutil.copy(
                        os.path.join(full_dataset_folder_image_and_masks, file), 
                        os.path.join('validation', file.split('/')[-1])
                    )
                else:
                    shutil.copy(
                        os.path.join(full_dataset_folder_image_and_masks, file), 
                        os.path.join('validation', base_name + '_image.png')
                    )
                
    else:
        print("Validation folder already exists and is not empty. Skipping validation set preparation.")

    all_stats = []
            
    for sample_size in sample_sizes:
        num_hq_samples = min(len(high_quality_image_case_day), sample_size)
        high_quality_cases = random.sample(high_quality_image_case_day, num_hq_samples)

        # Individual images. Will be used to synthesize data to accompany their corresponding case days.
        synthetic_training_images = high_quality_cases.copy()

        # Locate the corresponding case days
        synthetic_image_cases_set = set([image.split('_slice')[0] for image in high_quality_cases])
        synthetic_image_cases = list(synthetic_image_cases_set)
        
        cases = synthetic_image_cases.copy()
        
        if len(cases) < sample_size:
            # Real case-day pairs
            high_quality_images_no_model_available = random.sample(train_set_case_days, sample_size - len(high_quality_cases))
            cases += [image.split('_slice')[0] for image in high_quality_images_no_model_available]

        print(f"Sample size: {sample_size}, Total cases: {len(cases)}, High quality cases: {len(high_quality_cases)}")

        # Prepare the training folder
        training_folder = f"train-folders/train-{sample_size}"
        os.makedirs(training_folder, exist_ok=True)

        num_images_in_synthetic_image_cases = 0

        for case_day in tqdm(cases):
            # Find the files that begin with the case day
            matching_files = glob.glob(os.path.join(full_dataset_folder_image_and_masks, f"{case_day}*"))

            if case_day in synthetic_image_cases_set:
                num_images_in_synthetic_image_cases += len(matching_files)

            for file in matching_files:
                try:
                    file = file.split('/')[-1]
                    base_name = file[:-4]  # Remove the '.png' extension
                    file = base_name + '.png'
                    if 'mask' in file:
                        shutil.copy(
                            os.path.join(full_dataset_folder_image_and_masks, file), 
                            os.path.join(training_folder, file.split('/')[-1])
                        )
                    else:
                        shutil.copy(
                            os.path.join(full_dataset_folder_image_and_masks, file), 
                            os.path.join(training_folder, base_name + '_image.png')
                        )
                except Exception as e:
                    print(e)

        # Train the UNet on the original training dataset; acquire statistics.
        if not SKIP_UNAUGMENTED_TRAINING:
            stats_export_location = benchmark_unet(
                training_folder=training_folder,
                validation_folder='validation',
                gen_model = "none",
                synthetic_real_ratio = 0,
                num_epochs = 30
            )

            print(f"Original training stats exported to: {stats_export_location}")
            results_path = f'/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/unet_performance_statistics/{stats_export_location}/best_dice_epoch.txt'

            stat_row = {
                'num_subjects': len(cases),
                'num_real_images': len(os.listdir(training_folder)),
                'num_synthetic_images': 0,
                'synthetic_to_real_ratio': 0,

                'unet_training_min_loss': 0,
                'unet_validation_peak_accuracy': 0,
                'unet_validation_min_loss': 0,
            }

            with open(results_path, 'r') as file:
                file_text = file.read()
                training_exp_data = file_text.split('\n')

                stat_row['unet_training_min_loss'] = float(training_exp_data[2].split(' ')[-1])
                stat_row['unet_validation_peak_accuracy'] = float(training_exp_data[1].split(' ')[-1])
                stat_row['unet_validation_min_loss'] = float(training_exp_data[3].split(' ')[-1])
                all_stats.append(stat_row)

            pd.DataFrame(all_stats).to_csv(
                'unet-performance-statistics.csv',
            )

        # Synthesize images
        synthetic_output_folder = f"{random_samples_folder}_sample_size_{sample_size}"

        # Determine the number of synthetic images to generate per sample -- average number of real images per case used for synthetic synthesis
        num_to_synthesize = max(num_images_in_synthetic_image_cases // len(synthetic_training_images), 10) * synthesis_multiplier

        print('Number of synthetic images to generate per sample:', num_to_synthesize, 'for a total of', num_to_synthesize * len(synthetic_training_images), 'synthetic images.')

        for img_name in tqdm(synthetic_training_images):
            python_command = f"python '{random_samples_py}' --input_name '{img_name}' --input_dir='{full_dataset_folder}' --mode random_samples --gen_start_scale 0 --nc_z 4 --nc_im 4 --gpu_id 0 --num_samples {num_to_synthesize} --out '{synthetic_output_folder}'"
            try:
                print(f"Running command: {python_command}")
                subprocess.run(python_command, shell=True, check=True)
                print("UNet training completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred during UNet training: {e}")

        # Move the synthetic images into the training folder
        synthetic_output_folder_random_samples_path = synthetic_output_folder + '/RandomSamples/'
        for case_day in os.listdir(synthetic_output_folder_random_samples_path):
            case_day_path = os.path.join(synthetic_output_folder_random_samples_path, case_day)
            folder_in_path = os.listdir(case_day_path)
            case_day_path_extended = os.path.join(synthetic_output_folder_random_samples_path, case_day, folder_in_path[0])
            for file in os.listdir(case_day_path_extended):
                file_path = os.path.join(case_day_path_extended, file)
                new_file_name = case_day + '_' + file.split('_')[0] + '_fake_' 
                if 'mask' in file:
                    new_file_name += 'stomach_mask.png'
                else:
                    new_file_name += 'image.png'

                img = io.read_image(file_path)[:3]
                gray = TF.rgb_to_grayscale(img, num_output_channels=1)

                dest = os.path.join(training_folder, new_file_name)
                io.write_png(gray, dest)

        # Train the Unet on the augmented dataset; acquire statistics.
        stats_export_location = benchmark_unet(
            training_folder=training_folder,
            validation_folder='validation',
            gen_model = "singan-seg",
            synthetic_real_ratio = 1,
            num_epochs = 30
        )

        print(f"Augmented training stats exported to: {stats_export_location}")
        results_path = f'/home/miguel/GI/1 - Segmentation/UNet-and-Synthesis-Results/unet_performance_statistics/{stats_export_location}/best_dice_epoch.txt'

        all_files = os.listdir(training_folder)
        num_fake = len([file for file in all_files if 'fake' in file])
        num_real = len([file for file in all_files if 'fake' not in file])

        stat_row = {
            'num_subjects': len(cases),
            'num_real_images': num_real,
            'num_synthetic_images': num_fake,
            'synthetic_to_real_ratio': num_fake / num_real if num_real > 0 else 0,

            'unet_training_min_loss': 0,
            'unet_validation_peak_accuracy': 0,
            'unet_validation_min_loss': 0
        }

        with open(results_path, 'r') as file:
            file_text = file.read()
            training_exp_data = file_text.split('\n')

            stat_row['unet_training_min_loss'] = float(training_exp_data[2].split(' ')[-1])
            stat_row['unet_validation_peak_accuracy'] = float(training_exp_data[1].split(' ')[-1])
            stat_row['unet_validation_min_loss'] = float(training_exp_data[3].split(' ')[-1])
            all_stats.append(stat_row)

        pd.DataFrame(all_stats).to_csv(
            'unet-performance-statistics.csv',
        )

if __name__ == "__main__":
    main()