# Author: Miguel Aenlle

import os
import argparse
import subprocess
from tqdm import tqdm

INPUT_FOLDER_PATH = '../../1.5 - Synthetic Data Generation/Singan-Seg/Input/finetuning-data-RGBA'
TRAINED_MODELS_FOLDER_PATH = 'TrainedModels'

RANDOM_SAMPLES_PY = '../../1.5 - Synthetic Data Generation/Singan-Seg/random_samples.py'
MAIN_TRAIN_PY = '../../1.5 - Synthetic Data Generation/Singan-Seg/main_train.py'

NUM_SAMPLES_TO_GENERATE = 50
OUTPUT_FOLDER = '../../1.5 - Synthetic Data Generation/Singan-Seg/Output/FinetuningSamples'

def main():
    singanseg_folder = os.path.abspath('../../../../1.5 - Synthetic Data Generation/Singan-Seg')
    os.chdir(singanseg_folder)

    # Retrieve the gpu_id
    # Retrieve the start index

    parser = argparse.ArgumentParser(description="Generate random samples for SinGAN training.")
    parser.add_argument('--start_index', type=int, default=0, help='Start index for processing files, inclusive.')
    parser.add_argument('--end_index', type=int, help='End index for processing files, inclusive.')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID to use for training.')

    args = parser.parse_args()

    start_index = args.start_index
    end_index = args.end_index if args.end_index is not None else None
    gpu_id = args.gpu_id

    # Omit everything with images already
    synthesized_images_path = os.path.join(OUTPUT_FOLDER, 'RandomSamples')

    synthesized_images_set = set([file + '.png' for file in os.listdir(synthesized_images_path)])

    files_without_images = []
    for file in os.listdir(INPUT_FOLDER_PATH):
        if file not in synthesized_images_set:
            files_without_images.append(file)
            
    files_without_images = files_without_images[start_index:end_index + 1] if end_index is not None else files_without_images[start_index:]

    for file in tqdm(files_without_images):
        if os.path.exists(os.path.join(TRAINED_MODELS_FOLDER_PATH, file[:-4])):
            python_command = f"python '{RANDOM_SAMPLES_PY}' --input_name {file} --input_dir='{INPUT_FOLDER_PATH}' --mode random_samples --gen_start_scale 0 --nc_z 4 --nc_im 4 --gpu_id {gpu_id} --num_samples {NUM_SAMPLES_TO_GENERATE} --out '{OUTPUT_FOLDER}'"
        else:
            python_command = f"python '{MAIN_TRAIN_PY}' --input_name {file}  --input_dir='{INPUT_FOLDER_PATH}' --nc_z 4 --nc_im 4 --gpu_id {gpu_id} --num_samples {NUM_SAMPLES_TO_GENERATE} --out '{OUTPUT_FOLDER}'"
        print(f"Running command: {python_command}")
        try:
            subprocess.run(python_command, shell=True, check=True)
            print(f"Command completed successfully for {file}\n")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {file}: {e}")
            continue

if __name__ == "__main__":
    main()
