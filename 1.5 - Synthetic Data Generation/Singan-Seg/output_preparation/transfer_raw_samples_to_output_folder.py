import os
import glob
import shutil
from tqdm import tqdm

def main():
    singanseg_path = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg/'

    # Folder to retrieve data from
    original_path = os.path.join(singanseg_path, 'output_preparation', 'raw_random_samples')

    # Folder to copy the data to
    target_path = os.path.join(singanseg_path, 'Output', 'RandomSamples')

    for case_folder in tqdm(os.listdir(original_path)):
        raw_case_folder = case_folder
        last_underscore_index = case_folder.rfind('_')
        case_folder = case_folder[:last_underscore_index] if last_underscore_index != -1 else case_folder
        case_folder_in_target = os.path.join(target_path, case_folder, 'gen_start_scale=0')
        if not os.path.exists(case_folder_in_target):
            print('Case folder does not exist in target:', case_folder)
            continue
        
        files = glob.glob(os.path.join(original_path, raw_case_folder, '*'))
        for file in files:
            file_name = os.path.basename(file)
            new_file_path = os.path.join(case_folder_in_target, file_name)

            # Copy the file to the target directory
            shutil.copy(file, new_file_path)
            print(f"Copied {file} to {new_file_path}")

if __name__ == '__main__':
    main()