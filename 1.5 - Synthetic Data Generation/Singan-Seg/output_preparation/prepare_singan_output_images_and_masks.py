import os
import shutil 

def main():
    singanseg_path = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg/'

    # Folder to retrieve the data from
    output_path = os.path.join(singanseg_path, 'Output', 'RandomSamples')

    # Folder to save the processed data
    target_path = os.path.join(singanseg_path, 'output_preparation', 'raw_random_samples')

    os.makedirs(target_path, exist_ok=True)

    for case_folder in os.listdir(output_path):
        for generation_folder in os.listdir(os.path.join(output_path, case_folder)):
            files = os.listdir(os.path.join(output_path, case_folder, generation_folder))
            for file in files:
                file_id = file[:file.find('_') if '_' in file else len(file)]
                case_folder_concat = case_folder + '_' + file_id

                # Rename the file to match the expected format
                old_path = os.path.join(output_path, case_folder, generation_folder, file)
                new_path = os.path.join(target_path, case_folder_concat, file)

                os.makedirs(os.path.dirname(new_path), exist_ok=True)

                shutil.move(old_path, new_path)
                print(f"Moved {old_path} to {new_path}")
    print(f"All files have been moved to {target_path}")

if __name__ == '__main__':
    main()