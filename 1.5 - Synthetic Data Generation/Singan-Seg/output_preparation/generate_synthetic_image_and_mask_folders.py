import os
import shutil

def main():
    singanseg_path = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg/'

    # Folder to retrieve the data from
    raw_random_samples_path = os.path.join(singanseg_path, 'output_preparation', 'raw_random_samples')

    # Folder to save the processed data to
    random_samples_split_path = os.path.join(singanseg_path, 'output_preparation', 'random_samples_split')

    random_samples_synthetic_images_path = os.path.join(random_samples_split_path, 'synthetic_images')
    random_samples_mask_path = os.path.join(random_samples_split_path, 'masks')

    # Ensure destination directories exist
    os.makedirs(random_samples_synthetic_images_path, exist_ok=True)
    os.makedirs(random_samples_mask_path, exist_ok=True)

    # Iterate over each subfolder in raw_random_samples_path
    for entry in os.listdir(raw_random_samples_path):
        folder_path = os.path.join(raw_random_samples_path, entry)
        if not os.path.isdir(folder_path):
            continue

        # Within each subfolder, copy *_img.png into synthetic_images and *_mask.png into masks
        for fname in os.listdir(folder_path):
            src = os.path.join(folder_path, fname)

            if fname.endswith('_img.png'):
                dst = os.path.join(random_samples_synthetic_images_path, entry+".png")
                shutil.copy2(src, dst)
            elif fname.endswith('_mask.png'):
                dst = os.path.join(random_samples_mask_path, entry+".png")
                shutil.copy2(src, dst)

if __name__ == '__main__':
    main()
