import os
import shutil

import glob
import numpy as np
from tqdm import tqdm
from PIL import Image

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

def main():
    dataset_path = '../../0 - Data Exploration & Analysis/GI-Roberta/gi-roberta-dataset/4D_MRI_GI_Roberta'

    path = f'{dataset_path}/4D_MRI_GI_Roberta/Different ways to view the data/View-Slicewise-Per-Subject'
    target_path = '../../0 - Data Exploration & Analysis/GI-Roberta/gi-roberta-dataset/full_dataset2'
    full_dataset_rgba_path = '../../0 - Data Exploration & Analysis/GI-Roberta/gi-roberta-dataset/full_dataset_rgba2'

    os.makedirs(target_path, exist_ok=True)

    for folder in os.listdir(path):
        if '_pngs' in folder:
            folder_path = os.path.join(path, folder)
            for file in os.listdir(folder_path):
                if file.endswith('.png'):
                    file_path = os.path.join(folder_path, file)
                    new_file_path = os.path.join(target_path, file)
                    shutil.copy(file_path, new_file_path)
                    print(f'Copied: {file_path} to {new_file_path}')

    process_to_rgba(target_path, full_dataset_rgba_path)

if __name__ == "__main__":
    main()