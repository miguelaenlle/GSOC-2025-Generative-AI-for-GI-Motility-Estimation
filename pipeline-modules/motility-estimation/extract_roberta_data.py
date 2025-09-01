# Author: Miguel Aenlle

import os
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.io import loadmat
from IPython.display import clear_output


def display_slices(image_mat, mask_mat, num=10):
    """
    Display up to 'num' slices from image_mat, clearing output after each.
    """
    for i in range(min(num, image_mat.shape[2])):
        padX = 1
        padY = 8
        image = image_mat[:, :, i][padX:-padX, padY:-padY]  # Crop the image to remove borders
        mask = mask_mat[:, :, i][padX:-padX, padY:-padY]  # Crop the mask to remove borders

        image_uint8 = (image * 255).astype('uint8')
        mask_uint8 = (mask).astype('uint8')

        mask_uint8[mask_uint8 > 0] = 255  # Replace all mask values greater than 0 with 255

        # Resize to 256x256
        image_uint8 = cv2.resize(image_uint8, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask_uint8 = cv2.resize(mask_uint8, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Replace all mask values greater than 0 with 255

        fig, ax = plt.subplots()
        ax.imshow(255 - image_uint8, cmap='gray')

        # Plot the mask
        ax.imshow(mask_uint8, cmap='jet', alpha=0.5)
        plt.show()

        # Clear the cell output and close the figure
        clear_output(wait=True)
        plt.close(fig)

def main():
    path = '../../0 - Data Exploration & Analysis/GI-Roberta/gi-roberta-dataset/4D_MRI_GI_Roberta/4D_MRI_GI_Roberta/Different ways to view the data/View-3D-at-time-X'
    image_folder_path = '../../0 - Data Exploration & Analysis/GI-Roberta/gi-roberta-dataset/all-cine-mri-pngs-corrected2'
    os.makedirs(image_folder_path, exist_ok=True)

    padX = 1
    padY = 8
    for file in tqdm(os.listdir(path)):
        if file.endswith('.mat'):
            if 'image' in file:
                # Load the .mat file
                file_path = os.path.join(path, file)
                image_mat = loadmat(file_path)['image_time']

                for i in range(image_mat.shape[2]):
                    image = image_mat[:, :, i][padX:-padX, padY:-padY]  # Crop the image to remove borders
                    image_uint8 = (image * 255).astype('uint8')

                    image_uint8 = 255 - cv2.resize(image_uint8, (256, 256), interpolation=cv2.INTER_LINEAR)

                    file_split = file.split('_')

                    file_name = f"FD_{file_split[1]}_time_{file.split('_')[4][:-4]}_slice_{i}_image"

                    cv2.imwrite(os.path.join(image_folder_path, f"{file_name}.png"), image_uint8)
                

            elif 'mask' in file:
                # Load the .mat file
                file_path = os.path.join(path, file)
                mask_mat = loadmat(file_path)['mask_time']

                for i in range(mask_mat.shape[2]):
                    mask = mask_mat[:, :, i][padX:-padX, padY:-padY]  # Crop the mask to remove borders
                    mask_uint8 = (mask).astype('uint8')
                    mask_uint8[mask_uint8 > 0] = 255  # Replace all mask values greater than 0 with 255
                    mask_uint8 = cv2.resize(mask_uint8, (256, 256), interpolation=cv2.INTER_LINEAR)

                    file_split = file.split('_')
                    # Save the mask
                    file_name = f"FD_{file_split[1]}_time_{file.split('_')[4][:-4]}_slice_{i}_mask"
                    cv2.imwrite(os.path.join(image_folder_path, f"{file_name}.png"), mask_uint8)

if __name__ == '__main__':
    main()