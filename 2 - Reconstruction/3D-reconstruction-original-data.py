# Author: Elizabeth N.

import os
import argparse
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import shutil
from scipy import ndimage

def keep_largest_component_by_volume(mask: np.ndarray,
                                     spacing: tuple[float, float, float] = (1.,1.,1.),
                                     connectivity: int = 3
                                    ) -> np.ndarray:
    """
    Given a 3D boolean mask, label all 3D-connected components,
    compute their physical volumes (voxels × spacing), and
    return a boolean mask keeping only the largest one.
    """
    # build structuring element for 26-connectivity if connectivity==3
    if connectivity == 3:
        structure = np.ones((3,3,3), dtype=np.int32)
    else:
        structure = ndimage.generate_binary_structure(3, connectivity)

    labeled, num_cc = ndimage.label(mask, structure=structure)
    if num_cc == 0:
        return mask

    # voxel counts per label (0 is background)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0

    # convert counts → physical volumes
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    phys_vols = counts * voxel_vol

    # pick the label with the max volume
    largest_label = phys_vols.argmax()
    return (labeled == largest_label)

def main(
    input_folder: str
):
    # Path to the folder containing the 2D slices
    output_folder = 'Reconstructed-3D-original'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)    

    os.makedirs(output_folder, exist_ok=True)

    # Function to parse the file name and extract subject, time, and slice info
    def parse_filename(filename):
        parts = filename.split('_')
        
        try:
            subject = f"{parts[0]}_{parts[1]}"  # Combine 'FD' and '031'
            time_point = int(parts[3])  # '86' is in parts[3]
            slice_idx = int(parts[5])
        except (IndexError, ValueError) as e:
            raise ValueError(f"Error parsing time or slice index in file: {filename}") from e

        return subject, time_point, slice_idx

    # Filter files to include only masks
    files = [f for f in os.listdir(input_folder) if f.endswith('_mask.png')]
    data_dict = {}

    # Organize files by subject and time point
    for file in files:
        subject, time_point, slice_idx = parse_filename(file)
        if subject not in data_dict:
            data_dict[subject] = {}
        if time_point not in data_dict[subject]:
            data_dict[subject][time_point] = {}
        data_dict[subject][time_point][slice_idx] = file

    # Reconstruct 3D volumes for each subject and time point
    for subject, time_data in tqdm(data_dict.items(), desc="Reconstructing subjects"):
        subject_output_folder = os.path.join(output_folder, subject)
        os.makedirs(subject_output_folder, exist_ok=True)

        for time_point, slices in tqdm(time_data.items()):
            # Determine the number of slices
            max_slice_idx = max(slices.keys())
            
            # Load slices into a 3D numpy array
            slice_shape = None
            volume = []

            # Retrieve the slices, in order
            # If a slice is missing, fill it with zeros

            for slice_idx in range(1, max_slice_idx + 1):
                if slice_idx in slices:
                    slice_path = os.path.join(input_folder, slices[slice_idx])
                    try:
                        slice_image = np.array(Image.open(slice_path))  # Load PNG as numpy array
                    except Exception as e:
                        continue

                    if slice_shape is None:
                        slice_shape = slice_image.shape
                    volume.append(slice_image)
                else:
                    # Fill missing slices with zeros
                    if slice_shape is not None:
                        volume.append(np.zeros(slice_shape))

            # Stack slices into a 3D volume
            volume = np.stack(volume, axis=-1)  # Stack along the z-axis
            # Normalize volume to the range [0, 255]
            volume = volume.astype(np.float32)  # Ensure float for normalization
            volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))  # Normalize to [0, 1]
            volume = (volume * 255).astype(np.uint8)  # Scale to [0, 255] as uint8

            volume = keep_largest_component_by_volume(volume)
            volume = volume.astype(np.uint8) * 255

            volume_reoriented = np.transpose(volume, (1, 0, 2))

            # Save as a .nii.gz file
            output_path = os.path.join(subject_output_folder, f"{subject}_time_{time_point}_reconstructed.nii.gz")
            spacing = (1.0, 1.0, 1.0)  # Example: Isotropic spacing of 1 mm
            nii = nib.Nifti1Image(volume, np.eye(4))
            nii.header.set_zooms(spacing)  # Set voxel spacing
            nib.save(nii, output_path)

            
            # print(f"Reconstructed volume shape: {volume.shape}")
            # print(f"Intensity range: {volume.min()} to {volume.max()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D volume from 2D slices stored in a directory"
    )
    parser.add_argument(
        "-i", "--input_folder",
        required=True,
        help="Path to the folder containing the 2D slices"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(input_folder=args.input_folder)