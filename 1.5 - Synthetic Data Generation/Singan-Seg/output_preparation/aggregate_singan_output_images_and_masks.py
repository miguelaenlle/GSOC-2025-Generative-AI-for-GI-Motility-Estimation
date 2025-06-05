import os
from tqdm import tqdm

def main():
    singan_seg_path = '/home/miguel/GI/1.5 - Synthetic Data Generation/Singan-Seg'
    os.makedirs(singan_seg_path + '/Output/random_samples_merged', exist_ok = True)
    for folder in tqdm(os.listdir(singan_seg_path + '/Output/RandomSamples')):
        if len(os.listdir(singan_seg_path + f'output_preparation/raw_random_samples/{folder}')) == 2:
            # Copy both into stomach_data_and_masks
            os.rename(singan_seg_path + f'/output_preparation/random_samples_merged/{folder}/{folder}.png', f'random_samples_merged/{folder}.png')
            os.rename(singan_seg_path + f'/output_preparation/random_samples_merged/{folder}/{folder}_stomach_mask.png', f'random_samples_merged/{folder}_stomach_mask.png')
            
if __name__ == '__main__':
    main()
