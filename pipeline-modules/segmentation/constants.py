TRAIN_EVAL_UNET_PY = '../../1 - Segmentation/Run-UNET/scripts/train_eval_unet.py'
UNET_LOGS_PATH = '../../1 - Segmentation/UNet-and-Synthesis-Results/unet_logs'
UNET_STATISTICS_PATH = '../performance-statistics'

MODEL_INFO = {
    'vanilla': {
        'dataset_path': '../../1.5 - Synthetic Data Generation/Singan-Seg/unet_singan_augmented_datasets_2',
        'architecture': 'vanilla',
        'expansion_factors': [0],
    },
    'uwm-unet': {
        'dataset_path': '../../1.5 - Synthetic Data Generation/Singan-Seg/unet_singan_augmented_datasets_2',
        'architecture': 'uwm-unet',
        'expansion_factors': [0],
    },
    'uwm-unet-singan': {
        'dataset_path': '../../1.5 - Synthetic Data Generation/Singan-Seg/unet_singan_augmented_datasets_with_finetuning',
        'architecture': 'uwm-unet',
        'expansion_factors': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    },
    'uwm-unet-singan-finetuned-n-10-10x': {
        'dataset_path': '../../1.5 - Synthetic Data Generation/Singan-Seg/unet_singan_augmented_datasets_with_finetuning_num_images_10',
        'architecture': 'uwm-unet',
        'expansion_factors': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    },
    'uwm-unet-singan-finetuned-n-5-10x': {
        'dataset_path': '../../1.5 - Synthetic Data Generation/Singan-Seg/unet_singan_augmented_datasets_with_finetuning_num_images_5',
        'architecture': 'uwm-unet',
        'expansion_factors': [0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    },
    'uwm-unet-singan-finetuned-n-10-1x': {
        'dataset_path': '../../1.5 - Synthetic Data Generation/Singan-Seg/unet_singan_augmented_datasets_with_finetuning_num_images_10_no_finetuning_augmentation',
        'architecture': 'uwm-unet',
        'expansion_factors': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    },
    'uwm-unet-singan-finetuned-n-5-1x': {
        'dataset_path': '../../1.5 - Synthetic Data Generation/Singan-Seg/unet_singan_augmented_datasets_with_finetuning_num_images_5_no_finetuning_augmentation',
        'architecture': 'uwm-unet',
        'expansion_factors': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    },
    'uwm-unet-singan-ss-finetuned-ft-aug': {
        'dataset_path': '../../1.5 - Synthetic Data Generation/Singan-Seg/unet_singan_augmented_datasets_with_finetuning_num_images_10',
        'architecture': 'uwm-unet',
        'expansion_factors': [0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        'self_supervised_finetuning': True,
        'pixelwise_confidence_threshold': 0.99
    },
    'uwm-unet-singan-ss-finetuned-ft-no-aug': {
        'dataset_path': '../../1.5 - Synthetic Data Generation/Singan-Seg/unet_singan_augmented_datasets_with_finetuning_num_images_10_self_supervised_no_finetuning_augmentation',
        'architecture': 'uwm-unet',
        'expansion_factors': [0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        'self_supervised_finetuning': True,
        'pixelwise_confidence_threshold': 0.99
    },
    'uwm-unet-singan-ss-finetuned-ft-aug-n-5-10x': {
        'dataset_path': '../../1.5 - Synthetic Data Generation/Singan-Seg/unet_singan_augmented_datasets_with_finetuning_num_images_5_self_supervised',
        'architecture': 'uwm-unet',
        'expansion_factors': [0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        'self_supervised_finetuning': True,
        'pixelwise_confidence_threshold': 0.99
    },
    'uwm-unet-singan-ss-finetuned-ft-aug-n-5-1x': {
        'dataset_path': '../../1.5 - Synthetic Data Generation/Singan-Seg/unet_singan_augmented_datasets_with_finetuning_num_images_5_self_supervised_no_finetuning_augmentation',
        'architecture': 'uwm-unet',
        'expansion_factors': [0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        'self_supervised_finetuning': True,
        'pixelwise_confidence_threshold': 0.99
    },
    'uwm-unet-diffusion-3': {
        'dataset_path': '../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/augmented_datasets_varied_expansion_factor',
        'architecture': 'uwm-unet',
        'expansion_factors': [0.5],
        'custom_cv_subjects': ['FD-027', 'FD-029', 'FD-031', 'FD-032'],
    },
    'uwm-unet-diffusion-high-ssim': {
        'dataset_path': '../../1.5 - Synthetic Data Generation/diffuse-gen/diffuse-gen/augmented_datasets_varied_expansion_factor_ssim',
        'architecture': 'uwm-unet',
        'expansion_factors': [0.5, 0.75],
        'custom_cv_subjects': ['FD-027', 'FD-029', 'FD-031', 'FD-032'],
    }
}