import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import dice

def load_image_gray(image_path):
    """Load an image from a file path."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.uint8)

def load_image_normalized(image_path):
    """Load an image from a file path."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    # Normalize to [0, 1] range
    image = image.astype(np.float32) / 255.0
    return image

def compute_ssim_mse(real_roi, synthetic_roi):
    """Compute SSIM and MSE between two ROIs."""
    real_gray = cv2.cvtColor(real_roi.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.uint8)
    synthetic_gray = cv2.cvtColor(synthetic_roi.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.uint8)

    ssim_value = ssim(real_gray, synthetic_gray)
    mse_value = np.mean((real_gray - synthetic_gray) ** 2)
    return ssim_value, mse_value

def compute_dice(real_mask, synthetic_mask):
    """Compute Dice coefficient between two masks."""
    real_bin = (real_mask > 0.5).astype(np.uint8)
    synth_bin = (synthetic_mask > 0.5).astype(np.uint8)
    dice_value = 1 - dice(real_bin.flatten(), synth_bin.flatten())
    return dice_value
