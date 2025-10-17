# src/prediction_service.py

import os
import numpy as np
import cv2
import rasterio
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.models import Model
from src.model_definition import build_unet # Need to import the structure

IMG_SIZE = 128
CLASSES = 10

def preprocess_and_predict(model, tiff_path, mask_path=None):
    """
    Loads TIFF input, preprocesses it, and runs the U-Net prediction.
    Returns: processed_image, predicted_mask, true_mask (NumPy arrays).
    """
    
    # 1. Load Input Data (TIFF)
    try:
        with rasterio.open(tiff_path) as src:
            img_raw = src.read(1)
    except Exception:
        return None, None, None

    # 2. Load True Mask Data (PNG)
    mask_true = None
    if mask_path and os.path.exists(mask_path):
        mask_true = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Ensure mask is scaled correctly for comparison (0-9 classes)
        if mask_true is not None:
             mask_true = cv2.resize(mask_true, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST).astype(np.uint8) 

    # 3. Preprocessing for Model Input
    img_processed = resize(img_raw, (IMG_SIZE, IMG_SIZE), anti_aliasing=True).astype(np.float32)
    
    # Normalize
    if img_processed.max() > 0:
        img_processed /= img_processed.max()
    
    # Add batch and channel dimensions: (1, 128, 128, 1)
    X_input = img_processed[np.newaxis, ..., np.newaxis] 

    # 4. Predict
    prediction_one_hot = model.predict(X_input)
    
    # 5. Post-process (Convert one-hot output back to 0-9 mask)
    mask_predicted = np.argmax(prediction_one_hot, axis=-1).squeeze()

    return img_processed, mask_predicted, mask_true

def colorize_mask(mask_2d):
    """Converts the 0-9 mask to a colorful RGB image for visualization."""
    import matplotlib.cm as cm
    
    # Use a visually distinct colormap (10 colors)
    cmap = cm.get_cmap('Spectral', CLASSES) 
    
    # Normalize mask values to 0-1 range for the colormap
    normalized_mask = mask_2d / (CLASSES - 1)
    
    # Apply colormap to create RGBA array
    colored_mask_rgba = cmap(normalized_mask)
    
    # Return RGB part (dropping the alpha channel)
    return colored_mask_rgba[:, :, :3]
