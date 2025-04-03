import numpy as np
import os
from skimage import io as skio
from skimage.transform import resize
import random
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  

try:
    import nrrd
except ImportError:
    try:
        import pynrrd as nrrd
        print("Using pynrrd instead of nrrd")
    except ImportError:
        print("Error: Cannot import nrrd or pynrrd library")
        exit(1)

def load_image(image_path, mask_path=None, prediction_path=None):
    """Loads an NRRD image, an optional mask image, and an optional prediction result.
    
    Args:
        image_path (str): Path to the NRRD image file
        mask_path (str, optional): Path to the contour mask file
        prediction_path (str, optional): Path to the prediction result image
        
    Returns:
        tuple: (image, tissue_mask, prediction_mask) if prediction_path is provided,
               otherwise (image, tissue_mask)
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image, _ = nrrd.read(image_path)
        image = image.astype(np.float32, copy=True)
        print(f"Successfully loaded image: {image_path}, shape: {image.shape}")

        tissue_mask = None
        if mask_path:
            if not os.path.exists(mask_path):
                print(f"Warning: Mask file not found: {mask_path}")
            else:
                os.environ['SKIMAGE_ALLOW_HUGE_IMAGES'] = '1'
                tissue_mask = skio.imread(mask_path)
                print(f"Successfully loaded mask: {mask_path}, shape: {tissue_mask.shape}")

                if len(tissue_mask.shape) > 2:
                    tissue_mask = tissue_mask[:, :, 0] > 0
                else:
                    tissue_mask = tissue_mask > 0

                if tissue_mask.shape[:2] != image.shape[:2]:
                    print(f"Warning: Mask size {tissue_mask.shape[:2]} does not match image size {image.shape[:2]}. Resizing mask.")
                    tissue_mask = resize(tissue_mask, image.shape[:2], order=0, preserve_range=True).astype(bool)

                if len(image.shape) == 3:
                    for i in range(image.shape[2]):
                        background_value = np.min(image[:, :, i])
                        temp = image[:, :, i].copy()
                        temp[~tissue_mask] = background_value
                        image[:, :, i] = temp
                else:
                    background_value = np.min(image)
                    image[~tissue_mask] = background_value

                print("Successfully applied tissue boundary mask")
        
        prediction_mask = None
        if prediction_path:
            if not os.path.exists(prediction_path):
                print(f"Warning: Prediction file not found: {prediction_path}")
            else:
                os.environ['SKIMAGE_ALLOW_HUGE_IMAGES'] = '1'
                prediction_mask = skio.imread(prediction_path)
                print(f"Successfully loaded prediction: {prediction_path}, shape: {prediction_mask.shape}")
                
                if len(prediction_mask.shape) > 2:
                    prediction_mask = prediction_mask[:, :, 0]
                prediction_mask = (prediction_mask > 0).astype(np.uint32)
                
                if prediction_mask.shape != image.shape[:2]:
                    print(f"Warning: Prediction size {prediction_mask.shape} does not match image size {image.shape[:2]}. Resizing prediction.")
                    prediction_mask = resize(prediction_mask, image.shape[:2], 
                                        order=0, preserve_range=True).astype(np.uint32)
        
        if prediction_path:
            return image, tissue_mask, prediction_mask
        else:
            return image, tissue_mask

    except FileNotFoundError as e:
        print(f"Error: {e}")
        if prediction_path:
            return None, None, None
        else:
            return None, None
    except Exception as e:
        print(f"Error loading image or mask: {e}")
        if prediction_path:
            return None, None, None
        else:
            return None, None
        
def load_image_with_prediction(image_path, prediction_path, mask_path=None):
    """Loads an NRRD image, a prediction image and an optional mask image.
    
    Note: This is a wrapper around the updated load_image function that now handles predictions.
    """
    return load_image(image_path, mask_path, prediction_path)

def get_random_tile(image, tile_size):
    """Extracts a random tile from the image."""
    if image is None:
        print("Error: Image is None. Cannot extract tile.")
        return None, (0, 0), None

    max_x, max_y = image.shape[:2]
    
    if max_x <= tile_size or max_y <= tile_size:
        print(f"Warning: Image dimensions ({max_x}, {max_y}) smaller than tile size {tile_size}. Using whole image.")
        return image, (0, 0), None
    
    x = random.randint(0, max_x - tile_size)
    y = random.randint(0, max_y - tile_size)
    
    if len(image.shape) == 3:
        tile = image[x:x + tile_size, y:y + tile_size, :]
    else:
        tile = image[x:x + tile_size, y:y + tile_size]
    
    return tile, (x, y), None

def get_random_tile_with_prediction(image, tile_size, prediction_mask=None):
    """Extracts a random tile from the image and corresponding prediction mask."""
    if image is None:
        print("Error: Image is None. Cannot extract tile.")
        return None, (0, 0), None

    max_x, max_y = image.shape[:2]
    
    if max_x <= tile_size or max_y <= tile_size:
        print(f"Warning: Image dimensions ({max_x}, {max_y}) smaller than tile size {tile_size}. Using whole image.")
        if prediction_mask is not None:
            if len(image.shape) == 3:
                return image, (0, 0), prediction_mask
            else:
                return image, (0, 0), prediction_mask
        else:
            return image, (0, 0), None
    
    if prediction_mask is not None and np.any(prediction_mask > 0):
        attempts = 0
        max_attempts = 10  
        
        while attempts < max_attempts:
            x = random.randint(0, max_x - tile_size)
            y = random.randint(0, max_y - tile_size)
            
            pred_region = prediction_mask[x:x + tile_size, y:y + tile_size]
            if np.sum(pred_region) > 100: 
                break
                
            attempts += 1
    else:
        x = random.randint(0, max_x - tile_size)
        y = random.randint(0, max_y - tile_size)
    
    if len(image.shape) == 3:
        tile = image[x:x + tile_size, y:y + tile_size, :]
    else:
        tile = image[x:x + tile_size, y:y + tile_size]
    
    if prediction_mask is not None:
        pred_tile = prediction_mask[x:x + tile_size, y:y + tile_size]
        return tile, (x, y), pred_tile
    else:
        return tile, (x, y), None

def debug_show_mask(viewer, mask_file):
    """Displays a saved mask for debugging."""
    try:
        mask_data = np.load(mask_file)
        viewer.add_labels(mask_data, name=f"Debug: {os.path.basename(mask_file)}")
        print(f"Loaded mask for debugging: {mask_file}")
        print(f"Mask shape: {mask_data.shape}, Unique values: {np.unique(mask_data)}")
    except Exception as e:
        print(f"Error loading mask for debug: {e}")

def normalize_image(image):
    """归一化图像到0-1范围。"""
    if image is None:
        return None
        
    if len(image.shape) == 3:
        normalized = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[2]):
            channel = image[:, :, i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val > min_val:
                normalized[:, :, i] = (channel - min_val) / (max_val - min_val)
            else:
                normalized[:, :, i] = channel - min_val  
        return normalized
    else:
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        else:
            return image - min_val 