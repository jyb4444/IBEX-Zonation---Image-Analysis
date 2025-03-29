import numpy as np
import os
from skimage import io as skio
from skimage.transform import resize
import random
import torch
try:
    import nrrd
except ImportError:
    try:
        import pynrrd as nrrd
        print("Using pynrrd instead of nrrd")
    except ImportError:
        print("Error: Cannot import nrrd or pynrrd library")
        exit(1)

def load_image(image_path, mask_path=None):
    """Loads an NRRD image and an optional mask image."""
    try:
        image, _ = nrrd.read(image_path)
        image = image.astype(np.float32, copy=True)

        tissue_mask = None
        if mask_path and os.path.exists(mask_path):
            os.environ['SKIMAGE_ALLOW_HUGE_IMAGES'] = '1'
            tissue_mask = skio.imread(mask_path)
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

        return image, tissue_mask

    except Exception as e:
        print(f"Error loading image or mask: {e}")
        return None, None

def get_random_tile(image, tile_size):
    """Extracts a random tile from the image."""
    max_x, max_y, _ = image.shape
    x = random.randint(0, max_x - tile_size)
    y = random.randint(0, max_y - tile_size)
    return image[x:x + tile_size, y:y + tile_size], (x, y)

def debug_show_mask(viewer, mask_file):
    """Displays a saved mask for debugging."""
    try:
        mask_data = np.load(mask_file)
        viewer.add_labels(mask_data, name=f"Debug: {mask_file}")
        print(f"Loaded mask for debugging: {mask_file}")
        print(f"Mask shape: {mask_data.shape}, Unique values: {np.unique(mask_data)}")
    except Exception as e:
        print(f"Error loading mask for debug: {e}")