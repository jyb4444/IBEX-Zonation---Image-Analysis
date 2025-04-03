from skimage import io as skio
import nrrd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

Image.MAX_IMAGE_PIXELS = None  

def create_overlay_visualization(nrrd_path, mask_path, output_path=None, zoom_regions=None):
    """
    Creates an overlay visualization of NRRD image and prediction mask.
    
    Parameters:
    nrrd_path: Path to the NRRD file.
    mask_path: Path to the mask image.
    output_path: Path to save the output image (optional).
    zoom_regions: List of regions to zoom in and display, format [(y_start, y_end, x_start, x_end), ...]
    
    Returns:
    None
    """
    image, header = nrrd.read(nrrd_path)
    print(f"NRRD Image Shape: {image.shape}")
    
    prediction_mask = skio.imread(mask_path)
    print(f"Prediction Mask Shape: {prediction_mask.shape}")
    
    print(f"Total NRRD Pixels: {np.prod(image.shape)}")
    print(f"Total Prediction Mask Pixels: {np.prod(prediction_mask.shape)}")
    
    if image.shape[:2] != prediction_mask.shape[:2]:
        print(f"Height Difference: {image.shape[0] - prediction_mask.shape[0]}")
        print(f"Width Difference: {image.shape[1] - prediction_mask.shape[1]}")
        return
    
    scale_factor = 10  
    small_image = image[::scale_factor, ::scale_factor]
    small_mask = prediction_mask[::scale_factor, ::scale_factor]
    
    if len(small_image.shape) == 3 and small_image.shape[2] == 3:
        rgb_image = small_image / 255.0 if small_image.max() > 1.0 else small_image
    else:
        normalized = small_image.astype(float) / small_image.max()
        rgb_image = np.stack([normalized, normalized, normalized], axis=2)
    
    fig = plt.figure(figsize=(20, 15))
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(rgb_image)
    ax1.set_title('Original NRRD Image (Scaled Down)')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(small_mask, cmap='gray')
    ax2.set_title('Prediction Mask (Scaled Down)')
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 2, 3)
    overlay = np.zeros((*rgb_image.shape[:2], 4))
    overlay[..., :3] = rgb_image
    overlay[..., 3] = 1.0  
    
    ax3.imshow(overlay)
    
    mask_rgba = np.zeros((*small_mask.shape, 4))
    mask_rgba[small_mask > 0, 0] = 1.0 
    mask_rgba[small_mask > 0, 3] = 0.7 
    
    ax3.imshow(mask_rgba)
    ax3.set_title('Overlay View (Scaled Down)')
    ax3.axis('off')
    
    ax4 = plt.subplot(2, 2, 4)
    
    if zoom_regions and len(zoom_regions) > 0:
        y_start, y_end, x_start, x_end = zoom_regions[0]
        
        zoom_image = image[y_start:y_end, x_start:x_end]
        zoom_mask = prediction_mask[y_start:y_end, x_start:x_end]
        
        if len(zoom_image.shape) == 3 and zoom_image.shape[2] == 3:
            zoom_rgb = zoom_image / 255.0 if zoom_image.max() > 1.0 else zoom_image
        else:
            normalized = zoom_image.astype(float) / zoom_image.max()
            zoom_rgb = np.stack([normalized, normalized, normalized], axis=2)
        
        zoom_overlay = np.zeros((*zoom_rgb.shape[:2], 4))
        zoom_overlay[..., :3] = zoom_rgb
        zoom_overlay[..., 3] = 1.0
        
        ax4.imshow(zoom_overlay)
        
        zoom_mask_rgba = np.zeros((*zoom_mask.shape, 4))
        zoom_mask_rgba[zoom_mask > 0, 0] = 1.0  
        zoom_mask_rgba[zoom_mask > 0, 3] = 0.7  
        
        ax4.imshow(zoom_mask_rgba)
        ax4.set_title(f'Zoomed Region ({x_start},{y_start}) to ({x_end},{y_end})')
    else:
        y_indices, x_indices = np.where(small_mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            center_y = int(np.mean(y_indices)) * scale_factor
            center_x = int(np.mean(x_indices)) * scale_factor
            
            size = 500  
            y_start = max(0, center_y - size//2)
            y_end = min(image.shape[0], center_y + size//2)
            x_start = max(0, center_x - size//2)
            x_end = min(image.shape[1], center_x + size//2)
            
            zoom_image = image[y_start:y_end, x_start:x_end]
            zoom_mask = prediction_mask[y_start:y_end, x_start:x_end]
            
            if len(zoom_image.shape) == 3 and zoom_image.shape[2] == 3:
                zoom_rgb = zoom_image / 255.0 if zoom_image.max() > 1.0 else zoom_image
            else:
                normalized = zoom_image.astype(float) / zoom_image.max()
                zoom_rgb = np.stack([normalized, normalized, normalized], axis=2)
            
            zoom_overlay = np.zeros((*zoom_rgb.shape[:2], 4))
            zoom_overlay[..., :3] = zoom_rgb
            zoom_overlay[..., 3] = 1.0
            
            ax4.imshow(zoom_overlay)
            
            zoom_mask_rgba = np.zeros((*zoom_mask.shape, 4))
            zoom_mask_rgba[zoom_mask > 0, 0] = 1.0  
            zoom_mask_rgba[zoom_mask > 0, 3] = 0.7  
            
            ax4.imshow(zoom_mask_rgba)
            ax4.set_title(f'Auto-Detected Region of Interest ({x_start},{y_start}) to ({x_end},{y_end})')
        else:
            ax4.text(0.5, 0.5, 'No Mask Area Found', horizontalalignment='center', verticalalignment='center')
    
    ax4.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    plt.show()
    
    mask_pixels = np.sum(prediction_mask > 0)
    total_pixels = prediction_mask.size
    coverage_percentage = (mask_pixels / total_pixels) * 100
    
    print(f"\nOverlay Statistics:")
    print(f"Mask Covered Pixels: {mask_pixels}")
    print(f"Total Pixels: {total_pixels}")
    print(f"Coverage Percentage: {coverage_percentage:.2f}%")
    
    if mask_pixels > 0:
        mask_values = prediction_mask[prediction_mask > 0]
        print(f"Mask Value Range: {mask_values.min()} to {mask_values.max()}")
        print(f"Mask Mean: {mask_values.mean():.2f}")
        print(f"Mask Standard Deviation: {mask_values.std():.2f}")
    
    return fig

if __name__ == "__main__":
    nrrd_path = "../image/PORTAL_CENTRAL_FULL_SECTION_20250131.nrrd"
    mask_path = "../image/prediction_result.png"
    output_path = "../image/overlap_visualization.png"
    
    # zoom_regions = [(1000, 1500, 1000, 1500)]
    
    create_overlay_visualization(nrrd_path, mask_path, output_path)