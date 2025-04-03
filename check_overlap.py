import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import nrrd

Image.MAX_IMAGE_PIXELS = None  

def process_images(nrrd_path, png_path, output_path):
    """
    Extracts the red channel of an NRRD image and maps the PNG contours to the corresponding location.
    
    Parameters:
    nrrd_path - Path to the NRRD image.
    png_path - Path to the PNG image containing the contours.
    output_path - Path to save the output image.
    """
    print(f"Reading NRRD image: {nrrd_path}")
    nrrd_data, header = nrrd.read(nrrd_path)
    print(f"NRRD image shape: {nrrd_data.shape}")
    
    if len(nrrd_data.shape) == 3 and nrrd_data.shape[2] >= 3:
        red_channel = nrrd_data[:, :, 0]
        print("Successfully extracted red channel")
    else:
        print("Warning: NRRD image is not in the expected 3-channel format, using the first available channel")
        red_channel = nrrd_data[:, :, 0] if len(nrrd_data.shape) == 3 else nrrd_data
    
    print(f"Reading PNG contour image: {png_path}")
    try:
        contours = np.array(Image.open(png_path).convert('L'))
        print(f"Contour image shape: {contours.shape}")
    except Exception as e:
        print(f"Error reading contour image: {e}")
        print("Attempting to read large image using alternative method...")
        
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(png_path)
        
        new_size = (red_channel.shape[1], red_channel.shape[0])
        contours = np.array(img.resize(new_size, Image.NEAREST).convert('L'))
        print(f"Resized contour image shape: {contours.shape}")
    
    if red_channel.shape != contours.shape:
        print(f"Warning: Image dimensions do not match. NRRD red channel: {red_channel.shape}, Contour image: {contours.shape}")
        print("Resizing contour image to match NRRD image...")
        contours_resized = np.array(Image.fromarray(contours).resize((red_channel.shape[1], red_channel.shape[0])))
        contours = contours_resized
        print(f"Resized contour image shape: {contours.shape}")
    
    red_normalized = (red_channel - np.min(red_channel)) / (np.max(red_channel) - np.min(red_channel)) * 255
    red_normalized = red_normalized.astype(np.uint8)
    
    rgb_image = np.stack([red_normalized, red_normalized, red_normalized], axis=2)
    
    contour_mask = (contours > 127) 
    rgb_image[contour_mask, 0] = 0  
    rgb_image[contour_mask, 1] = 255
    rgb_image[contour_mask, 2] = 0  
    
    print(f"Saving result to: {output_path}")
    Image.fromarray(rgb_image).save(output_path)
    
    downsample_factor = max(1, rgb_image.shape[0] // 1000, rgb_image.shape[1] // 1000)
    if downsample_factor > 1:
        print(f"Image too large, downsampling for display (factor: {downsample_factor})")
        
    red_display = red_normalized[::downsample_factor, ::downsample_factor]
    contours_display = contours[::downsample_factor, ::downsample_factor]
    rgb_display = rgb_image[::downsample_factor, ::downsample_factor]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Red Channel")
    plt.imshow(red_display, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Contour Image")
    plt.imshow(contours_display, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Combined Result")
    plt.imshow(rgb_display)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_comparison.png'))
    plt.show()
    
    print("Processing complete!")

if __name__ == "__main__":
    nrrd_path = "./image/PORTAL_CENTRAL_FULL_SECTION_20250131.nrrd"
    png_path = "./restored_contours.png"
    output_path = "red_channel_with_contours.png"
    
    process_images(nrrd_path, png_path, output_path)