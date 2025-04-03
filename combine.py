import numpy as np
import json
import os
try:
    import nrrd
except ImportError:
    try:
        import pynrrd as nrrd
        print("Using pynrrd instead of nrrd")
    except ImportError:
        print("Error: Cannot import nrrd or pynrrd library")
        exit(1)
import napari
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  

def restore_contours():
    if not os.path.exists("annotations.json"):
        print("Error: annotations.json file not found")
        return
    
    nrrd_filename = "./image/PORTAL_CENTRAL_FULL_SECTION_20250131.nrrd"
    print(f"Loading original NRRD file: {nrrd_filename}")
    try:
        image, header = nrrd.read(nrrd_filename)
        image = image.astype(np.float32, copy=True)  
        print(f"Original image shape: {image.shape}")
    except Exception as e:
        print(f"Error loading NRRD file: {e}")
        return
    
    global_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    print("Loading annotations.json")
    with open("./annotations.json", "r") as f:
        annotations = json.load(f)
    
    print(f"Found {len(annotations)} annotations")
    
    successful_masks = 0
    for i, anno in enumerate(annotations):
        if anno.get("has_label_mask", False) and "mask_file" in anno:
            mask_file = anno["mask_file"]
            
            if not os.path.exists(mask_file):
                print(f"Warning: Mask file {mask_file} not found")
                continue
            
            try:
                print(f"Loading mask file {mask_file}")
                label_mask = np.load(mask_file)
                
                x_pos, y_pos = anno["tile_position"]
                tile_size = anno["tile_size"]
                
                print(f"Annotation position: ({x_pos}, {y_pos}), size: {tile_size}")
                
                x_end = min(image.shape[0], x_pos + tile_size)
                y_end = min(image.shape[1], y_pos + tile_size)
                
                mask_height = x_end - x_pos
                mask_width = y_end - y_pos
                
                print(f"Application range: ({x_pos}:{x_end}, {y_pos}:{y_end})")
                print(f"Actual size: height={mask_height}, width={mask_width}")
                
                actual_mask = label_mask
                if label_mask.shape[0] > mask_height or label_mask.shape[1] > mask_width:
                    actual_mask = label_mask[:mask_height, :mask_width]
                    print(f"Adjusted mask size to fit target area")
                
                global_mask[x_pos:x_end, y_pos:y_end] = np.logical_or(
                    global_mask[x_pos:x_end, y_pos:y_end],
                    actual_mask > 0
                ).astype(np.uint8) * 255  
                
                successful_masks += 1
                print(f"Applied mask {i+1}/{len(annotations)}")
                
            except Exception as e:
                print(f"Error processing mask file {mask_file}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"Successfully restored {successful_masks}/{len(annotations)} masks")
    
    if np.sum(global_mask) == 0:
        print("Error: No contours were restored. The global mask is empty.")
        return
    
    try:
        mask_image = Image.fromarray(global_mask)
        mask_image.save("restored_contours.png")
        print("Restored contours saved to 'restored_contours.png'")
    except Exception as e:
        print(f"Error saving global mask as image: {e}")
    
    viewer = napari.Viewer()
    
    if len(image.shape) == 3:
        viewer.add_image(image, name="Original Image")
    else:
        viewer.add_image(image, name="Original Image")
    
    viewer.add_labels(global_mask, name="Restored Contours")
    
    print("Displaying results in napari. Close the viewer window when done.")
    napari.run()

if __name__ == "__main__":
    restore_contours()