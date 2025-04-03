import argparse
import os
import napari
import torch
import sys
import numpy as np
import skimage.io as skio
from skimage.transform import resize

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src import image_processing, napari_ui, model_training, annotation_manager

def parse_args():
    parser = argparse.ArgumentParser(description="Napari Annotation and Model Training.")
    parser.add_argument("-i", "--image", required=True, help="Path to the NRRD image file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory to save files.")
    parser.add_argument("-p", "--prediction", type=str, help='Path to the prediction result image')
    parser.add_argument("-m", "--mask", type=str, help="Path to the contour mask file (optional).")
    parser.add_argument("-s", "--size", type=int, default=6000, help="Size of the tile (default: 6000).")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Initialize AnnotationManager
    annotation_manager_instance = annotation_manager.AnnotationManager(args.output)

    # Load image, mask and prediction
    if args.prediction and os.path.exists(args.prediction):
        print(f"Loading image with prediction: {args.prediction}")
        image, tissue_mask, prediction_mask = image_processing.load_image(args.image, args.mask, args.prediction)
    else:
        print("Loading image without prediction")
        image, tissue_mask = image_processing.load_image(args.image, args.mask)
        prediction_mask = None

    # Check if image was loaded successfully
    if image is None:
        print("Error: Could not load image. Exiting.")
        return  # Exit the function if image loading failed
    
    # Get initial tile
    if prediction_mask is not None:
        tile, (x_offset, y_offset), pred_tile = image_processing.get_random_tile_with_prediction(image, args.size, prediction_mask)
    else:
        tile, (x_offset, y_offset), pred_tile = image_processing.get_random_tile(image, args.size)

    # Setup Napari viewer and UI
    viewer, tile_layer, labels_layer, control_widget = napari_ui.setup_napari_viewer(tile, pred_tile)

    # Initialize current annotation context
    current_annotation = {
        "tile_position": (int(x_offset), int(y_offset)),
        "tile_size": args.size,
        "has_label_mask": False,
        "timestamp": str(np.datetime64('now'))
    }
    
    # Connect UI events to functions
    napari_ui.connect_events(
        viewer,
        tile_layer,
        labels_layer,
        image,
        args.size,
        annotation_manager_instance,
        current_annotation,
        model_training.train_model,
        lambda img, size: image_processing.get_random_tile_with_prediction(img, size, prediction_mask),
        image_processing.debug_show_mask,
        model_training.apply_model_to_image
    )

    # Display usage instructions
    print("""
    Annotation Instructions:
    1. Use the paint brush tool (brush icon) in the left toolbar to draw contours on the "Contour Annotation" layer
    2. After drawing contours on the current image, click "Save Current Annotation" button on the right panel
    3. Click "Next Image" button to load a new image
    4. After annotating all images, click "Train Model" button to start model training
    5. Click "View Current Annotation" to check if your annotations are being captured

    Note: 
    - The annotation layer uses a bright magenta color for better visibility
    - The prediction layer (if available) is shown in cyan
    - You must click "Save Current Annotation" button after drawing to save your work
    - If you don't see your annotations, try using the "View Current Annotation" button
    """)

    # Run the Napari application
    napari.run()

if __name__ == "__main__":
    main()