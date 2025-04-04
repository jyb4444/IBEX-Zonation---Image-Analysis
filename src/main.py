import argparse
import os
import napari
import torch
import random
import sys
import string
import numpy as np
import skimage.io as skio
from skimage.transform import resize
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget, QSlider, QLineEdit, QLabel, QMessageBox
from PyQt5.QtCore import Qt

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src import image_processing, napari_ui, model_training, annotation_manager

def show_instructions_dialog():
    """Shows an instruction dialog with annotation guidelines."""
    instructions = """
    Annotation Instructions:
    0. Please add your name in command line like '-u "kejiyuan"'.
    1. Use the paint polygon tool (polygon icon) in the left toolbar to draw contours on the "Contour Annotation" layer
    2. After drawing contours on the current image, click "Save Current Annotation" button on the right panel
    3. Click "Next Image" button to load a new image
    4. "Train Model" button does not work now. 
    5. Click "View Current Annotation" to check if your annotations are being captured

    Note: 
    - The default size of tile is 3000
    - The annotation layer uses a bright magenta color for better visibility
    - The prediction layer (if available) is shown in cyan
    - You must click "Save Current Annotation" button after drawing to save your work
    - If you don't see your annotations, try using the "View Current Annotation" button
    """
    
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Annotation Instructions")
    msg_box.setText(instructions)
    msg_box.setIcon(QMessageBox.Information)
    msg_box.exec_()

def parse_args():
    parser = argparse.ArgumentParser(description="Napari Annotation and Model Training.")
    parser.add_argument("-i", "--image", required=True, help="Path to the NRRD image file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory to save files.")
    parser.add_argument("-p", "--prediction", type=str, help='Path to the prediction result image')
    parser.add_argument("-m", "--mask", type=str, help="Path to the contour mask file (optional).")
    parser.add_argument("-s", "--size", type=int, default=3000, help="Size of the tile (default: 3000).")
    parser.add_argument("-u", "--username", required=True, type=str, help="The user who use napari")
    return parser.parse_args()

def main():
    args = parse_args()
    characters = string.ascii_letters + string.digits
    random_str = ''.join(random.choice(characters) for _ in range(10))
    print(random_str)
    output_folder = args.output + "_" + random_str + "_" + args.username
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize AnnotationManager
    annotation_manager_instance = annotation_manager.AnnotationManager(output_folder)

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
    
    # Add help button to the control widget
    help_button = QPushButton("Show Instructions")
    help_button.clicked.connect(show_instructions_dialog)
    
    # Add the help button to the control layout
    control_layout = control_widget.layout()
    control_layout.addWidget(help_button)
    
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

    # Show instructions dialog at startup
    show_instructions_dialog()

    # Run the Napari application
    napari.run()

if __name__ == "__main__":
    main()