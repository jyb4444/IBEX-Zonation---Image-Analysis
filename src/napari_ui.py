import napari
import numpy as np
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget, QSlider, QLineEdit, QLabel
from PyQt5.QtCore import Qt
from napari.utils.colormaps import DirectLabelColormap

def setup_napari_viewer(tile):
    """Sets up the Napari viewer with image and label layers."""
    viewer = napari.Viewer()

    # Add empty tile layer
    tile_layer = viewer.add_image(np.zeros_like(tile), name="tile_layer", opacity=0)

    # Separate channels and add them as individual image layers
    num_channels = tile.shape[2]
    colormaps = ['red', 'blue', 'green', 'yellow', 'gray']
    for i in range(num_channels):
        channel = tile[:, :, i]
        viewer.add_image(channel, name=f"Channel {i+1}", colormap=colormaps[i])

    # Add labels layer for contour annotation
    labels_layer = viewer.add_labels(
        np.zeros(tile.shape[:2], dtype=np.uint32),
        name='Contour Annotation'
    )

    # Configure labels layer
    labels_layer.blending = 'additive'
    labels_layer.opacity = 0.8
    labels_layer.brush_size = 10
    colors = {1: [255, 0, 255, 255]}
    labels_layer.color = colors

    # Create control widgets
    control_widget = QWidget()
    layout = QVBoxLayout()
    control_widget.setLayout(layout)

    # Add control widget to napari
    print("Adding control widget...")
    viewer.window.add_dock_widget(control_widget, area="right", name="Controls")
    print("Control widget added.")

    return viewer, tile_layer, labels_layer, control_widget

def create_threshold_sliders(layout, channel_layer, channel_idx):
    """Creates threshold sliders for a specific channel."""
    min_data, max_data = channel_layer.data.min(), channel_layer.data.max()

    min_slider = QSlider(Qt.Horizontal)
    max_slider = QSlider(Qt.Horizontal)
    min_slider.setRange(int(min_data), int(max_data))
    max_slider.setRange(int(min_data), int(max_data))
    min_slider.setValue(int(min_data))
    max_slider.setValue(int(max_data))

    min_input = QLineEdit()
    max_input = QLineEdit()
    min_input.setText(str(int(min_data)))
    max_input.setText(str(int(max_data)))

    def apply_threshold(value):
        min_val = min_slider.value()
        max_val = max_slider.value()
        channel_layer.data = np.clip(channel_layer.data, min_val, max_val)
        channel_layer.refresh()

    def apply_manual_input():
        try:
            min_val = float(min_input.text())
            max_val = float(max_input.text())
            min_slider.setValue(int(min_val))
            max_slider.setValue(int(max_val))
            channel_layer.data = np.clip(channel_layer.data, min_val, max_val)
            channel_layer.refresh()
        except ValueError:
            pass

    def update_input_from_slider():
        min_input.setText(str(min_slider.value()))
        max_input.setText(str(max_slider.value()))

    min_slider.valueChanged.connect(apply_threshold)
    max_slider.valueChanged.connect(apply_threshold)
    min_input.textChanged.connect(apply_manual_input)
    max_input.textChanged.connect(apply_manual_input)
    min_slider.valueChanged.connect(update_input_from_slider)
    max_slider.valueChanged.connect(update_input_from_slider)

    layout.addWidget(QLabel(f"Channel {channel_idx + 1} Min Threshold:"))
    layout.addWidget(min_slider)
    layout.addWidget(min_input)
    layout.addWidget(QLabel(f"Channel {channel_idx + 1} Max Threshold:"))
    layout.addWidget(max_slider)
    layout.addWidget(max_input)

def connect_events(viewer, tile_layer, labels_layer, image, tile_size, annotation_manager, current_annotation, train_model_func, get_random_tile_func, debug_show_mask_func, apply_model_to_image_func):
    """Connects UI events to their corresponding functions."""

    def load_next_image():
        global current_annotation
        # Remove all layers
        viewer.layers.clear()

        # Get a new random tile
        tile, (x_offset, y_offset) = get_random_tile_func(image, tile_size)

        # Re-add layers
        tile_layer = viewer.add_image(tile, name="tile_layer", opacity=0)
        num_channels = tile.shape[2]
        colormaps = ['red', 'blue', 'green', 'yellow', 'gray']
        for i in range(num_channels):
            channel = tile[:, :, i]
            viewer.add_image(channel, name=f"Channel {i+1}", colormap=colormaps[i])

        labels_layer = viewer.add_labels(
            np.zeros(tile.shape[:2], dtype=np.uint32),
            name='Contour Annotation'
        )
        labels_layer.blending = 'additive'
        labels_layer.opacity = 0.8
        labels_layer.brush_size = 10
        colors = {1: [255, 0, 255, 255]}
        labels_layer.color = colors

        print(f"Loaded new tile, position: ({x_offset}, {y_offset})")

        # Reset current annotation
        current_annotation = {
            "tile_position": (int(x_offset), int(y_offset)),
            "tile_size": tile_size,
            "has_label_mask": False,
            "timestamp": str(np.datetime64('now'))
        }

        # Recreate threshold sliders
        control_widget = viewer.window.dock_widgets['Controls'].widget
        layout = control_widget.layout()
        for i, layer in enumerate(viewer.layers):
            if layer.name.startswith("Channel"):
                create_threshold_sliders(layout, layer, i)

    def save_current_annotations():
        global current_annotation
        labels_data = labels_layer.data
        has_labels = np.any(labels_data > 0)

        if current_annotation is None:
            current_annotation = {
                "tile_position": (int(x_offset), int(y_offset)),
                "tile_size": tile_size,
                "has_label_mask": False,
                "timestamp": str(np.datetime64('now'))
            }

        if has_labels:
            annotation_metadata = annotation_manager.save_mask(labels_data, current_annotation["tile_position"], tile_size)
            if annotation_metadata:
                debug_show_mask_func(viewer, annotation_metadata["mask_file"])
                current_annotation["has_label_mask"] = True
                current_annotation["mask_file"] = annotation_metadata["mask_file"]
                annotation_manager.add_annotation(current_annotation)
            else:
                print("Warning: Mask file was not created successfully!")
        else:
            print("No annotations found to save")

    def view_current_annotations():
        labels_data = labels_layer.data
        unique_values = np.unique(labels_data)
        labeled_pixels = np.sum(labels_data > 0)

        print(f"Current annotation stats:")
        print(f"  - Label values: {unique_values}")
        print(f"  - Labeled pixels: {labeled_pixels}")
        print(f"  - Total size: {labels_data.shape}")

        if labeled_pixels == 0:
            print("No annotations in current view.")
        else:
            temp_view = np.zeros(labels_data.shape, dtype=np.uint8)
            temp_view[labels_data > 0] = 255
            for layer in list(viewer.layers):
                if layer.name.startswith("Current Annotations Debug"):
                    viewer.layers.remove(layer)
            viewer.add_image(temp_view, name="Current Annotations Debug", colormap='yellow')

    def refresh_view():
        for layer in list(viewer.layers):
            if layer.name.startswith("Current Annotations Debug") or layer.name.startswith("Debug: mask_"):
                viewer.layers.remove(layer)

        temp_data = tile_layer.data.copy()
        tile_layer.data = temp_data

        temp_labels = labels_layer.data.copy()
        labels_layer.data = np.zeros_like(temp_labels)
        labels_layer.data = temp_labels

        print("Interface has been refreshed")

    # Create buttons
    next_button = QPushButton("Next Image")
    save_button = QPushButton("Save Current Annotation")
    view_button = QPushButton("View Current Annotation")
    refresh_button = QPushButton("Refresh Interface")
    train_button = QPushButton("Train Model")

    # Connect buttons to functions
    next_button.clicked.connect(load_next_image)
    save_button.clicked.connect(save_current_annotations)
    view_button.clicked.connect(view_current_annotations)
    refresh_button.clicked.connect(refresh_view)
    train_button.clicked.connect(lambda: train_model_func(annotation_manager.annotations, image, viewer))

    # Add buttons to layout
    control_widget = viewer.window._dock_widgets['Controls'].widget
    layout = control_widget.layout()
    layout.addWidget(next_button)
    layout.addWidget(save_button)
    layout.addWidget(view_button)
    layout.addWidget(refresh_button)
    layout.addWidget(train_button)

    # Create threshold sliders for each channel
    for i, layer in enumerate(viewer.layers):
        if layer.name.startswith("Channel"):
            create_threshold_sliders(layout, layer, i)