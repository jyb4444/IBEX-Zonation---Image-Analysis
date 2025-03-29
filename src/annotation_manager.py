import os
import json
import numpy as np

class AnnotationManager:
    def __init__(self, output_path):
        self.annotations = []
        self.output_path = output_path

    def add_annotation(self, annotation):
        self.annotations.append(annotation)

    def save_annotations(self):
        annotations_path = os.path.join(self.output_path, "annotations.json")
        with open(annotations_path, "w") as f:
            json.dump(self.annotations, f)
        print(f"All annotations saved to {annotations_path}, total: {len(self.annotations)} images")

    def save_mask(self, labels_data, tile_position, tile_size):
        mask_filename = os.path.join(self.output_path, f"mask_{len(self.annotations)}.npy")
        np.save(mask_filename, labels_data)
        if os.path.exists(mask_filename):
            print(f"Successfully saved mask to file: {mask_filename}")
            annotation_metadata = {
                "mask_file": mask_filename,
                "tile_position": tile_position,
                "tile_size": tile_size
            }
            return annotation_metadata
        else:
            print(f"Warning: File {mask_filename} was not created successfully!")
            return None