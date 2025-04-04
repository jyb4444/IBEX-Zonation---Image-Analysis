import os
import json
import numpy as np
import random
import string

class AnnotationManager:
    def __init__(self, output_path):
        self.annotations = []
        self.output_path = output_path

    def add_annotation(self, annotation):
        self.annotations.append(annotation)

    def _generate_random_filename(self, length=10):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    def save_annotations(self):
        annotations_path = os.path.join(self.output_path, "annotations.json")
        with open(annotations_path, "w") as f:
            json.dump(self.annotations, f)
        print(f"All annotations saved to {annotations_path}, total: {len(self.annotations)} images")

    def save_mask(self, labels_data, tile_position, tile_size):
        # mask_filename = os.path.join(self.output_path, f"mask_{len(self.annotations)}.npy")
        random_string = self._generate_random_filename()
        mask_filename = os.path.join(self.output_path, f"mask_{random_string}.npy")
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