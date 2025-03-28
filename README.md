# Active learning deep neural network for liver vessel detection


## Prerequisites

- **Python 3.x** (preferably 3.7+)
- **Napari** - for interactive image visualization.
- **QtPy** - for building the graphical interface with sliders and input fields.
- **NumPy** - for numerical operations on image data.

### Command-Line Flags


The script accepts the following command-line flags:

- `-i`, `--image` (required): Path to the NRRD image file.
- `-m`, `--mask` (optional): Path to the contour mask file (in the same format as the image).
- `-o`, `--output` (required): Directory to save the output results.
- `-s`, `--size` (optional): Size of the tile (default: 6000). 

### Example Command

To run the script with the required arguments, use:
### Example Usage

1. **Run with sliders and manual input enabled (default)**:
   ```bash
   python napari_select.py -i path/to/your/image.npy -o path/to/output/image.npy -m /path/to/tissue_mask.png -s 6000
