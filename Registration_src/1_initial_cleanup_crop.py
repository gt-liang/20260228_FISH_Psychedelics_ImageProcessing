"""Collects .czi images from raw_data, applies MIP, crops to common shape, and saves as .npy arrays."""

# %%
import os
import numpy as np
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from loguru import logger

logger.info('Import ok')

# Set working directory
new_directory = '/Users/ronanoconnell/Library/CloudStorage/OneDrive-BaylorCollegeofMedicine/BFL/IMAGE PROCESSING SCRIPTS/20250729_20x_FISH_images - in progress/'
os.chdir(new_directory)

# Define folders
input_path = 'raw_data'
output_folder = 'python_results/initial_cleanup2/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ---------------- Step 1: Determine min crop size ----------------
logger.info('Scanning for minimum image dimensions...')

file_list = [f for f in os.listdir(input_path) if f.endswith('.czi')]
image_shapes = []

for filename in file_list:
    img = AICSImage(os.path.join(input_path, filename)).get_image_data("CZYX")
    mip = np.asarray([np.max(img[ch], axis=0) for ch in range(img.shape[0])])  # MIP
    image_shapes.append(mip.shape[1:])  # [Y, X]

min_height = min([shape[0] for shape in image_shapes])
min_width = min([shape[1] for shape in image_shapes])
crop_shape = (min_height, min_width)

logger.info(f"All images will be cropped to: {crop_shape}")

# ---------------- Step 2: Convert, crop, and save images ----------------
def czi_converter(image_name, input_folder, output_folder, tiff=False, array=True, mip=True, crop_shape=None):
    """Convert a single .czi to cropped npy or tif format"""
    os.makedirs(output_folder, exist_ok=True)

    # Load and MIP
    image = AICSImage(f'{input_folder}.czi').get_image_data("CZYX")
    if mip:
        image = np.asarray([np.max(image[ch], axis=0) for ch in range(image.shape[0])])

    # Crop to standard shape
    if crop_shape is not None:
        H, W = crop_shape
        image = image[:, :H, :W]

    # Save outputs
    if tiff:
        OmeTiffWriter.save(image, f'{output_folder}{image_name}.tif', dim_order='CYX')
    if array:
        np.save(f'{output_folder}{image_name}.npy', image)

# ---------------- Step 3: Clean image names & process ----------------
do_not_quantitate = []

image_names = []
for filename in file_list:
    if all(word not in filename for word in do_not_quantitate):
        clean_name = filename.split('.czi')[0]
        image_names.append(clean_name)

# Remove duplicates
image_names = list(dict.fromkeys(image_names))

# Convert and save each image
for name in image_names:
    czi_converter(
        name,
        input_folder=os.path.join(input_path, name),
        output_folder=output_folder,
        crop_shape=crop_shape
    )

logger.info('Initial cleanup complete. All images saved with consistent size.')
# %%
