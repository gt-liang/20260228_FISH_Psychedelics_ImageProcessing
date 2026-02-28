"""Applies cellpose algorithms to determine cellular and nuclear masks
"""
#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from cellpose import plot
from loguru import logger
from skimage import filters
from cellpose.io import logger_setup
logger_setup();

# Get the current working directory
current_directory = os.getcwd()
print("Current directory:", current_directory)
# Change the current working directory
new_directory = '/Users/ronanoconnell/Library/CloudStorage/OneDrive-BaylorCollegeofMedicine/BFL/IMAGE PROCESSING SCRIPTS/20250729_20x_FISH_images - in progress/'
os.chdir(new_directory)
# Verify the change
current_directory = os.getcwd()
print("Current directory:", current_directory)

input_folder = 'python_results/initial_cleanup/'
output_folder = 'python_results/cellpose_masking/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


def apply_cellpose(images, image_type='cyto', channels=None, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, resample=False):
    """Apply standard cellpose model to list of images.

    Args:
        images (ndarray): numpy array of 16 bit images
        image_type (str, optional): Cellpose model. Defaults to 'cyto'.
        channels (int, optional): define CHANNELS to run segementation on (grayscale=0, R=1, G=2, B=3) where channels = [cytoplasm, nucleus]. Defaults to None.
        diameter (int, optional): Expected diameter of cell or nucleus. Defaults to None.
        flow_threshold (float, optional): maximum allowed error of the flows for each mask. Defaults to 0.4.
        cellprob_threshold (float, optional): The network predicts 3 outputs: flows in X, flows in Y, and cell “probability”. The predictions the network makes of the probability are the inputs to a sigmoid centered at zero (1 / (1 + e^-x)), so they vary from around -6 to +6. Decrease this threshold if cellpose is not returning as many ROIs as you expect. Defaults to 0.0.
        resample (bool, optional): Resampling can create smoother ROIs but take more time. Defaults to False.

    Returns:
        ndarray: array of masks, flows, styles, and diameters
    """
    if channels is None:
        channels = [0, 0]
    model = models.Cellpose(model_type=image_type)
    masks, flows, styles, diams = model.eval(
        images, diameter=diameter, channels=channels, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, resample=resample)
    return masks, flows, styles, diams


def visualise_cell_pose(images, masks, flows, channels = None):
    """Display cellpose results for each image

    Args:
        images (ndarray): single channel (one array)
        masks (ndarray): one array
        flows (_type_): _description_
        channels (_type_, optional): _description_. Defaults to None.
    """
    if channels is None:
        channels = [0, 0]
    for image_number, image in enumerate(images):
        maski = masks[image_number]
        flowi = flows[image_number][0]
        
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, image, maski, flowi, channels=channels)
        plt.tight_layout()
        plt.show()


# ----------------Initialise file list----------------
file_list = [filename for filename in os.listdir(
    input_folder) if 'npy' in filename]

imgs = {filename.replace('.npy', ''): np.load(
    f'{input_folder}{filename}') for filename in file_list}

# ----------------Grab nucs for first dated channel----------------
# Strip .npy and split each filename into parts
clean_names = [filename.replace('.npy', '') for filename in file_list]
split_names = [name.split('_') for name in clean_names]

# Extract components
dates = [s[0] for s in split_names]
wells = ['_'.join(s[1:3]) for s in split_names]  # 20x_C3
full_names = ['_'.join(s[:3]) for s in split_names]  # 20250718_20x_C3

# Step 1: Find the first date
first_date = sorted(set(dates))[0]

# Step 2: Build ref image names from first date + unique wells
unique_wells = sorted(set(wells))
ref_images = [f'{first_date}_{well}' for well in unique_wells]

# Step 3: Find the first match in imgs for each well
to_process = {}
for key in imgs:
    for ref in ref_images:
        if key.startswith(ref):
            to_process[ref] = imgs[key]  # Only keep the first match per well
            break



# ----------------Outline nucs----------------
# Instead of extracting just channel 2 (DAPI), extract all channels per image [need to do this in case images are different sizes and require cropping]
all_channels_cropped = []

# Find minimum height and width across all images and channels
min_height = min(image.shape[1] for image in to_process.values())
min_width = min(image.shape[2] for image in to_process.values())

print(f"Cropping all images to uniform shape: ({min_height}, {min_width})")

# Crop all images (all channels) to uniform size
for name, image in to_process.items():
    cropped = image[:, :min_height, :min_width]  # image shape = (channels, height, width)
    all_channels_cropped.append(cropped)

# Now extract just nuclei channel (channel 2) for Cellpose
cellmask_channel_uniform = [img[2] for img in all_channels_cropped]

# Now run Cellpose on uniformly sized images
masks, flows, styles, diams = apply_cellpose(
    cellmask_channel_uniform, image_type='nuclei', diameter=20, flow_threshold=0.4,
    resample=True, cellprob_threshold=0.0)
#20x images use diameter = 20
#63x images use diameter = 60

# Save masks as a pickled numpy object (to handle possible shape differences)
np.save(f'{output_folder}cellpose_nucmasks.npy', masks, allow_pickle=True)
logger.info('Cellpose nuclear masks saved')




# %%
