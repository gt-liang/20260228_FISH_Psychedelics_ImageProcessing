"""Quality control: use napari to validate cellpose-generated masks
"""
#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import napari
from loguru import logger
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
from napari.settings import get_settings
get_settings().application.ipy_interactive = False

# Get the current working directory
current_directory = os.getcwd()
print("Current directory:", current_directory)
# Change the current working directory
new_directory = '/Users/ronanoconnell/Library/CloudStorage/OneDrive-BaylorCollegeofMedicine/BFL/IMAGE PROCESSING SCRIPTS/20250729_20x_FISH_images - in progress/'
os.chdir(new_directory)
# Verify the change
current_directory = os.getcwd()
print("Current directory:", current_directory)

image_folder = 'python_results/initial_cleanup/'
mask_folder = 'python_results/cellpose_masking/'
output_folder = 'python_results/napari_masking/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


def filter_masks(before_image, image_name, mask):
    """Quality control of cellpose-generated masks
    - Select the cell layer and using the fill tool set to 0, remove all unwanted cells.
    - Finally, using the brush tool add or adjust any masks within the appropriate layer.

    Args:
        before_image (ndarray): self explanatory
        image_name (str): self explanatory
        mask (ndarray): self explanatory

    Returns:
        ndarray: stacked masks
    """

    nuclei = mask.copy()
    
    viewer = napari.Viewer()

    # Add multi-channel image with channel_axis=0 so napari treats channels correctly
    viewer.add_image(before_image, name='before_image', channel_axis=0)

    # Add mask labels (2D), which will overlay on all channels
    viewer.add_labels(nuclei, name='nuclei')

    napari.run()

    np.save(f'{output_folder}{image_name}_mask.npy',
            np.stack([nuclei]))
    logger.info(
        f'Processed {image_name}. Mask saved to {output_folder}{image_name}')

    return np.stack([nuclei])


def stack_channels(name, masks_filtered, cells_filtered_stack):
    masks_filtered[name] = cells_filtered_stack

# ----------------Initialise file list----------------
# read in numpy masks
nuc_masks = np.load(f'{mask_folder}cellpose_nucmasks.npy', allow_pickle=True)

# Check the shape and type
print(f'nuc_masks type: {type(nuc_masks)}, length: {len(nuc_masks)}')
print(f'nuc_masks[0] shape: {nuc_masks[0].shape}')

# clean filenames
file_list = [filename for filename in os.listdir(
    image_folder) if 'npy' in filename]

images = {filename.replace('.npy', ''): np.load(
    f'{image_folder}{filename}') for filename in file_list}

image_names = sorted(images.keys())
mask_stacks = {
    name: np.expand_dims(nuc_masks[i], axis=0)
    for i, name in enumerate(image_names)
}

# make new dictionary to check for cell size and borders
image_keys = images.keys()
image_values = zip(images.values(), mask_stacks.values())
saturation_check = dict(zip(image_keys, image_values))

# ----------------filtering small cells and/or near border----------
masks_filtered = {}
images_cropped = {}  # Optional: to store cropped images if needed downstream

logger.info('Filtering: removing small & border-touching cells; cropping all channels to match mask dimensions')

for name, image in saturation_check.items():
    labels_filtered = []

    # Extract the mask
    mask = image[1][0, :, :]

    # Extract full image stack (shape: [channels, height, width])
    img_stack = image[0]

    # Crop all channels to match the mask shape if needed
    min_height = min(mask.shape[0], img_stack.shape[1])
    min_width = min(mask.shape[1], img_stack.shape[2])

    mask = mask[:min_height, :min_width]
    img_stack_cropped = img_stack[:, :min_height, :min_width]

    # Store cropped image if needed later (e.g., for visualization, export)
    images_cropped[name] = img_stack_cropped

    # Filter out small objects
    unique_labels = np.unique(mask)
    for label in unique_labels[1:]:  # skip background
        pixel_count = np.count_nonzero(mask == label)
        if pixel_count > 20:
            labels_filtered.append(label)

    # Keep only approved labels
    cells_filtered = np.where(np.isin(mask, labels_filtered), mask, 0)

    # Remove cells near border
    cells_filtered = clear_border(cells_filtered, buffer_size=10)

    # Stack into expected format (1, H, W)
    cells_filtered_stack = np.stack((cells_filtered.copy(),))
    stack_channels(name, masks_filtered, cells_filtered_stack)


# ---------------Manually filter masks---------------
# Manually validate cellpose segmentation.
already_filtered_masks = [filename.replace('_mask.npy', '') for filename in os.listdir(
    f'{output_folder}') if '_mask.npy' in filename]

unval_images = dict([(key, val) for key, val in images.items()
                    if key not in already_filtered_masks])

filtered_masks = {}
for image_name, image_stack in unval_images.items():
    image_stack
    mask_stack = masks_filtered[image_name].copy()
    filtered_masks[image_name] = filter_masks(image_stack, image_name, mask_stack)


# ---------------Process filtered masks---------------
# TODO make below lines a new script
# To reload previous masks for per-cell extraction
filtered_masks = {masks.replace('_mask.npy', ''): np.load(
    f'{output_folder}{masks}', allow_pickle=True) for masks in os.listdir(f'{output_folder}') if '_mask.npy' in masks}


# # DID NOT RUN THIS SECTION & BEYOND
# logger.info('removing nuclei from cell masks')
# cytoplasm_masks = {}
# for name, img in filtered_masks.items():
#     name
#     cell_mask = img[0, :, :]
#     nuc_mask = img[1, :, :]
#     # make binary masks
#     cell_mask_binary = np.where(cell_mask, 1, 0)
#     nuc_mask_binary = np.where(nuc_mask, 1, 0)
#     single_cytoplasm_masks = []
#     # need this elif in case images have no masks
#     if len(np.unique(cell_mask).tolist()) > 1:
#         for num in np.unique(cell_mask).tolist()[1:]:
#             num
#             # subtract whole nuclear mask per cell
#             cytoplasm = np.where(cell_mask == num, cell_mask_binary, 0)
#             cytoplasm_minus_nuc = np.where(cytoplasm == nuc_mask_binary, 0, cytoplasm)
#             if np.count_nonzero(cytoplasm) != np.count_nonzero(cytoplasm_minus_nuc):
#                 # re-assign label
#                 cytoplasm_num = np.where(cytoplasm_minus_nuc, num, 0)
#                 single_cytoplasm_masks.append(cytoplasm_num)
#             else:
#                 single_cytoplasm_masks.append(
#                     np.zeros(np.shape(cell_mask)).astype(int))
#     else:
#         single_cytoplasm_masks.append(
#         np.zeros(np.shape(cell_mask)).astype(int))
#     # add cells together and update dict
#     summary_array = sum(single_cytoplasm_masks)
#     cytoplasm_masks[name] = summary_array
# logger.info('nuclei removed')

# # ---------------save arrays---------------
# np.save(f'{output_folder}cytoplasm_masks.npy', cytoplasm_masks)
# logger.info('mask arrays saved')
