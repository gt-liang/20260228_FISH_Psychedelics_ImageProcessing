"""Collects images from the raw_data folder and organizes them into an np stack for later analyses
""" 
#%%
import os
import numpy as np
from aicsimageio.writers import OmeTiffWriter
from aicsimageio import AICSImage
from loguru import logger

logger.info('Import ok')

# Get the current working directory
current_directory = os.getcwd()
print("Current directory:", current_directory)
# Change the current working directory
new_directory = '/Users/ronanoconnell/Library/CloudStorage/OneDrive-BaylorCollegeofMedicine/BFL/IMAGE PROCESSING SCRIPTS/20250729_20x_FISH_images - in progress/'
os.chdir(new_directory)
# Verify the change
current_directory = os.getcwd()
print("Current directory:", current_directory)

input_path = 'raw_data'
output_folder = 'python_results/initial_cleanup/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

def czi_converter(image_name, input_folder, output_folder, tiff=False, array=True, mip=True, trim=False):
    """Stack images from nested .czi files and save for subsequent processing

    Args:
        image_name (str): image name (usually iterated from list)
        input_folder (str): filepath
        output_folder (str): filepath
        tiff (bool, optional): Save tiff. Defaults to False.
        array (bool, optional): Save np array. Defaults to True.
        compress (bool, optional): Compress if image is too large. Defaults to False.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # import image
    image = AICSImage(f'{input_folder}.czi').get_image_data("CZYX")
    ch_list = list(range(0, np.shape(image)[0]))

    # for maximum projections
    if mip == True:
        image = np.asarray([np.max(image[ch, :, :, :], axis=0) for ch in ch_list])

    # for if image sizes are irregular and want to trim to basic shape
    if trim == True:
        N = 1000
        image = np.asarray([image[ch, -N:, :N] for ch in ch_list])

    # for saving image to TIFF file
    if tiff == True:
        OmeTiffWriter.save(
            image, f'{output_folder}{image_name}.tif', dim_order='CYX')

    # for saving image to np array (for analysis)
    if array == True:
        np.save(f'{output_folder}{image_name}.npy', image)


# ---------------Initalize file_list---------------
# find directories of interest
file_list = [filename for filename in os.listdir(input_path) if 'czi' in filename]

do_not_quantitate = [] 

# ---------------Collect image names and load---------------
image_names = []
for filename in file_list:
    if all(word not in filename for word in do_not_quantitate):
        filename = filename.split('.czi')[0]
        image_names.append(filename)

# remove duplicates
image_names = list(dict.fromkeys(image_names))

# collect images
for name in image_names:
    czi_converter(name, input_folder=f'{input_path}/{name}',
                output_folder=f'{output_folder}')

logger.info('initial cleanup complete')

# %%
