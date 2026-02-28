"""Detect puncta, measure features, register images, assign puncta to nuclei, visualize data
"""
#%%
# STEP 1: IMPORT PACKAGES & DEFINE FUNCTIONS

import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
import functools
import cv2
from skimage import measure, segmentation, morphology
from scipy.stats import skewtest
from skimage.morphology import remove_small_objects
from statannotations.Annotator import Annotator
from loguru import logger
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.spatial import ConvexHull

plt.rcParams.update({'font.size': 14})

# Get the current working directory
current_directory = os.getcwd()
print("Current directory:", current_directory)
# Change the current working directory
new_directory = '/Users/ronanoconnell/Library/CloudStorage/OneDrive-BaylorCollegeofMedicine/BFL/IMAGE PROCESSING SCRIPTS/20250425_63x_FISH_images-main/' 
os.chdir(new_directory)
# Verify the change
current_directory = os.getcwd()
print("Current directory:", current_directory)

input_folder = 'python_results/initial_cleanup/'
mask_folder = 'python_results/napari_masking/'
output_folder = 'python_results/summary_calculations/'
plotting_folder = 'python_results/plotting/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

if not os.path.exists(plotting_folder):
    os.mkdir(plotting_folder)

def feature_extractor(mask, properties=False):

    if not properties:
        properties = ['area', 'eccentricity', 'label', 'major_axis_length', 'minor_axis_length', 'perimeter', 'coords']

    return pd.DataFrame(skimage.measure.regionprops_table(mask, properties=properties))

# ----------------Initialise file list----------------
file_list = [filename for filename in os.listdir(
    input_folder) if 'npy' in filename]

images = {filename.replace('.npy', ''): np.load(
    f'{input_folder}{filename}') for filename in file_list}

# make dictionary of masks (first day only)
masks = {masks.replace('_mask.npy', ''): np.load(
    f'{mask_folder}{masks}', allow_pickle=True) for masks in os.listdir(f'{mask_folder}') if '_mask.npy' in masks}

# expand mask dictionary to all images
all_masks = {}
for image_name, image in images.items():
    for mask_name, mask in masks.items():
        mask_well = mask_name.split('_')
        mask_well = f'{mask_well[1]}_{mask_well[2]}'
        if mask_well in image_name:
            all_masks[image_name] = mask

# make dictionary from images and masks array
image_mask_dict = {
    key: np.stack([images[key][0, :, :], images[key][1, :, :], images[key][2, :, :], all_masks[key][0, :, :]])
    for key in all_masks
}

#Test Visualize images
image_name = "20250409_B6_63x-01"
image_name = "20250410_B6_63x-01"
plt.imshow(image_mask_dict[image_name][0])
plt.imshow(image_mask_dict[image_name][1])
plt.imshow(image_mask_dict[image_name][2])
plt.imshow(image_mask_dict[image_name][3])


#%% 
# STEP 2: Collect Features on a per IMAGE basis
# ----------------collect feature information----------------
#COPILOT's VERSION FOR ALL CHANNELS
# find cell outlines for later plotting
logger.info('collecting feature info')

threshold = 6 # 6x std. dev. above mean
 
feature_information_list = []
for name, image in image_mask_dict.items():
    image_properties_list = []
    for channel in range(3):
        
        binary = np.where(image[channel, :, :] > 5000, 1, 0)
        puncta_masks = measure.label(binary)
        puncta_masks = remove_small_objects(puncta_masks, 10)  # Consider changing this to help pick up weak signals
        image_properties = feature_extractor(puncta_masks)

        # make list for cov and skew, add as columns to properties
        granule_cov_list = []
        granule_skew_list = []
        granule_intensity_list = []
        for granule_num in np.unique(puncta_masks)[1:]:
            granule_num
            granule = np.where(puncta_masks == granule_num, image[channel, :, :], 0)
            granule = granule[granule != 0]
            granule_cov = np.std(granule) / np.mean(granule)
            granule_cov_list.append(granule_cov)
            res = skewtest(granule)
            granule_skew = res.statistic
            granule_skew_list.append(granule_skew)
            granule_intensity_list.append(np.mean(granule))
        image_properties['granule_cov'] = granule_cov_list
        image_properties['granule_skew'] = granule_skew_list
        image_properties['granule_intensity'] = granule_intensity_list

        if len(image_properties) < 1: #adds zeros to dataframe
            image_properties.loc[len(image_properties)] = 0

        image_properties['channel'] = channel
        image_properties_list.append(image_properties)

    properties = pd.concat(image_properties_list)
    properties['image_name'] = name

    feature_information_list.append(properties)
        
feature_information = pd.concat(feature_information_list)
logger.info('completed feature collection')

# save data for plotting coords
feature_information.to_csv(f'{output_folder}puncta_detection_feature_info.csv')


# # UNNECESSARY NOW... We are now creating a dataframe of nuclei coordinates and numbers to cross reference against our feature_information dataframe in order to assign puncta to nuclei (Refactored by ChatGPT)
# from skimage.measure import label
# from skimage.morphology import remove_small_objects

# nuclear_information_list = []
# channel = 3  # Nuclear channel

# for name, image in image_mask_dict.items():
#     nuclear_properties_list = []

#     nuclei_labels = np.unique(image[channel])
#     for nucleus_label in nuclei_labels:
#         if nucleus_label == 0:
#             continue  # skip background

#         # Create binary mask for current nucleus
#         binary_mask = (image[channel] == nucleus_label).astype(int)
#         labeled_nucleus = label(binary_mask)

#         # Filter out small components if needed
#         if np.max(labeled_nucleus) > 1:
#             labeled_nucleus = remove_small_objects(labeled_nucleus, min_size=10)

#         # Extract properties
#         props = feature_extractor(labeled_nucleus)

#         if len(props) == 0:
#             continue  # skip if nothing found

#         # Assign a unique nucleus number for each region
#         for i, row in props.iterrows():
#             row = row.copy()  # avoid SettingWithCopyWarning
#             row['nucleus_number'] = nucleus_label + (i / 10)
#             row['channel'] = channel
#             row['image_name'] = name
#             nuclear_properties_list.append(row)

#     if nuclear_properties_list:
#         properties_df = pd.DataFrame(nuclear_properties_list)
#         nuclear_information_list.append(properties_df)

# # Combine into single DataFrame
# nuclear_information = pd.concat(nuclear_information_list, ignore_index=True)
# logger.info('Completed nuclear feature collection')

# # save data for plotting coords
# nuclear_information.to_csv(f'{output_folder}nuclear_detection_feature_info.csv')


#%% 
# STEP 3: 20250430 - Image Registration using puncta from across channels and then saving registered puncta coordinates into Feature_Information as a new column named "registered_coords"

from collections import defaultdict
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from skimage.draw import disk

# Helper: extract well from image name (e.g., "WellA01_R1" → "WellA01")
def get_well_name(image_name):
    return '_'.join(image_name.split('_')[1:])

def get_well_date(image_name):
    return '_'.join(image_name.split('_')[0:1])

def coords_to_mask(coords, shape, radius=2):
    mask = np.zeros(shape, dtype=np.float32)
    for y, x in coords.astype(int):
        rr, cc = disk((y, x), radius=radius, shape=shape)
        mask[rr, cc] = 1.0
    return mask


# Group images by well
well_groups = defaultdict(list)
for image_name in image_mask_dict.keys():
    well = get_well_name(image_name)
    well_groups[well].append(image_name)

top_puncta_dict = {}  # key: image_name, value: ndarray of coordinates
for image_name in image_mask_dict.keys():

    df = feature_information[(feature_information['image_name'] == image_name) & (feature_information['area'] > 10)]

    # unique_val = np.unique(df['cell_number'])
    # may need to pick the largest puncta if there are overlaps
    # top_puncta = []
    # for cell_num in unique_val:
    # # Select all puncta with area > 5
    #     top_cell_puncta = df[(df['area'] > 5) & (df['cell_number'] == cell_num)]
    #     if len(top_cell_puncta) > 1:
    #         top_cell_puncta = top_cell_puncta[top_cell_puncta['area'] == top_cell_puncta['area'].max()]
    #     top_puncta.append(top_cell_puncta)
    # top_puncta = pd.concat(top_puncta, ignore_index=True)
        
    # Extract centroids as array
    centroids = np.vstack(df['coords'].values)
    top_puncta_dict[image_name] = centroids

# Example: Register images in each well to the first image in that well
registered_images = {}
## Optional: initialize the new column to NaN for all rows
# feature_information['registered_coords'] = np.nan

for well, image_list in well_groups.items():
    # Sort image list alphabetically to get the earliest date (first alphabetically)
    image_list_sorted = sorted(image_list)
    
    # Use the first image in the sorted list as the reference image
    ref_image_name = image_list_sorted[0]
    ref_img = image_mask_dict[ref_image_name]

    threshold = 10000
    # Define channels in terms of R, G, B
    rref = np.where(ref_img[0] > threshold, ref_img[0], 0)
    gref = np.where(ref_img[1] > threshold, ref_img[1], 0)
    bref = np.where(ref_img[2] > threshold, ref_img[2], 0)
    # Stack into RGB image
    rgb_ref = np.stack([rref, gref, bref], axis=-1)

    ref_coords = top_puncta_dict[ref_image_name]
    ref_mask = coords_to_mask(ref_coords, ref_img.shape[1:]) # Generates mask based on puncta above

    for image_name in image_list_sorted:
        coords = top_puncta_dict[image_name]
        img = image_mask_dict[image_name]
        mask = coords_to_mask(coords, img.shape[1:])
        
        # --- Compute shift between reference and current image ---
        shift_vector, _, _ = phase_cross_correlation(ref_mask, mask, upsample_factor=10)
        
        # --- Apply shift to image ---
        shifted_img = shift(img, shift=(0, shift_vector[0], shift_vector[1]))  # (C, H, W)
        registered_images[image_name] = shifted_img
        
        # --- Apply shift to coordinates in feature_information ---
        shift_y, shift_x = shift_vector
        mask_rows = feature_information['image_name'] == image_name

        # Create registered_coords column with shifted values
        feature_information.loc[mask_rows, 'registered_coords'] = (
            feature_information.loc[mask_rows, 'coords']
            .apply(lambda coord: np.array(coord) + np.array([shift_y, shift_x]))
        )

        # # DISPLAY (only run to test things are working)
        # # Choose channels to plot as RGB, e.g., channels 0, 1, and 2
        # rimg = np.where(img[0] > threshold, img[0], 0)
        # gimg = np.where(img[1] > threshold, img[1], 0)
        # bimg = np.where(img[2] > threshold, img[2], 0)
        # rshift = np.where(shifted_img[0] > threshold, shifted_img[0], 0)
        # gshift = np.where(shifted_img[1] > threshold, shifted_img[1], 0)
        # bshift = np.where(shifted_img[2] > threshold, shifted_img[2], 0)
        # # Stack into RGB image
        # rgb_img = np.stack([rimg, gimg, bimg], axis=-1)
        # rgb_shift = np.stack([rshift, gshift, bshift], axis=-1)

        # plt.figure(figsize=(9, 3))

        # plt.subplot(1, 3, 1)
        # plt.imshow(rgb_ref)
        # plt.title("Reference: "+image_name, fontsize = 12)
        # plt.axis('off')

        # plt.subplot(1, 3, 2)
        # plt.imshow(rgb_img)
        # plt.title("Original: "+image_name, fontsize = 12)
        # plt.axis('off')

        # plt.subplot(1, 3, 3)
        # plt.imshow(rgb_shift)
        # plt.title("Shifted: "+image_name, fontsize = 12)
        # plt.axis('off')

        # plt.tight_layout()
        # plt.show()


#%%
# STEP 4: Collect Features on a per IMAGE basis from REGISTERED IMAGES
# ----------------collect feature information----------------
#COPILOT's VERSION FOR ALL CHANNELS
# find cell outlines for later plotting
logger.info('collecting registered feature info')

threshold = 3 # 3x std. dev. above mean

feature_information_list = []
for name, image in registered_images.items():

    nuclei = image_mask_dict[name][3, :, :] # take original mask channel from each image

    cell_binary_mask = np.where(nuclei != 0, 1, 0)
    contours = measure.find_contours(cell_binary_mask, 0.8)
    contour = [x for x in contours if len(x) >= 100]
    unique_val, counts = np.unique(nuclei, return_counts=True)
    
    # loop to extract params from cells
    for num in unique_val[1:]:
        num
        cell_properties_list = []
        for channel in range(3):

            # Grab nucleus (original)
            cell_mask = np.where(nuclei == num, num, 0)
            cell_contour = feature_extractor(cell_mask)
            cell_std = np.std(cell[cell != 0])
            cell_mean = np.mean(cell[cell != 0])
            
            # Grab puncta (registered)
            cell = np.where(nuclei == num, image[channel, :, :], 0)
            binary = (cell > min((cell_mean + (cell_std * threshold)), 65000)).astype(int)  # takes the minimum of either saturated or std dev above mean of cell
            puncta_masks = measure.label(binary)
            puncta_masks = remove_small_objects(puncta_masks, 5)  # Consider changing this to help pick up weak signals
            cell_properties = feature_extractor(puncta_masks)

            # make list for cov and skew, add as columns to properties
            granule_cov_list = []
            granule_skew_list = []
            granule_intensity_list = []
            for granule_num in np.unique(puncta_masks)[1:]:
                granule_num
                granule = np.where(puncta_masks == granule_num, image[channel, :, :], 0)
                granule = granule[granule != 0]
                granule_cov = np.std(granule) / np.mean(granule)
                granule_cov_list.append(granule_cov)
                res = skewtest(granule)
                granule_skew = res.statistic
                granule_skew_list.append(granule_skew)
                granule_intensity_list.append(np.mean(granule))
            cell_properties['granule_cov'] = granule_cov_list
            cell_properties['granule_skew'] = granule_skew_list
            cell_properties['granule_intensity'] = granule_intensity_list

            if len(cell_properties) < 1: #adds zeros to dataframe
                cell_properties.loc[len(cell_properties)] = 0

            cell_properties['channel'] = channel
            cell_properties_list.append(cell_properties)

        properties = pd.concat(cell_properties_list)
        properties['image_name'] = name
        properties['cell_number'] = num
        properties['cell_size'] = np.size(cell[cell != 0])
        properties['cell_intensity_mean'] = cell_mean
        properties['cell_contour'] = [cell_contour['coords'][0]] * len(properties)

        # add cell outlines to coords
        properties['cell_coords'] = [contour] * len(properties)

        feature_information_list.append(properties)
        
registered_feature_information = pd.concat(feature_information_list)
logger.info('completed feature collection')

# save data for plotting coords
registered_feature_information.to_csv(f'{output_folder}registered_feature_info.csv')


#%%
# STEP 5 v1: PLOT THINGS? -- IN PROGRESS

# Have one data frame with registered puncta coordinates and nuclear coordinates (registered_feature_information)

# 5a: Plot max puncta and assign to nucleus (CURRENTLY ONLY WORKS FOR ONE COLOR PER CHANNEL)

from skimage.measure import find_contours

def autopct_format(pct):
    return f'{int(round(pct))}%' if pct >= 1 else ''  # Hide anything <1%


for name, image in registered_images.items():
    name
    unique_val, counts = np.unique(image[3, :, :], return_counts=True)
    nuclei = image_mask_dict[name][3, :, :] # take original mask channel from each image

    #plot nuclei
    dataframe = registered_feature_information[registered_feature_information['image_name'] == name]
    
    max_puncta = []
    for cell_num in unique_val:
        nucleus = dataframe[dataframe['cell_number'] == cell_num]
        if nucleus['area'].max() == 0:
            max_puncta_temp = nucleus.iloc[[0]]  # first row as a DataFrame
        else:
            max_puncta_temp = nucleus[nucleus['area'] == nucleus['area'].max()]
        max_puncta.append(max_puncta_temp)
    all_max_puncta = pd.concat(max_puncta, ignore_index=True)
    #Assign nuclei with missing puncta to channel 3 
    all_max_puncta.loc[all_max_puncta['area'] < 10, 'channel'] = 3

    # Define expected channels
    expected_channels = [0, 1, 2, 3]
    # Count actual occurrences of each channel
    channel_counts = all_max_puncta['channel'].value_counts().sort_index()
    # Reindex to include all channels and fill missing with 0
    channel_counts = channel_counts.reindex(expected_channels, fill_value=0)
    # Convert to DataFrame
    puncta_counts = channel_counts.reset_index()
    puncta_counts.columns = ['channel', 'count']

    color_map = {0: '#D30C7B', 1: '#62E2E9', 2: '#F9DC5C', 3: 'Gray'}
    
    fig, ax1 = plt.subplots(figsize = (5, 5))
    for number in all_max_puncta['cell_number'].unique():
        number
        cell_area = all_max_puncta[all_max_puncta['cell_number'] == number]['cell_size'].iloc[0]
        if cell_area < 10:
            continue

        # if puncta_area <= 10:
        #     # make gray default
        #     channel = 3
        # else:
        channel = all_max_puncta[all_max_puncta['cell_number'] == number]['channel'].iloc[0]

        # grab cell coords
        cell_line = all_max_puncta[all_max_puncta['cell_number'] == number]['cell_coords'].iloc[0]
        # plot cell

        # Generate binary mask of current nucleus
        binary_mask = (nuclei == number)

        # Find contours
        contours = find_contours(binary_mask, 0.5)

        # Use the longest contour (usually outer boundary)
        if contours:
            contour = max(contours, key=lambda x: len(x))
            ax1.fill(contour[:, 1], contour[:, 0], color=color_map[channel])
            ax1.plot(contour[:, 1], contour[:, 0], color='k', linewidth=1)

    # # Create scale bar
    # scalebar = ScaleBar(0.0779907, 'um', location = 'lower left', pad = 0.3, sep = 2, box_alpha = 0, color='k', length_fraction=0.3)
    # ax1.add_artist(scalebar)
    # ax1.set_xlim(0, 1024)
    # ax1.set_ylim(0, 1024)

    # # Add inset pie chart for channel distribution
    # inset_ax = fig.add_axes([0.0, 0.8, 0.15, 0.15])  # [x0, y0, width, height]

    # # Prepare pie data
    # counts = puncta_counts['count']
    # labels = (100 * puncta_counts['count'] / puncta_counts['count'].sum()).round(0).astype(int)
    # colors = [color_map[ch] for ch in puncta_counts['channel']]

    # # Plot pie chart
    # inset_ax.pie(counts, colors=colors, startangle=90, 
    #             counterclock=False, labeldistance=0.4, textprops={'fontsize': 6}, autopct=autopct_format)

    # # title and save
    # fig.suptitle(name, y=0.925)
    # fig.tight_layout()
    # fig.savefig(f'{plotting_folder}{name}_dom_punct.png', bbox_inches='tight',pad_inches = 0.1, dpi = 600)
    # plt.close()


#%% 
# STEP 6: ASSIGN FULL BARCODES TO NUCLEI 
# Group by well name, nucleus number, and date to pull out the info required to map nucleus number to a barcode/channel sequence
    # What happens if there is a spot missed? Assign NaN as channel?

from collections import defaultdict
import numpy as np
import pandas as pd

def get_well_name(image_name):
    return '_'.join(image_name.split('_')[1:])

def get_well_date(image_name):
    return '_'.join(image_name.split('_')[0:1])

# Group images by well
well_groups = defaultdict(list)
for image_name in image_mask_dict.keys():
    well = get_well_name(image_name)
    well_groups[well].append(image_name)

# List to accumulate results across wells
all_barcode_assignments = []

for well, image_list in well_groups.items():
    # Sort image list alphabetically to get the earliest date
    image_list_sorted = sorted(image_list)

    for date in image_list_sorted:
        dataframe = registered_feature_information[registered_feature_information['image_name'] == date]
        unique_val = np.unique(dataframe['cell_number'])
        
        # Optional: only use nuclei mask if needed later
        # nuclei = image_mask_dict[date][3, :, :]  

        max_puncta = []
        for cell_num in unique_val:
            nucleus = dataframe[dataframe['cell_number'] == cell_num][['image_name', 'cell_number', 'channel', 'area', 'coords']]

            if nucleus['area'].max() == 0:
                max_puncta_temp = nucleus.iloc[[0]]  # first row as DataFrame
            else:
                max_puncta_temp = nucleus[nucleus['area'] == nucleus['area'].max()]
            max_puncta.append(max_puncta_temp)

        all_max_puncta = pd.concat(max_puncta, ignore_index=True)
        all_max_puncta.loc[all_max_puncta['area'] < 10, 'channel'] = 3  # assign fallback channel
        all_barcode_assignments.append(all_max_puncta)

# Combine across all wells and images
barcode_assignments = pd.concat(all_barcode_assignments, ignore_index=True)
# Add well and date columns from image_name
barcode_assignments['well'] = barcode_assignments['image_name'].apply(get_well_name)
barcode_assignments['date'] = barcode_assignments['image_name'].apply(get_well_date)

# Reformat dataframe so that it looks like: well - cell - date 1 channel - date 2 channel - date 3 channel etc. 

# Make sure 'cell_number' is included
df_wide = barcode_assignments.pivot_table(
    index=['well', 'cell_number'],
    columns='date',
    values='channel',
    aggfunc='first'  # or 'max' if multiple values per date/cell/well
).reset_index()

# Count the occurrences of each unique row excluding 'cell_number'
row_counts = df_wide.drop(columns='cell_number').groupby(list(df_wide.columns.difference(['cell_number']))).size().reset_index(name='count')
row_counts_sorted = row_counts.sort_values(by='count', ascending=False)

# Test
condition = 'B5_63x-01'
a = row_counts_sorted[row_counts_sorted['well'] == condition]
print(a)



