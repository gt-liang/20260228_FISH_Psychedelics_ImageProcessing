"""Detect puncta, measure features, visualize data
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
from pathlib import Path

plt.rcParams.update({'font.size': 14})

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

# Check shapes before stacking (optional debug print)
for key in all_masks:
    print(f"Image '{key}' channel shapes:")
    for ch in range(5):
        print(f"  Channel {ch+1} shape: {images[key][ch].shape}")
    print(f"  Mask shape: {all_masks[key].shape}")

# make dictionary from images and masks array, squeezing mask to 2D first
image_mask_dict = {}

for key in all_masks:
    img = images[key]
    mask = np.squeeze(all_masks[key])  # remove all singleton dims, shape (H, W)

    # Extract all channels as a list of 2D arrays
    channels_list = [img[ch, :, :] for ch in range(img.shape[0])]

    # Append the mask at the end
    channels_list.append(mask)

    # Stack all channels + mask along a new first dimension
    combined_stack = np.stack(channels_list, axis=0)

    image_mask_dict[key] = combined_stack


#Test Visualize images
image_name = "20250718_20x_D4_Hyb1-01"
plt.imshow(image_mask_dict[image_name][0])
plt.imshow(image_mask_dict[image_name][1])
plt.imshow(image_mask_dict[image_name][2])
plt.imshow(image_mask_dict[image_name][3])
plt.imshow(image_mask_dict[image_name][4])


#%% 
# Collect Features
# ----------------collect feature information----------------
import time
import warnings
import numpy as np
from skimage.morphology import remove_small_objects
from skimage import measure
from scipy.stats import skewtest
import random
from skimage.measure import find_contours


warnings.filterwarnings(
    "ignore",
    message="Only one label was provided to `remove_small_objects`. Did you mean to use a boolean array?",
    category=UserWarning
)

logger.info('collecting feature info')
threshold = 3  # 6x std. dev. above mean
feature_rows = []
N = len(image_mask_dict)

for name, image in list(image_mask_dict.items())[:N]: #loop through N images in the image_mask_dict
    # sample_image = list(image_mask_dict.items())[0]  # first image only
    # name, image = sample_image
    # print(f'\nProcessing sample image: {name}')
    # t0 = time.time()

    print(f'\nProcessing image: {name}')
    t0 = time.time()

    max_channel = image.shape[0]
    mask_ch = max_channel - 1
    mask = image[mask_ch]
    binary_mask = mask != 0

    # unique_val = np.unique(mask)
    # if len(unique_val) <= 1:
    #     print("  No cells found, skipping image.")
    # else:
    #     # 🔹 Sample up to 20 random nonzero cells (skip 0)
    #     sampled_cells = random.sample(list(unique_val[1:]), min(100, len(unique_val) - 1))

    # For looping
    unique_val = np.unique(mask)
    if len(unique_val) <= 1:
        print("  No cells found, skipping image.")
        continue

    for i, num in enumerate(unique_val[1:]):  # skip background (for loop)
    #for i,num in enumerate(sampled_cells): # for sub-sample
        if i % 100 == 0:
            print(f'  Processed {i} cells')

        cell_mask = (mask == num)
        cell_size = np.count_nonzero(cell_mask)
        if cell_size == 0:
            continue

        # Extract contour from binary cell mask
        contours = find_contours(cell_mask.astype(float), 0.5)

        # Choose the longest contour (sometimes multiple per region)
        if contours:
            contour = max(contours, key=len)
            cell_contour = contour.tolist()
        else:
            cell_contour = []  # fallback if none found

        #for channel in range(mask_ch):  ## Option to loop through ALL image channels except mask
        for channel in [0]: # Option for single channel (GFP here)
            channel_data = image[channel]
            cell_signal = channel_data * cell_mask
            cell_vals = cell_signal[cell_signal > 0]
            if len(cell_vals) == 0:
                continue

            mean = np.mean(cell_vals)
            std = np.std(cell_vals)
            cutoff = min(mean + threshold * std, 65000)

            binary = (cell_signal > cutoff).astype(np.uint8)
            labeled = measure.label(binary)
            labeled = remove_small_objects(labeled, 1)

            props = measure.regionprops(labeled, intensity_image=channel_data)

            if not props:
                # Add dummy row if no granules found
                feature_rows.append({
                    'image_name': name,
                    'cell_number': num,
                    'channel': channel,
                    'cell_size': cell_size,
                    'cell_intensity_mean': mean,
                    #'granule_cov': 0,
                    #'granule_skew': 0,
                    'granule_area': 0,
                    'granule_intensity': 0,
                    'granule_coords': 0,
                    'cell_contour': cell_contour
                })
            else:
                for p in props:
                    granule_vals = p.intensity_image[p.image]
                    #cov = np.std(granule_vals) / np.mean(granule_vals) if np.mean(granule_vals) else 0
                    #skew = skewtest(granule_vals).statistic if len(granule_vals) > 2 else 0
                    granule_size = p.area
                    granule_coords = p.coords.tolist()  # pixel locations
                    feature_rows.append({
                        'image_name': name,
                        'cell_number': num,
                        'channel': channel,
                        'cell_size': cell_size,
                        'cell_intensity_mean': mean,
                        #'granule_cov': cov,
                        #'granule_skew': skew,
                        'granule_area': granule_size,
                        'granule_intensity': np.mean(granule_vals),
                        'granule_coords': granule_coords,
                        'cell_contour': cell_contour
                    })

    print(f'  Finished {name} in {time.time() - t0:.2f} sec')

# Create final DataFrame once
# 🔹 Save results
feature_information = pd.DataFrame(feature_rows)
feature_information.to_csv(f'{output_folder}puncta_detection_feature_info.csv', index=False)
logger.info('Completed FASTMODE feature collection')





## Using dataframe (feature_information) I want to visualize the puncta
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import os

# Assign the maximum puncta in a channel to a cell
# For each cell_number, get the index of the row with the max granule_area
idx = feature_information.groupby(['image_name', 'cell_number'])['granule_area'].idxmax()

# Select those rows to form the new dataframe
feature_information_max_puncta = feature_information.loc[idx].reset_index(drop=True)

# Remove rows containing cells that are above a certain size
cell_thresh = 600
feature_information_max_puncta = feature_information_max_puncta[feature_information_max_puncta['cell_size'] < cell_thresh]

# Choose the image you want to visualize
target_image = "20250718_20x_C6_Hyb1-01"
df = feature_information_max_puncta[feature_information_max_puncta["image_name"] == target_image]

puncta_thresh = 3
num_with = len(df[df['granule_area'] > puncta_thresh])
num_without = len(df) - num_with

# Create figure and axis properly
fig = plt.figure(figsize=(10, 10), facecolor='black')
ax = fig.gca()
ax.set_title(f"Cells and Puncta in Image: {target_image}", color='black', fontsize=14, pad=20)
plt.axis('on')

# Plot cells and puncta
for _, row in df.iterrows():
    cell_contour = np.array(row["cell_contour"])
    if cell_contour.ndim == 2 and cell_contour.shape[1] == 2:
        ax.plot(cell_contour[:, 1], cell_contour[:, 0], color='black', linewidth=0.5)

    granule_coords = np.array(row["granule_coords"])
    if granule_coords.ndim == 2 and granule_coords.shape[1] == 2 and len(granule_coords) > puncta_thresh:
        ax.scatter(granule_coords[:, 1], granule_coords[:, 0], color='magenta', s=0.5)

# Add pie chart inset
pie_ax = inset_axes(ax, width="20%", height="20%", loc='upper right', borderpad=2)
pie_ax.pie(
    [num_with, num_without],
    colors=['magenta', 'gray'],
    startangle=90,
    autopct=lambda pct: f"{'With' if pct > 50 else 'Without'}\n{pct:.0f}%",
    textprops={'color': 'white', 'fontsize': 8}
)
pie_ax.set_aspect('equal')

plt.tight_layout()

# Set up output path and create directory if needed
output_path = Path(os.getcwd()) / 'python_results' / 'plotting' / f"cells_puncta_{target_image}.svg"
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save figure
fig.savefig(output_path, format='svg', transparent=True)
plt.show()




































#%%
# # PLOT INDIVIDUAL PUNCTA FOR EACH IMAGE
# # Define puncta size thresholds
# Cy5_thresh = 18
# TxRed_thresh = 18
# gfp_thresh = 18

# # plot proofs
# for name, image in image_mask_dict.items():
#     name
#     unique_val, counts = np.unique(image[3, :, :], return_counts=True)

#     # extract coords
#     cell = np.where(image[3, :, :] != 0, image[0, :, :], 0)
#     image_df = feature_information[(feature_information['image_name'] == name)]
#     if len(image_df) > 0:
#         cell_contour = image_df['cell_coords'].iloc[0]
#         coord_list = np.array(image_df.coords)

#         ch1 = image_df[image_df['channel'] == 0]
#         ch1_area = image_df['area'][image_df['channel'] == 0]
#         ch1_coords = np.array(ch1.coords)
#         ch1_coords = ch1_coords[ch1_area > Cy5_thresh]

#         ch2 = image_df[image_df['channel'] == 1]
#         ch2_area = image_df['area'][image_df['channel'] == 1]
#         ch2_coords = np.array(ch2.coords)
#         ch2_coords = ch2_coords[ch2_area > TxRed_thresh]

#         ch3 = image_df[image_df['channel'] == 2]
#         ch3_area = image_df['area'][image_df['channel'] == 2]
#         ch3_coords = np.array(ch3.coords)
#         ch3_coords = ch3_coords[ch3_area > gfp_thresh]

#         # plot
#         fig, (ax1) = plt.subplots(1,figsize = (5, 5)) #add ax1 for plotting ax1
#         # ax1.imshow(image[0,:,:], alpha=0.60)
#         # ax1.imshow(image[1,:,:], cmap=plt.cm.gray)
#         # ax2.imshow(cell)
#         for cell_line in cell_contour:
#             ax1.fill(cell_line[:, 1], cell_line[:, 0], linewidth=0.5, c='k')
#         if len(ch1_coords) > 1:
#             for puncta in ch1_coords:
#                 if isinstance(puncta, np.ndarray):
#                     ax1.plot(puncta[:, 1], puncta[:, 0], linewidth=0.5, color='magenta')
#         if len(ch2_coords) > 1:
#             for puncta in ch2_coords:
#                 if isinstance(puncta, np.ndarray):
#                     ax1.plot(puncta[:, 1], puncta[:, 0], linewidth=0.5, color='red')
#         if len(ch3_coords) > 1:
#             for puncta in ch3_coords:
#                 if isinstance(puncta, np.ndarray):
#                     ax1.plot(puncta[:, 1], puncta[:, 0], linewidth=0.5, color='yellow')
#         for ax in fig.get_axes():
#             ax.label_outer()

#         # Create scale bar
#         scalebar = ScaleBar(0.0779907, 'um', location = 'lower left', pad = 0.3, sep = 2, box_alpha = 0, color='k', length_fraction=0.3)
#         ax1.add_artist(scalebar)
#         ax1.set_xlim(0, 1024)
#         ax1.set_ylim(0, 1024)

#         # title and save
#         fig.suptitle(name, y=0.925)
#         fig.tight_layout()
#         fig.savefig(f'{plotting_folder}{name}_proof.png', bbox_inches='tight',pad_inches = 0.1, dpi = 600)
#         plt.close()










#%%

# 2025-04-28 play (color nuclei by dominant puncta channel & add nuclear outlines)
for name, image in image_mask_dict.items():
    name
    unique_val, counts = np.unique(image[3, :, :], return_counts=True)

    #plot nuclei
    dataframe = feature_information[feature_information['image_name'] == name]
    
    max_puncta = []
    for cell_num in unique_val:
        nucleus = dataframe[dataframe['cell_number'] == cell_num]
        if nucleus['area'].max() == 0:
            max_puncta_temp = nucleus.iloc[[0]]  # first row as a DataFrame
        else:
            max_puncta_temp = nucleus[nucleus['area'] == nucleus['area'].max()]
        max_puncta.append(max_puncta_temp)
    all_max_puncta = pd.concat(max_puncta, ignore_index=True)

    color_map = {0: '#D30C7B', 1: '#62E2E9', 2: '#F9DC5C', 3: 'Gray'}
    
    fig, ax1 = plt.subplots(figsize = (5, 5))
    for number in all_max_puncta['cell_number'].unique():
        number
        puncta_area = all_max_puncta[all_max_puncta['cell_number'] == number]['area'].iloc[0]
        if puncta_area <= 10:
            # make gray default
            channel = 3
        else:
            channel = all_max_puncta[all_max_puncta['cell_number'] == number]['channel'].iloc[0]
        # grab cell contour
        cell_line = all_max_puncta[all_max_puncta['cell_number'] == number]['cell_contour'].iloc[0]
        # plot cell
        ax1.fill(cell_line[:, 1], cell_line[:, 0], color=color_map[channel])

        # Compute convex hull of nuclei
        hull = ConvexHull(cell_line)
        # Get the vertices in order and close the loop by repeating the first vertex at the end
        hull_path = np.append(hull.vertices, hull.vertices[0])
        # Plot closed outline (only boundary points)
        ax1.plot(cell_line[hull_path, 1], cell_line[hull_path, 0], color='k', linewidth=1)

    # Create scale bar
    scalebar = ScaleBar(0.0779907, 'um', location = 'lower left', pad = 0.3, sep = 2, box_alpha = 0, color='k', length_fraction=0.3)
    ax1.add_artist(scalebar)
    ax1.set_xlim(0, 1024)
    ax1.set_ylim(0, 1024)

    # title and save
    fig.suptitle(name, y=0.925)
    fig.tight_layout()
    fig.savefig(f'{plotting_folder}{name}_dom_punct.png', bbox_inches='tight',pad_inches = 0.1, dpi = 600)
    plt.close()



#%% IN PROGRESS
# 2025-04-30 play (AFTER IMAGE REGISTRATION -- color nuclei by dominant puncta channel & add nuclear outlines & calculate proportion of each channel identified)

def autopct_format(pct):
    return f'{int(round(pct))}%' if pct >= 1 else ''  # Hide anything <1%


for name, image in image_mask_dict.items():
    name
    unique_val, counts = np.unique(image[3, :, :], return_counts=True)

    #plot nuclei
    dataframe = feature_information[feature_information['image_name'] == name]
    
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
        #puncta_area = all_max_puncta[all_max_puncta['cell_number'] == number]['area'].iloc[0]
        # if puncta_area <= 10:
        #     # make gray default
        #     channel = 3
        # else:
        channel = all_max_puncta[all_max_puncta['cell_number'] == number]['channel'].iloc[0]
        # grab cell contour
        cell_line = all_max_puncta[all_max_puncta['cell_number'] == number]['cell_contour'].iloc[0]
        # plot cell
        ax1.fill(cell_line[:, 1], cell_line[:, 0], color=color_map[channel])

        # Compute convex hull of nuclei
        hull = ConvexHull(cell_line)
        # Get the vertices in order and close the loop by repeating the first vertex at the end
        hull_path = np.append(hull.vertices, hull.vertices[0])
        # Plot closed outline (only boundary points)
        ax1.plot(cell_line[hull_path, 1], cell_line[hull_path, 0], color='k', linewidth=1)

    # Create scale bar
    scalebar = ScaleBar(0.0779907, 'um', location = 'lower left', pad = 0.3, sep = 2, box_alpha = 0, color='k', length_fraction=0.3)
    ax1.add_artist(scalebar)
    ax1.set_xlim(0, 1024)
    ax1.set_ylim(0, 1024)

    # Add inset pie chart for channel distribution
    inset_ax = fig.add_axes([0.0, 0.8, 0.15, 0.15])  # [x0, y0, width, height]

    # Prepare pie data
    counts = puncta_counts['count']
    labels = (100 * puncta_counts['count'] / puncta_counts['count'].sum()).round(0).astype(int)
    colors = [color_map[ch] for ch in puncta_counts['channel']]

    # Plot pie chart
    inset_ax.pie(counts, colors=colors, startangle=90, 
                counterclock=False, labeldistance=0.4, textprops={'fontsize': 6}, autopct=autopct_format)

    # title and save
    fig.suptitle(name, y=0.925)
    fig.tight_layout()
    fig.savefig(f'{plotting_folder}{name}_dom_punct.png', bbox_inches='tight',pad_inches = 0.1, dpi = 600)
    plt.close()



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
        
        binary = np.where(image[channel, :, :] > 10000, 1, 0)
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


# We are now creating a dataframe of nuclei coordinates and numbers to cross reference against our feature_information dataframe in order to assign puncta to nuclei (Refactored by ChatGPT)
from skimage.measure import label
from skimage.morphology import remove_small_objects

nuclear_information_list = []
channel = 3  # Nuclear channel

for name, image in image_mask_dict.items():
    nuclear_properties_list = []

    nuclei_labels = np.unique(image[channel])
    for nucleus_label in nuclei_labels:
        if nucleus_label == 0:
            continue  # skip background

        # Create binary mask for current nucleus
        binary_mask = (image[channel] == nucleus_label).astype(int)
        labeled_nucleus = label(binary_mask)

        # Filter out small components if needed
        if np.max(labeled_nucleus) > 1:
            labeled_nucleus = remove_small_objects(labeled_nucleus, min_size=10)

        # Extract properties
        props = feature_extractor(labeled_nucleus)

        if len(props) == 0:
            continue  # skip if nothing found

        # Assign a unique nucleus number for each region
        for i, row in props.iterrows():
            row = row.copy()  # avoid SettingWithCopyWarning
            row['nucleus_number'] = nucleus_label + (i / 10)
            row['channel'] = channel
            row['image_name'] = name
            nuclear_properties_list.append(row)

    if nuclear_properties_list:
        properties_df = pd.DataFrame(nuclear_properties_list)
        nuclear_information_list.append(properties_df)

# Combine into single DataFrame
nuclear_information = pd.concat(nuclear_information_list, ignore_index=True)
logger.info('Completed nuclear feature collection')

# save data for plotting coords
nuclear_information.to_csv(f'{output_folder}nuclear_detection_feature_info.csv')



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
        # plt.title("Reference Image")
        # plt.axis('off')

        # plt.subplot(1, 3, 2)
        # plt.imshow(rgb_img)
        # plt.title("Original Image")
        # plt.axis('off')

        # plt.subplot(1, 3, 3)
        # plt.imshow(rgb_shift)
        # plt.title("Shifted Image")
        # plt.axis('off')

        # plt.tight_layout()
        # plt.show()




#%%
# STEP 4: PLOT THINGS? -- IN PROGRESS

# 4a: overlay nuclei (unless they already are dictionaried in?)

# Round up all coordinate pairs in "registered coordinates"
# For a given image name, compare dataframes (feature_information & nuclear_information) to see if REGISTERED puncta coordinates are in nuclear coordinates (match, contains, other?) & assign puncta to nuclei



# 4b: Plot things nicely with cell outlines and pie charts








#%%
# 2025-04-29 play (color nuclei by dominant puncta channel & add nuclear outlines & calculate proportion of each channel identified)

def autopct_format(pct):
    return f'{int(round(pct))}%' if pct >= 1 else ''  # Hide anything <1%


for name, image in image_mask_dict.items():
    name
    unique_val, counts = np.unique(image[3, :, :], return_counts=True)

    #plot nuclei
    dataframe = feature_information[feature_information['image_name'] == name]
    
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
        #puncta_area = all_max_puncta[all_max_puncta['cell_number'] == number]['area'].iloc[0]
        # if puncta_area <= 10:
        #     # make gray default
        #     channel = 3
        # else:
        channel = all_max_puncta[all_max_puncta['cell_number'] == number]['channel'].iloc[0]
        # grab cell contour
        cell_line = all_max_puncta[all_max_puncta['cell_number'] == number]['cell_contour'].iloc[0]
        # plot cell
        ax1.fill(cell_line[:, 1], cell_line[:, 0], color=color_map[channel])

        # Compute convex hull of nuclei
        hull = ConvexHull(cell_line)
        # Get the vertices in order and close the loop by repeating the first vertex at the end
        hull_path = np.append(hull.vertices, hull.vertices[0])
        # Plot closed outline (only boundary points)
        ax1.plot(cell_line[hull_path, 1], cell_line[hull_path, 0], color='k', linewidth=1)

    # Create scale bar
    scalebar = ScaleBar(0.0779907, 'um', location = 'lower left', pad = 0.3, sep = 2, box_alpha = 0, color='k', length_fraction=0.3)
    ax1.add_artist(scalebar)
    ax1.set_xlim(0, 1024)
    ax1.set_ylim(0, 1024)

    # Add inset pie chart for channel distribution
    inset_ax = fig.add_axes([0.0, 0.8, 0.15, 0.15])  # [x0, y0, width, height]

    # Prepare pie data
    counts = puncta_counts['count']
    labels = (100 * puncta_counts['count'] / puncta_counts['count'].sum()).round(0).astype(int)
    colors = [color_map[ch] for ch in puncta_counts['channel']]

    # Plot pie chart
    inset_ax.pie(counts, colors=colors, startangle=90, 
                counterclock=False, labeldistance=0.4, textprops={'fontsize': 6}, autopct=autopct_format)

    # title and save
    fig.suptitle(name, y=0.925)
    fig.tight_layout()
    fig.savefig(f'{plotting_folder}{name}_dom_punct.png', bbox_inches='tight',pad_inches = 0.1, dpi = 600)
    plt.close()




