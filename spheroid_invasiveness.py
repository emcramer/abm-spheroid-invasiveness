"""
Script for processing invasiveness of tumor spheroids for "Digitize your biology!"

"""

import numpy as np
import pandas as pd
import skimage as ski
import scipy as sp
import cv2
import pcdl
import glob
import os
import time
import math
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

### Utility Functions
def load_simulation(sim_out_dir, time_step = 1440, **kwargs):
    all_timesteps = [pcdl.pyMCDS(ts, settingxml='config.xml') for ts in sorted(glob.glob(sim_out_dir+"output*.xml"))]
    selected_steps = [ts for ts in all_timesteps if ts.get_time() % time_step == 0]
    return selected_steps

def load_simulation_by_interval(sim_out_dir, interval = 1440, **kwargs):
    print("Loading simulations...")
    tree = ET.parse(sim_out_dir+'config.xml')
    xml_root = tree.getroot()
    save_interval = int(xml_root.find('.//save/full_data/interval').text)
    print(f"Save interval: {save_interval}")
    if save_interval < interval:
        all_output_files = sorted(glob.glob(sim_out_dir+"output*.xml"))
        selected_output_files = all_output_files[0::int(interval/save_interval)]
        selected_steps = [pcdl.pyMCDS(ts, settingxml='config.xml') for ts in selected_output_files]
        return selected_steps
    else:
        print("Provided interval must be larger than the save interval.")

### Image Processing Functions
def binarize_channel_otsu(image):
    _, otsu_thresh = cv2.threshold(image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresh

def find_largest_contour_channel(cleaned_image):
    if cleaned_image.dtype is not np.uint8:
        cleaned_image = cleaned_image.astype(np.uint8)
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found!")
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest
        
def create_mask_channel(image_shape, contour):
    mask = np.zeros(image_shape, dtype=np.uint8)
    # Draw the contour on the mask
    if contour is not None:
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        mask = sp.ndimage.binary_fill_holes(mask)
    if len(mask.shape) > 2:
        mask = mask.sum(axis=0)
    return mask

def segment_spheroid_channel(image, mask):
    if image.shape != mask.shape:
        print(f"Image shape and mask shape do not match! Image shape is {image.shape} while mask shape is {mask.shape}")
        return 0
    segmented = image * mask
    return segmented

### Invasiveness Quantification Functions
def calculate_centroid(contour):
    # Calculate the area of the largest contour
    contour_area = cv2.contourArea(contour)
    # Calculate normalized central moments
    moments = cv2.moments(contour)
    # Calculate the centroid of the largest contour
    centroid_x = int(moments['m10'] / moments['m00'])
    centroid_y = int(moments['m01'] / moments['m00'])
    return centroid_x, centroid_y

def calculate_radial_distance(contour):
    centroid = calculate_centroid(contour)
    # Calculate the radial distance of each point to the center of mass
    radial_distances = np.sqrt((contour[:, 0, 0] - centroid[0]) ** 2 + (contour[:, 0, 1] - centroid[1]) ** 2)
    # Calculate the median radial distance
    median_radial_distance = np.median(radial_distances)
    return radial_distances, median_radial_distance, centroid
    
def calculate_peaks(radial_distances, height):
    # Identify peaks along the contour that exceed the median distance to the center of mass
    peaks, properties = sp.signal.find_peaks(radial_distances, distance=9, height=height)
    return peaks, properties

def calc_contour(pseudo_brightfield, **kwargs):
    binarized = binarize_channel_otsu(pseudo_brightfield)
    closed = ski.morphology.closing(binarized, footprint=ski.morphology.disk(5))
    filled = sp.ndimage.binary_fill_holes(closed, structure=np.ones((5,5))).astype(int)
    contour = find_largest_contour_channel(filled)
    mask = create_mask_channel(pseudo_brightfield.shape, contour)
    segmented = segment_spheroid_channel(pseudo_brightfield, mask)
    return contour, [pseudo_brightfield, binarized, closed, filled, mask, segmented]

def quantify_contour(contour, **kwargs):
    if 'median_radial_distance' not in kwargs.keys():
        radial_distances, median_rad, centroid = calculate_radial_distance(contour)
    else:
        radial_distances, _ , centroid = calculate_radial_distance(contour)
    height = kwargs.get('median_radial_distance', median_rad)
    peaks, properties = calculate_peaks(radial_distances, height)
    return peaks, properties, radial_distances, centroid

def count_invasive_projections(contour, **kwargs):
    peaks, props, radds, centroid = quantify_contour(contour, **kwargs)
    return len(peaks)

def invasive_projection_height(contour, **kwargs):
    peaks, props, radds, centroid = quantify_contour(contour, **kwargs)
    return props['peak_heights']

def invasive_area_ratio(contour, **kwargs):
    peaks, props, radds, centroid = quantify_contour(contour)
    median_radial_distance = np.median(radds)
    spheroid_core_area = np.pi*median_radial_distance**2
    invasive_area = cv2.contourArea(contour) - spheroid_core_area
    invasive_ratio = invasive_area/spheroid_core_area
    return invasive_ratio, spheroid_core_area, invasive_area

### Pseudo Image Generation
def ceil_to_odd(n):
    """
    Round a number up to the nearest odd integer.

    Args:
        n (float): The number to round up.

    Returns:
        int: The smallest odd integer greater than or equal to n.
    """
    # Round up the number
    rounded_up = np.ceil(n)
    
    # Check if it's even
    if rounded_up % 2 == 0:
        # If even, add 1 to make it odd
        return rounded_up + 1
    else:
        # If already odd, return it
        return rounded_up

def get_coords(mcds):
    """
    Extract cell coordinates from a PhysiCell MultiCellDS object.

    Args:
        mcds (obj: pcdl.MultiCellDS): A MultiCellDS object containing simulation data.

    Returns:
        tuple: A tuple containing:
            - list: A list of tuples representing cell coordinates (x, y).
            - ndarray: A NumPy array of cell types.
    """
    cell_coords = mcds.get_cell_df()[["cell_type", "position_x", "position_y"]]
    coords = list(
        zip(
            cell_coords[['position_x']].values + 1000,
            cell_coords[['position_y']].values + 1000
        )
    )
    coords = [(int(round(c[0][0], 0)), int(round(c[1][0],0))) for c in coords]
    return coords, cell_coords[['cell_type']].values

    
def get_imsize(mcds):
    """
    Determine the size of the output image based on the simulation domain.

    Args:
        mcds (obj: pcdl.MultiCellDS): A MultiCellDS object containing simulation data.

    Returns:
        tuple: A tuple representing the image size (width, height).
    """
    im_size = (
        int(mcds.get_xyz_range()[0][1] - mcds.get_xyz_range()[0][0]), 
        int(mcds.get_xyz_range()[0][1] - mcds.get_xyz_range()[0][0])
    )
    return im_size

def extract_conc_contour(mcds, substrate, z_slice=0):
    """
    Extract substrate concentration contours from a PhysiCell MultiCellDS object.

    Args:
        mcds (obj: pcdl.MultiCellDS): A MultiCellDS object containing simulation data.
        substrate (str): The name of the substrate to extract.
        z_slice (int, optional): The z-slice to extract data from. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - ndarray: The x-coordinates of the substrate concentration data.
            - ndarray: The y-coordinates of the substrate concentration data.
            - ndarray: The substrate concentration values at the specified z-slice.
            - tuple: The minimum and maximum substrate concentration values.
    """
    # handle z_slice input
    _, _, ar_p_axis = mcds.get_mesh_mnp_axis()
    if not (z_slice in ar_p_axis):
        z_slice = ar_p_axis[abs(ar_p_axis - z_slice).argmin()]
        
    # get data z slice
    df_conc = mcds.get_conc_df(drop=set(), keep=set()) #values=1, drop=set(), keep=set())
    df_conc = df_conc.loc[(df_conc.mesh_center_p == z_slice),:]
    # extend to x y domain border
    df_mmin = df_conc.loc[(df_conc.mesh_center_m == df_conc.mesh_center_m.min()), :].copy()
    df_mmin.mesh_center_m = mcds.get_xyz_range()[0][0]
    df_mmax = df_conc.loc[(df_conc.mesh_center_m == df_conc.mesh_center_m.max()), :].copy()
    df_mmax.mesh_center_m = mcds.get_xyz_range()[0][1]
    df_conc = pd.concat([df_conc, df_mmin, df_mmax], axis=0)
    df_nmin = df_conc.loc[(df_conc.mesh_center_n == df_conc.mesh_center_n.min()), :].copy()
    df_nmin.mesh_center_n = mcds.get_xyz_range()[1][0]
    df_nmax = df_conc.loc[(df_conc.mesh_center_n == df_conc.mesh_center_n.max()), :].copy()
    df_nmax.mesh_center_n = mcds.get_xyz_range()[1][1]
    df_conc = pd.concat([df_conc, df_nmin, df_nmax], axis=0)
    # sort dataframe
    df_conc.sort_values(['mesh_center_m', 'mesh_center_n', 'mesh_center_p'], inplace=True)
    
    # meshgrid shape
    ti_shape = (mcds.get_voxel_ijk_axis()[0].shape[0]+2, mcds.get_voxel_ijk_axis()[1].shape[0]+2)
    x = (df_conc.loc[:,'mesh_center_m'].values).reshape(ti_shape)
    y = (df_conc.loc[:,'mesh_center_n'].values).reshape(ti_shape)
    z = (df_conc.loc[:,substrate].values).reshape(ti_shape)

    vmin = np.floor(df_conc.loc[:,substrate].min())
    vmax = np.ceil(df_conc.loc[:,substrate].max())
    vvals = (vmin, vmax)
    
    # return the values of the substrate at x, y, and z locations
    return x, y, z, vvals

def generate_cells(mcds, **kwargs):
    # get point spread function kernel/sigma to apply
    psf_sigma = kwargs.get('psf_kernel', None)
    
    # get the radius of the cells from the simulation
    cell_radius = ceil_to_odd(mcds.get_cell_df()['radius'].mean())

    # get the distribution of the radii of the cells
    cell_radius_std = np.std(mcds.get_cell_df()['radius'].values)

    # get the coordinates of the cells
    coords, cell_types = get_coords(mcds)

    # get the dimensions of the simulation (
    image_shape = get_imsize(mcds)

    # Initialize a 2D image array (all zeros)
    image = np.zeros(image_shape)
    
    # Apply a ring around the disk at each (x, y) point
    for (x, y) in coords:
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            # randomly get a cell radius
            this_cell_radius = np.random.normal(cell_radius, cell_radius_std)
            rr, cc = ski.draw.disk((y, x), this_cell_radius, shape=image_shape)
            
            # Create a radial intensity gradient (dark in center, bright at edge)
            distance_from_center = np.sqrt((rr - y) ** 2 + (cc - x) ** 2)
            normalized_distance = distance_from_center / cell_radius
            
            # Intensity increases as you move toward the edge
            intensity_profile = normalized_distance
            
            # Set the intensity values in the disk region
            image[rr, cc] += intensity_profile * (normalized_distance <= 1)

    cells = (image/image.max())*255

    # apply a blur to the substrates
    if psf_sigma is not None:
        cells = sp.ndimage.gaussian_filter(cells, psf_sigma)
    
    return cells, cell_radius

def generate_substrates(mcds, **kwargs):
    # get point spread function kernel/sigma to apply
    psf_sigma = kwargs.get('psf_sigma', None)
    
    # get environmental substrates to include in the grayscale generation
    substrates = kwargs.get('substrates', None)

    # get the intensity factor to apply for each substrate, should be (0, 255]
    intensity_factor = kwargs.get('intensity_factor', [255])

    # get the dimensions of the simulation & init an array
    image_shape = get_imsize(mcds)
    subs_all = np.zeros(image_shape)

    # iterate through substrates and obtain concentrations/contours
    if substrates is not None:
        for i, sub in enumerate(substrates):
            sub_x, sub_y, sub_z , vvals = extract_conc_contour(mcds, sub, z_slice=0)
            sub_resized = cv2.resize(sub_z, dsize=image_shape, interpolation=cv2.INTER_CUBIC)
            if sub_resized.max() > 0: # catch the instance at t=0 when there may not be any of the substrate in the environment             
                if len(intensity_factor) > 1:
                    sub_resized = (sub_resized/sub_resized.max())*intensity_factor[i]
                else:
                    sub_resized = (sub_resized/sub_resized.max())*intensity_factor[0]
            subs_all = np.stack((subs_all, sub_resized), axis=0)
    else:
        print("Please indicate a substrate. Returned matrix is 0 everywhere.")
        return np.expand_dims(subs_all, axis=0)
            
    # generate a maximum intensity projection of the substrates
    subs_max_i = np.sum(subs_all, axis=0)

    # apply a blur to the substrates if psf argument passed
    subs_blurred = None
    if psf_sigma is not None:
        subs_blurred = sp.ndimage.gaussian_filter(subs_max_i, sigma=psf_sigma)
        subs_blurred = np.expand_dims(subs_blurred, axis=0)

    # concatenate the substrate data
    subs_max_i = np.expand_dims(subs_max_i, axis = 0)
    all_substrate = np.concatenate((subs_all, subs_max_i), axis=0)
    if subs_blurred is not None:
        all_substrate = np.concatenate((all_substrate, subs_blurred), axis=0)
    
    return all_substrate

def abm2gray(mcds, **kwargs):
    cells, avg_cell_radius = generate_cells(mcds, **kwargs)
    substrate_data = generate_substrates(mcds, **kwargs)
    combined = np.sum(np.stack((substrate_data[-1,:,:], cells), axis=0), axis=0)
    gray = combined/combined.max()*255
    return gray, cells, substrate_data

### Quantifying Output from Simulations
def quantify_sim(output_parent_dir):
    out_dirs = sorted(
        [d for d in os.listdir(output_parent_dir) if os.path.isdir(os.path.join(output_parent_dir,d))]
    )

    # only run if the results file needs to be re-generated
    simulation_results = []
    for j, d in enumerate(tqdm(out_dirs)):
        print(f"Loading simulation: {d}")
        selected_steps = load_simulation_by_interval(output_parent_dir+d+"/")
        print("Finished loading.")
        n_steps = len(selected_steps)
        timesteps = [f"{24*k}h" for k in range(n_steps)]
        invasive_area_ratios = [None] * n_steps
        spheroid_core_areas = [None] * n_steps
        invasive_areas = [None] * n_steps
        invasive_projection_heights = [None] * n_steps
        n_invasive_projections = [None] * n_steps
        
        print(f"Processing simulation: {d}")
        rad_threshold = 0
        for i, ts in enumerate(selected_steps):
            pseudo_brightfield, cells, substrates = abm2gray(ts,substrates=['ecm'])
            contour, matrices = calc_contour(pseudo_brightfield)
            if i == 0: # if this is the first time step, then set the threshold based on the median radial distance
                invasive_projection_heights[i] = invasive_projection_height(contour)
                rad_threshold = np.median(invasive_projection_heights[i])
            else:
                invasive_projection_heights[i] = invasive_projection_height(contour, median_radial_distance=rad_threshold)
            invasive_area_ratios[i], spheroid_core_areas[i], invasive_areas[i] = invasive_area_ratio(contour, median_radial_distance=rad_threshold)
            n_invasive_projections[i] = count_invasive_projections(contour, median_radial_distance=rad_threshold)

        props_df = pd.DataFrame({
        'sample_name': f"sample_{j}",
        'experimental_condition': d[0:-4],
        'timepoint':timesteps,
        'n_invasive_projections':n_invasive_projections,
        'invasive_projection_heights': invasive_projection_heights,
        'invasive_areas':invasive_areas,
        'spheroid_core_areas':spheroid_core_areas,
        'invasive_area_ratios':invasive_area_ratios
        })
        props_df.set_index('timepoint', inplace=True)
        simulation_results.append(props_df)
        print("Finished processing simulation.")
        
    simulation_results_df = pd.concat(simulation_results)
    simulation_results_df['hours'] = [int(hr[0:-1]) for hr in simulation_results_df.index]
    simulation_results_df['experimental_condition'] = ["_".join(s.split("_")[0:-1]) for s in simulation_results_df['experimental_condition']]
    simulation_results_df.to_csv("../output/tables/"+output_parent_dir.split("/")[-1]+".csv")
    return simulation_results_df

### Plotting Utilities and Variables
def get_ratio_label(cat_str):
    if "only" in cat_str and "tumor" in cat_str:
        return f"1:0 Tumor:Fibroblast"
    elif "only" in cat_str and "fibroblast" in cat_str:
        return f"0:1 Tumor:Fibroblast"
    else:
        tumor = cat_str.split("_")[1]
        fibro = cat_str.split("_")[-2]
        return f"{tumor}:{fibro} Tumor:Fibroblast"

# Define jitter function to add random noise to the x-coordinates
def add_jitter(x, jitter_amount=0.1):
    return x + np.random.uniform(-jitter_amount, jitter_amount, size=len(x))

hour_markers = ['.', '^', '*', 'D', 's', 'x', 'v', '+', 'h']

def plot_jitter_errbars(
    df,
    group_var,
    x_tick_var,
    x_label_var,
    y_var,
    ax=None,
    err_type='sem',
    condition_markers = ['.', '^', '*', 'D', 's', 'x', 'v', '+', 'h', 'o', '>', '<']
):

    # Use a color palette from Matplotlib's colormap (tab10 for 10 distinct colors)
    cmap = plt.get_cmap('tab10', len(df[group_var].unique()))
    colors = {condition: cmap(i) for i, condition in enumerate(df[group_var].unique())}
    color_keys = list(colors.keys())

    labels_raw = list(set(df[group_var]))
    labels_proc = [get_ratio_label(l) for l in labels_raw]
    output_labels_map = {k:p for k,p in zip(labels_raw, labels_proc)}

    # setting x axis ticks and labels
    xticks = list(np.unique(df[x_tick_var].values))
    xlabels = sorted(list(np.unique(df[x_label_var].values)), key=lambda x: int(''.join(filter(str.isdigit, str(x)))))

    # Group by 'experimental_condition' and 'timepoint'
    grouped = df.groupby([group_var, x_tick_var])

    # Calculate mean and standard error for each group
    mean_se_df = grouped[y_var].agg(['mean', err_type]).reset_index()

    # Ensure data is sorted by 'timepoint'
    mean_se_df = mean_se_df.sort_values(by=[x_tick_var])

    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
    

    # plot the points with jitter
    i=0
    for key, grp in df.groupby([group_var]):
        jittered_timepoints = add_jitter(grp[x_tick_var].values, jitter_amount=8)  # Add jitter to timepoints
        ax.scatter(
            list(jittered_timepoints), 
            grp[y_var], 
            marker=condition_markers[i], 
            alpha=0.5, 
            label=output_labels_map[key[0]], 
            s = 10,
            color=colors[key[0]]
            )
        i += 1

    # Plot the means with standard error bars
    i=0
    for key, grp in mean_se_df.groupby(group_var):
        jittered_timepoints = add_jitter(grp[x_tick_var].values, jitter_amount=8)  # Add jitter to timepoints
        # Add error bars for the mean with standard error
        ax.errorbar(
            list(jittered_timepoints), 
            grp['mean'], 
            yerr=grp[err_type], 
            fmt=condition_markers[i], 
            capsize=5, 
            alpha=1.0, 
            color=colors[key]
            )
        i+=1

    # Custom legend with alpha=1
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for j, handle in enumerate(handles):
        label_color = handle.properties()['facecolor']
        new_handle = plt.Line2D([], [], color=label_color, alpha=1.0, marker=condition_markers[j], linestyle='', markersize=8)
        new_handles.append(new_handle)

    # Move the legend to below the x-axis
    ax.legend(new_handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

    # set the axis ticks and labels
    ax.set_xticks(xticks)  # Set the ticks
    ax.set_xticklabels(xlabels)  # Set the tick labels
    #ax.set_title("Standard error of the mean shown")
    #ax.set_xlabel("Timepoint (hours)")
    #ax.set_ylabel("Number of Invasive Projections (count)")

#fig.suptitle("Number of invasive projections over time under hypothesis two", y=.95)
#plt.show()


### Run Analysis
def main():
    # quantify the output for each simulation
    h1_ics = quantify_sim("../PhysiCell/output/hypothesis1_ics_permuted/")
    h2_ics = quantify_sim("../PhysiCell/output/hypothesis2_ics_permuted/")
    h1_rs = quantify_sim("../PhysiCell/output/hypothesis1_random_seeds/")
    h2_rs = quantify_sim("../PhysiCell/output/hypothesis2_random_seeds/")

    # visualization for each simulation
    # Use a color palette from Matplotlib's colormap (tab10 for 10 distinct colors)
    cmap = cm.get_cmap('tab10', len(h1_ics['experimental_condition'].unique()))
    colors = {condition: cmap(i) for i, condition in enumerate(h1_ics['experimental_condition'].unique())}
    color_keys = list(colors.keys())

    labels_raw = list(set(h1_ics['experimental_condition']))
    labels_proc = [get_ratio_label(l) for l in labels_raw]
    output_labels_map = {k:p for k,p in zip(labels_raw, labels_proc)}

    # setting x axis ticks and labels
    xticks = list(np.unique(h1_ics['hours'].values))
    xlabels = sorted(list(np.unique(h1_ics.index.values)), key=lambda x: int(''.join(filter(str.isdigit, x))))

    # viz for random seeds
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax = ax.ravel()
    plot_jitter_errbars(
        h1_rs, 
        group_var = 'experimental_condition', 
        x_tick_var = 'hours', 
        x_label_var = 'timepoint', 
        y_var = 'n_invasive_projections',
        ax = ax[0],
        err_type='std'
    )
    ax[0].set_title('Hypothesis 1')
    ax[0].set_ylabel('Number of Invasive Projections')
    ax[0].set_xlabel('Timepoint')
    ax[0].get_legend().remove()

    plot_jitter_errbars(
        h2_rs, 
        group_var = 'experimental_condition', 
        x_tick_var = 'hours', 
        x_label_var = 'timepoint', 
        y_var = 'n_invasive_projections',
        ax = ax[1],
        err_type='std'
    )
    ax[1].set_title('Hypothesis 2')
    ax[1].set_ylabel('Number of Invasive Projections')
    ax[1].set_xlabel('Timepoint')

    # add and enhance legend
    handles, labels = ax[1].get_legend_handles_labels()
    new_handles = []
    for j, handle in enumerate(handles):
        label_color = handle.properties()['facecolor']
        new_handle = plt.Line2D([], [], color=label_color, alpha=1.0, marker=condition_markers[j], linestyle='', markersize=8)
        new_handles.append(new_handle)
    ax[1].legend(new_handles, labels, bbox_to_anchor=[-0.1, -0.1], ncol=4, loc='upper center')

    fig.suptitle('Tumor spheroid invasiveness from randomized initial cell conditions', y=0.95, fontsize=18)
    plt.savefig("../output/figures/randomized_ics.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    # viz for random initial cell conditions
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax = ax.ravel()
    plot_jitter_errbars(
        h1_ics, 
        group_var = 'experimental_condition', 
        x_tick_var = 'hours', 
        x_label_var = 'timepoint', 
        y_var = 'n_invasive_projections',
        ax = ax[0],
        err_type='std'
    )
    ax[0].set_title('Hypothesis 1')
    ax[0].set_ylabel('Number of Invasive Projections')
    ax[0].set_xlabel('Timepoint')
    ax[0].get_legend().remove()

    plot_jitter_errbars(
        h2_ics, 
        group_var = 'experimental_condition', 
        x_tick_var = 'hours', 
        x_label_var = 'timepoint', 
        y_var = 'n_invasive_projections',
        ax = ax[1],
        err_type='std'
    )
    ax[1].set_title('Hypothesis 2')
    ax[1].set_ylabel('Number of Invasive Projections')
    ax[1].set_xlabel('Timepoint')

    # add and enhance legend
    handles, labels = ax[1].get_legend_handles_labels()
    new_handles = []
    for j, handle in enumerate(handles):
        label_color = handle.properties()['facecolor']
        new_handle = plt.Line2D([], [], color=label_color, alpha=1.0, marker=condition_markers[j], linestyle='', markersize=8)
        new_handles.append(new_handle)
    ax[1].legend(new_handles, labels, bbox_to_anchor=[-0.1, -0.1], ncol=4, loc='upper center')

    fig.suptitle('Tumor spheroid invasiveness from randomized initial cell conditions', y=0.95, fontsize=18)
    plt.savefig("../output/figures/randomized_ics.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    print("Done!")

# Run it!!
main()