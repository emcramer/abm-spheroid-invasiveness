"""
Module for generating a pseudo-brightfield image from PhysiCell ABM output.

"""

import numpy as np
import pandas as pd
import skimage as ski
import scipy as sp
import cv2, pcdl

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