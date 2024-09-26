import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import skimage as ski
from tqdm import tqdm
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import binary_fill_holes
import cv2, nd2, os, glob, re, pcdl

def calculate_centroid(contour):
    """
    Calculates the centroid of a given contour.

    This function computes the centroid (center of mass) of a contour using 
    image moments. It first calculates the area of the contour, then computes 
    the normalized central moments to determine the coordinates of the centroid.

    Parameters:
    contour (numpy.ndarray): A numpy array representing the contour. 
                             It should be of shape (n, 1, 2), where `n` 
                             is the number of points in the contour. 

    Returns:
    tuple: A tuple containing:
        - int: The x-coordinate of the centroid.
        - int: The y-coordinate of the centroid.

    Example:
    >>> centroid_x, centroid_y = calculate_centroid(contour)
    >>> print("Centroid Coordinates:", centroid_x, centroid_y)
    """
    # Calculate the area of the largest contour
    contour_area = cv2.contourArea(contour)
    # Calculate normalized central moments
    moments = cv2.moments(contour)
    # Calculate the centroid of the largest contour
    centroid_x = int(moments['m10'] / moments['m00'])
    centroid_y = int(moments['m01'] / moments['m00'])
    return centroid_x, centroid_y


def binarize_channel_otsu(image):
    _, otsu_thresh = cv2.threshold(image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresh

def calc_contour_props(mask, radial_distances, median_radial_distance, peaks, **kwargs):
    """
    Calculates contour properties based on radial distances and identifies peaks.

    This function calculates various properties related to contours from the 
    provided mask and radial distances. It identifies peaks in the radial 
    distances and can optionally plot the results if specified in the keyword arguments.

    Parameters:
    mask (numpy.ndarray): A binary mask of the contour, typically a 2D numpy array.
    radial_distances (numpy.ndarray): An array of radial distances calculated 
                                       from the contour points.
    median_radial_distance (float): The median of the radial distances.
    peaks (numpy.ndarray): An array of peak indices identified in the radial distances.
    **kwargs: Additional keyword arguments. Supports:
        - 'plot' (bool): If True, plots the radial distance and peaks.

    Returns:
    dict: A dictionary containing the following contour properties:
        - 'radial_distances' (numpy.ndarray): The array of radial distances.
        - 'median_radial_distance' (float): The median radial distance.
        - 'peaks' (numpy.ndarray): The indices of identified peaks.
        - 'peak_properties' (dict): A dictionary of properties for each identified peak.

    Example:
    >>> properties = calc_contour_props(mask, radial_distances, median_radial_distance, peaks, plot=True)
    >>> print(properties)
    """
    peaks, peak_props = calculate_peaks(radial_distances, median_radial_distance)
    if 'plot' in kwargs.keys():
        if kwargs['plot']:
            plot_radial_distance(mask, radial_distances, median_radial_distance, peaks)
    properties = {
        'radial_distances': radial_distances,
        'median_radial_distance': median_radial_distance,
        'projections': peaks,
        'projection_properties': peak_props,
        'n_projections': len(peaks)
    }
    return properties

def calculate_peaks(radial_distances, height):
    """
    Identifies peaks in radial distances that exceed a specified height.

    This function uses the `find_peaks` method from the SciPy library to identify 
    peaks in an array of radial distances. It looks for peaks that exceed the 
    given height threshold and considers a minimum distance between consecutive 
    peaks.

    Parameters:
    radial_distances (numpy.ndarray): An array of radial distances calculated 
                                       from a contour. It should be a 1D numpy array.
    height (float): The minimum height threshold for a peak to be considered valid.

    Returns:
    tuple: A tuple containing:
        - numpy.ndarray: An array of indices of the identified peaks.
        - dict: A dictionary of properties for each identified peak, such as 
                prominence and width.

    Example:
    >>> peaks, properties = calculate_peaks(radial_distances, height=0.5)
    >>> print("Identified Peaks:", peaks)
    >>> print("Peak Properties:", properties)
    """
    # Identify peaks along the contour that exceed the median distance to the center of mass
    peaks, properties = find_peaks(radial_distances, distance=9, height=height)
    return peaks, properties

def calculate_radial_distance(contour):
    """
    Calculates the radial distances of points in a contour from its centroid.

    This function computes the centroid of a given contour and then calculates 
    the radial distance of each point in the contour to the centroid. 
    It also computes the median of these radial distances.

    Parameters:
    contour (numpy.ndarray): A numpy array representing the contour. 
                             It should be of shape (n, 1, 2), where 
                             `n` is the number of points in the contour. 
                             Each point is represented as (x, y) coordinates.

    Returns:
    tuple: A tuple containing:
        - numpy.ndarray: An array of radial distances of each point from the centroid.
        - float: The median of the radial distances.

    Example:
    >>> radial_distances, median_distance = calculate_radial_distance(contour)
    >>> print("Radial Distances:", radial_distances)
    >>> print("Median Radial Distance:", median_distance)
    """
    centroid = calculate_centroid(contour)
    # Calculate the radial distance of each point to the center of mass
    radial_distances = np.sqrt((contour[:, 0, 0] - centroid[0]) ** 2 + (contour[:, 0, 1] - centroid[1]) ** 2)
    # Calculate the median radial distance
    median_radial_distance = np.median(radial_distances)
    return radial_distances, median_radial_distance


def segment_spheroid_channel(image, mask):
    """
    Segments an image using a binary mask.

    This function applies a binary mask to the input image, 
    effectively isolating the regions of interest defined by 
    the mask. If the shapes of the image and the mask do not 
    match, an error message is printed and the function returns 0.

    Parameters:
    image (numpy.ndarray): The input image to be segmented. 
                           It should be a 2D or 3D numpy array.
    mask (numpy.ndarray): A binary mask of the same shape as the image. 
                          It should be a numpy array with values of 
                          0 (background) and 1 (foreground).

    Returns:
    numpy.ndarray or int: The segmented image where the masked areas 
                          are retained, and the rest are set to zero. 
                          Returns 0 if the shapes do not match.

    Example:
    >>> segmented_image = segment_spheroid_channel(input_image, binary_mask)
    >>> cv2.imshow('Segmented Image', segmented_image)
    >>> cv2.waitKey(0)
    >>> cv2.destroyAllWindows()
    """
    if image.shape != mask.shape:
        print(f"Image shape and mask shape do not match! Image shape is {image.shape} while mask shape is {mask.shape}")
        return 0
    segmented = image * mask
    return segmented

def create_mask_channel(image_shape, contour):
    """
    Creates a binary mask from a given contour based on the specified image shape.

    This function initializes a mask with zeros (black) and draws the specified 
    contour onto it. If the contour is not `None`, it fills the contour shape 
    on the mask. Optionally, if the resulting mask has more than two dimensions, 
    it collapses the dimensions by summing along the specified axes.

    Parameters:
    image_shape (tuple): The shape of the image (height, width) 
                         used to initialize the mask. 
                         It should be a tuple of two integers.
    contour (numpy.ndarray): The contour to draw on the mask. 
                             It should be a numpy array of shape (n, 1, 2), 
                             where `n` is the number of points in the contour.

    Returns:
    numpy.ndarray: A binary mask of the same shape as the input image, 
                   where the area inside the contour is filled.

    Example:
    >>> mask = create_mask_channel((512, 512), contour)
    >>> cv2.imshow('Mask', mask)
    >>> cv2.waitKey(0)
    >>> cv2.destroyAllWindows()
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    # Draw the contour on the mask
    if contour is not None:
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        mask = binary_fill_holes(mask)
    if len(mask.shape) > 2:
        mask = mask.sum(axis=0)
    return mask


def find_largest_contour_channel(cleaned_image):
    """
    Finds the largest contour in a given binary image.

    This function takes a binary image as input, converts it to an 
    unsigned 8-bit integer format if necessary, and then finds all 
    external contours in the image. It identifies and returns the 
    largest contour based on contour area.

    Parameters:
    cleaned_image (numpy.ndarray): A binary image (2D array) 
                                   where contours are to be detected. 
                                   The image should be in a format 
                                   compatible with OpenCV functions.

    Returns:
    numpy.ndarray or None: The largest contour as a numpy array of 
                           points if found, otherwise returns None 
                           if no contours are detected.

    Raises:
    ValueError: If the input image is not a 2D array or is not in a 
                valid format for contour detection.
    
    Example:
    >>> largest_contour = find_largest_contour_channel(binary_image)
    >>> if largest_contour is not None:
    >>>     print("Largest contour found:", largest_contour)
    >>> else:
    >>>     print("No contours found.")
    """
    if cleaned_image.dtype is not np.uint8:
        cleaned_image = cleaned_image.astype(np.uint8)
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found!")
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest

def extract_conc_contour(mcds, substrate, z_slice=0):
    """
    Extracts a 2D concentration map of a specific substrate from a given z-slice in an agent-based model (ABM) simulation.

    Parameters:
    -----------
    mcds : object
        A pyMCDS object containing the agent-based model (ABM) simulation data, including information on mesh grids and concentration fields.
    
    substrate : str
        The name of the substrate (e.g., oxygen, glucose) for which the concentration data is to be extracted.
    
    z_slice : float, optional
        The z-coordinate (height) at which to extract the concentration data. Defaults to 0.
        If the provided z_slice is not available in the ABM mesh, the function will find the closest available slice.

    Returns:
    --------
    x : numpy.ndarray
        A 2D array representing the x-coordinates (mesh_center_m) for the concentration data in the extracted slice.
    
    y : numpy.ndarray
        A 2D array representing the y-coordinates (mesh_center_n) for the concentration data in the extracted slice.
    
    z : numpy.ndarray
        A 2D array of the concentration values for the specified substrate at the given (x, y) coordinates.
    
    vvals : tuple of floats
        A tuple containing the minimum and maximum concentration values for the specified substrate in the extracted slice.

    Process:
    --------
    1. Validates the `z_slice` input and adjusts it to the closest available z-slice if necessary.
    2. Retrieves the concentration data for the specified substrate at the given z-slice.
    3. Ensures the concentration data extends to the x and y domain borders by adding boundary rows and columns.
    4. Sorts the data by mesh center coordinates (m, n, p) for consistent ordering.
    5. Reshapes the concentration data into 2D arrays for x, y, and z dimensions to create a meshgrid.
    6. Returns the concentration data along with the minimum and maximum values.

    Example:
    --------
    # Extract the concentration map for 'oxygen' at z-slice 0
    x, y, z, vvals = extract_conc_contour(mcds, 'oxygen', z_slice=0)

    Notes:
    ------
    - The function assumes that the pyMCDS object (`mcds`) contains a structured mesh grid and concentration data.
    - The concentration data is reshaped into a 2D format to allow easy plotting of contour maps.
    - Boundary values are added to ensure that the data extends fully to the domain edges.
    """
    
    # Handle z_slice input: find closest available z-slice if necessary
    _, _, ar_p_axis = mcds.get_mesh_mnp_axis()
    if not (z_slice in ar_p_axis):
        z_slice = ar_p_axis[abs(ar_p_axis - z_slice).argmin()]
        
    # Get concentration data for the specified z-slice
    df_conc = mcds.get_conc_df(drop=set(), keep=set())
    df_conc = df_conc.loc[(df_conc.mesh_center_p == z_slice), :]

    # Extend concentration data to x and y domain borders
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

    # Sort the dataframe by mesh center coordinates
    df_conc.sort_values(['mesh_center_m', 'mesh_center_n', 'mesh_center_p'], inplace=True)
    
    # Define the shape of the meshgrid
    ti_shape = (mcds.get_voxel_ijk_axis()[0].shape[0] + 2, mcds.get_voxel_ijk_axis()[1].shape[0] + 2)
    
    # Reshape data into 2D arrays for x, y, and z
    x = df_conc.loc[:, 'mesh_center_m'].values.reshape(ti_shape)
    y = df_conc.loc[:, 'mesh_center_n'].values.reshape(ti_shape)
    z = df_conc.loc[:, substrate].values.reshape(ti_shape)

    # Determine the minimum and maximum values of the concentration for the given substrate
    vmin = np.floor(df_conc.loc[:, substrate].min())
    vmax = np.ceil(df_conc.loc[:, substrate].max())
    vvals = (vmin, vmax)
    
    # Return x, y, z coordinates and concentration range values
    return x, y, z, vvals

def get_coords(mcds):
    """
    Extracts the coordinates and cell types from an agent-based model (ABM) simulation dataset.

    Parameters:
    -----------
    mcds : object
        A pyMCDS object that contains the simulation data at a particular timestep, 
        including information about the cells' positions and types.

    Returns:
    --------
    coords : list of tuples
        A list of tuples representing the (x, y) coordinates of each cell. 
        The coordinates are offset by +1000 to shift the origin to a new reference frame.
    
    cell_types : numpy.ndarray
        A NumPy array containing the cell types corresponding to the extracted coordinates.

    Process:
    --------
    1. Retrieves the cell data (cell type, x-position, y-position) from the simulation dataset using the `get_cell_df` method.
    2. Offsets the x and y coordinates by +1000 to re-center the coordinate system.
    3. Rounds the coordinates to the nearest integer for easier processing.
    4. Returns a list of (x, y) coordinates and the corresponding cell types.

    Example:
    --------
    # Extract coordinates and cell types from the ABM data
    coords, cell_types = get_coords(mcds)

    Notes:
    ------
    - The function assumes that the pyMCDS object (`mcds`) contains a DataFrame with columns 'position_x' and 'position_y'.
    - The coordinates are adjusted by adding 1000 to both x and y values, which may be needed based on the specific coordinate system of the simulation.
    """
    cell_coords = mcds.get_cell_df()[["cell_type", "position_x", "position_y"]]
    
    # Adjust coordinates by adding 1000 and rounding to nearest integer
    coords = list(
        zip(
            cell_coords[['position_x']].values + 1000,
            cell_coords[['position_y']].values + 1000
        )
    )
    coords = [(int(round(c[0][0], 0)), int(round(c[1][0],0))) for c in coords]
    
    # Return coordinates and cell types
    return coords, cell_coords[['cell_type']].values

    
def get_imsize(mcds):
    """
    Determines the size of the output image based on the spatial extent of the ABM simulation data.

    Parameters:
    -----------
    mcds : object
        A pyMCDS object containing the spatial data of the agent-based model (ABM) simulation, 
        including the x, y, and z coordinate ranges.

    Returns:
    --------
    im_size : tuple of ints
        A tuple (width, height) representing the size of the output image in pixels.
        This is calculated based on the range of the x-coordinate.

    Process:
    --------
    1. Retrieves the range of x-coordinates from the simulation using the `get_xyz_range` method.
    2. Computes the image size by calculating the difference between the maximum and minimum x-values, 
       assuming a square image with equal width and height.
    3. Returns the calculated image size as a tuple (width, height).

    Example:
    --------
    # Get the size of the image to be generated
    im_size = get_imsize(mcds)

    Notes:
    ------
    - The function assumes that the x and y dimensions have the same extent, 
      and it uses the range of x-coordinates to determine the size of the image.
    """
    im_size = (
        int(mcds.get_xyz_range()[0][1] - mcds.get_xyz_range()[0][0]), 
        int(mcds.get_xyz_range()[0][1] - mcds.get_xyz_range()[0][0])
    )
    return im_size

    
def generate_cells_image_psf(coordinates, image_size, kernel_size=5, intensity=255):
    """
    Generate a grayscale image from a list of (x, y) coordinates with a point spread function applied.
    
    Args:
        coordinates (list): List of (x, y) tuples representing the coordinates of the points.
        image_size (tuple): Size of the image in the form (height, width).
        kernel_size (int): Size of the kernel for the point spread function (Gaussian blur).
        intensity (int): Intensity value for the points (default: 255 for white).
    
    Returns:
        numpy.ndarray: The resulting grayscale image.
    """
    
    # Create a blank grayscale image
    height, width = image_size
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Place each point onto the image
    for x, y in coordinates:
        if 0 <= x < width and 0 <= y < height:
            image[y, x] = intensity  # Set the pixel value at (y, x) to the desired intensity
    
    # Apply a Gaussian blur as a point spread function
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)
    
    return blurred_image

def generate_cells_image_dilation(coordinates, image_size, kernel_size=5, intensity=255):
    """
    Generate a grayscale image from a list of (x, y) coordinates with dilation applied.
    
    Args:
        coordinates (list): List of (x, y) tuples representing the coordinates of the points.
        image_size (tuple): Size of the image in the form (height, width).
        kernel_size (int): Size of the kernel for dilation.
        intensity (int): Intensity value for the points (default: 255 for white).
    
    Returns:
        numpy.ndarray: The resulting grayscale image with dilation.
    """
    
    # Create a blank grayscale image
    height, width = image_size
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Place each point onto the image
    for x, y in coordinates:
        if 0 <= x < width and 0 <= y < height:
            image[y, x] = intensity  # Set the pixel value at (y, x) to the desired intensity
    
    # Define the structuring element (kernel) for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply dilation to spread the points
    dilated_image = cv2.dilate(image, kernel)
    
    return dilated_image

def abm2gray(abm_timestep, kernel_size):
    """
    Converts an agent-based model (ABM) timestep into a grayscale image representation.

    Parameters:
    -----------
    abm_timestep : object
        An object representing a specific timestep from an agent-based model simulation. 
        This object contains the information about cell positions and extracellular matrix (ECM) concentrations at the given timestep.
    
    kernel_size : int
        The size of the dilation kernel used to generate the cells' image from the coordinates. 
        This defines how much the cells are expanded in the image representation.

    Returns:
    --------
    gray : numpy.ndarray
        A 2D grayscale image representing both the spatial arrangement of cells and the concentration of ECM at the given timestep. 
        Cells are represented as bright regions, and ECM concentration is added to the image, with higher concentrations leading to brighter values.

    Steps:
    ------
    1. Extracts the cell coordinates and types from the agent-based model timestep.
    2. Determines the image size based on the ABM timestep.
    3. Generates a binary image representing the cells, using dilation based on the `kernel_size`.
    4. Extracts the ECM concentration contour (in the z-plane) at the current timestep.
    5. Resizes the ECM concentration contour to match the image size, using cubic interpolation, and scales the ECM values to enhance contrast.
    6. Combines the cell image and the ECM concentration into a single grayscale image.

    Example Usage:
    --------------
    # Convert a specific ABM timestep into a grayscale image with a dilation kernel size of 19
    gray_image = abm2gray(abm_timestep, kernel_size=19)

    Notes:
    ------
    - The function uses OpenCV (`cv2`) for image resizing and dilation operations.
    - The intensity of the ECM in the output image is multiplied by 255 to match the scale of the cell image.
    - This function assumes that the `abm_timestep` object contains all the necessary data for extracting cell coordinates and ECM concentrations.
    """
    
    # Extract coordinates and cell types from the ABM timestep
    coords, cell_types = get_coords(abm_timestep)
    
    # Get the image size based on the ABM timestep
    im_size = get_imsize(abm_timestep)
    
    # Generate a binary image of cells using dilation based on the given kernel size
    cells = generate_cells_image_dilation(coords, im_size, kernel_size, intensity=255)
    
    # Extract ECM concentration contour in the z-plane at the current timestep
    ecm_x, ecm_y, ecm_z, vvals = extract_conc_contour(abm_timestep, 'ecm', z_slice=0)
    
    # Resize the ECM contour to match the image size and scale the values by 255
    ecm_resized = cv2.resize(ecm_z, dsize=im_size, interpolation=cv2.INTER_CUBIC) * 255
    
    # Combine the ECM and cell images into a single grayscale image
    gray = ecm_resized + cells
    
    return gray


def load_mcds_timesteps(directory, **kwargs):
    """
    Loads and selects timesteps from MultiCellDS (MCDS) simulation output files within a specified directory.
    
    Parameters:
    -----------
    directory : str
        The path to the directory containing the MCDS output XML files. The function expects filenames to have the format 
        'output*.xml', where '*' represents the timestep number or identifier.
    
    **kwargs : dict, optional
        Additional keyword arguments. Supports:
            - 'step_time': int
                The time step interval (in minutes) to filter the simulation timesteps. Only timesteps that are multiples 
                of this value will be selected. Default is 1440 (corresponding to 1 day).
    
    Returns:
    --------
    select_ts : list
        A list of pyMCDS objects representing the selected timesteps that match the given `step_time` interval.
    
    Processing Steps:
    -----------------
    1. The function scans the provided directory for XML files with filenames matching 'output*.xml'.
    2. It loads each file as a pyMCDS object.
    3. The `step_time` is retrieved from the keyword arguments (or defaults to 1440 minutes, equivalent to 1 day).
    4. The function filters the loaded timesteps, keeping only those that match a multiple of the `step_time` value.
    
    Example Usage:
    --------------
    # Load timesteps with a time interval of one day
    timesteps = load_mcds_timesteps('/path/to/simulation/directory', step_time=1440)
    
    Notes:
    ------
    - The function uses `pyMCDS` to load XML files, so the `pcdl` module must be imported and `pyMCDS` properly installed.
    - The `step_time` value should match the timestep intervals used in the simulation data.
    """
    
    # Load all timesteps from the directory, sorted by filename
    all_ts = [pcdl.pyMCDS(ts) for ts in sorted(glob.glob(directory.rstrip("/") + "/output*.xml"))]
    
    # Retrieve step_time from kwargs or use the default of 1440 (1 day)
    step_time = kwargs.get('step_time', 1440)  # Defaults to one day (1440 minutes)
    
    # Select only timesteps where the time is a multiple of step_time
    select_ts = [ts for ts in all_ts if ts.get_time() % step_time == 0]
    
    return select_ts

def analyze_abm_timestep(timestep, **kwargs):
    """
    Analyzes a given agent-based model (ABM) timestep and extracts properties of the largest contour.

    This function generates a grayscale image from cell locations and ECM secretions, binarizes it, fills in holes, 
    extracts the largest contour, and calculates properties such as radial distances and peaks. Optionally, the function 
    can plot various stages of image processing.

    Parameters:
    timestep (object): The timestep image from the agent-based model to be analyzed. 
                       It is expected to be an image array or similar object.
    **kwargs: Additional keyword arguments:
        - 'kernel_size' (int): The kernel size to be used in the grayscale conversion (default: 19).
        - 'plot' (bool): If True, plots the image at various stages of processing (default: False).
        - 'radial_distance_threshold' (float): A threshold for identifying peaks in radial distances (optional).

    Returns:
    tuple: A tuple containing:
        - dict: A dictionary of properties for the contour, such as radial distances and peak properties.
        - numpy.ndarray: The grayscale version of the timestep image.
        - numpy.ndarray: The binary mask generated from the largest contour.

    Example:
    >>> props, grayscale_img, mask = analyze_abm_timesteps(timestep, kernel_size=19, plot=True, radial_distance_threshold=0.5)
    >>> print(props)
    """
    kernel_size = kwargs.get('kernel_size', 19)        
    # Grayscale conversion and binarization
    grayscale_image = abm2gray(timestep, kernel_size)
    binarized = binarize_channel_otsu(grayscale_image)
    closed = ski.morphology.closing(binarized, footprint=ski.morphology.disk(5))
    filled = binary_fill_holes(closed, structure=np.ones((5,5))).astype(int)
    # Extract the largest contour and create a mask
    abm_contours = find_largest_contour_channel(filled)
    mask = create_mask_channel(grayscale_image.shape, abm_contours)
    segmented = segment_spheroid_channel(grayscale_image, mask)
    
    # Plot if the 'plot' argument is True
    if kwargs.get('plot', False):
        fig, axs = plt.subplots(2, 3, figsize=(12, 12))
        axs = axs.ravel()
        [ax.axis('off') for ax in axs]
        axs[0].imshow(grayscale_image, cmap='gray')
        axs[1].imshow(binarized, cmap='gray')
        axs[2].imshow(mask, cmap='gray')
        plot_contour(grayscale_image, abm_contour, axs[3])
        axs[4].imshow(segmented, cmap='gray')
        fig.subplots_adjust(wspace=0.2, hspace=0.1)
        plt.show()
        
    # Calculate radial distances and peaks for the contours
    if 'radial_distance_threshold' in list(kwargs.keys()):
        threshold = kwargs['radial_distance_threshold']
    else: 
        threshold = 0
    radial_distances, median_radial_distance = calculate_radial_distance(abm_contours)
    peaks, properties = calculate_peaks(radial_distances, threshold)
    # Calculate contour properties and return
    props = calc_contour_props(mask, radial_distances, median_radial_distance, peaks)
    return props, grayscale_image, mask

def analyze_abm_timesteps(timestep_list, plot=True, **kwargs):
    """
    Analyzes a list of timesteps of an agent-based model (ABM) by processing grayscale images, 
    extracting contours, generating binary masks, and calculating image properties for each timestep.
    
    Parameters:
    -----------
    timestep_list : list
        A list of timestep objects, where each timestep contains the image data and a time method.
    plot : bool, optional (default=True)
        If True, generates plots at each timestep showing various stages of image processing such as 
        grayscale images, binarized masks, and contour overlays.
    **kwargs : dict, optional
        Additional keyword arguments. Currently supports:
            - 'kernel_size': int
                The size of the kernel used for smoothing the ABM images. Default is 19.
    
    Returns:
    --------
    image_properties : dict
        A dictionary containing calculated properties of each image at each timestep. 
        The keys represent the time in hours for each timestep, and the values contain contour 
        properties, radial distances, median radial distances, and peak data.
    
    Processing Steps:
    -----------------
    1. Convert the ABM timestep image to grayscale using a Gaussian filter with the provided or default kernel size.
    2. Binarize the grayscale image using Otsu's method to separate the foreground (spheroid) from the background.
    3. Apply morphological closing to fill small gaps in the binary image.
    4. Fill holes in the binary image to ensure the object (spheroid) is fully enclosed.
    5. Extract the largest contour (representing the main spheroid).
    6. Create a binary mask based on the extracted contour.
    7. Segment the spheroid based on the grayscale image and the binary mask.
    
    For each timestep:
    ------------------
    - Optionally, plots are generated that show:
        1. Grayscale image
        2. Binarized image
        3. Binary mask
        4. Contour overlaid on the grayscale image
        5. Segmented spheroid
    - Radial distances of the contour points are calculated, and peaks in the radial distances are identified.
    - Properties of the contour and its relationship to the spheroid (radial distances, median distances, peaks) are calculated and stored.
    
    Example Usage:
    --------------
    image_props = analyze_abm_timesteps(timestep_list, plot=True, kernel_size=21)
    
    Notes:
    ------
    - The function assumes that the timestep object has a method `get_time()` that returns the time in minutes.
    - The function utilizes helper functions such as:
        - `abm2gray()`: Converts an ABM image to grayscale.
        - `binarize_channel_otsu()`: Binarizes the grayscale image using Otsu's thresholding.
        - `find_largest_contour_channel()`: Finds the largest contour in the binary image.
        - `create_mask_channel()`: Creates a binary mask based on the extracted contour.
        - `segment_spheroid_channel()`: Segments the spheroid in the grayscale image using the binary mask.
        - `calculate_radial_distance()`: Calculates the radial distances of the contour from the centroid.
        - `calculate_peaks()`: Identifies peaks in the radial distances.
        - `calc_contour_props()`: Computes various contour properties such as area, circularity, and peak information.
    """
    
    # Retrieve kernel size from kwargs or use default value
    if 'kernel_size' in kwargs.keys():
        kernel_size = kwargs['kernel_size']
        print(f'Kernel: {kernel_size}')
    else:
        kernel_size = 19
    
    grayscale_images = [None]*len(timestep_list)  # Placeholder for grayscale images
    abm_contours = [None]*len(timestep_list)  # Placeholder for contours
    masks = [None]*len(timestep_list)  # Placeholder for masks
    image_properties = {}  # Dictionary to store properties of each image
    median_radial_distance = None  # Placeholder for median radial distance

    # Process each timestep
    for i, ts in enumerate(timestep_list):
        simtime = ts.get_time() / 60  # Convert time from minutes to hours
        
        # Grayscale conversion and binarization
        grayscale_images[i] = abm2gray(ts, kernel_size)
        binarized = binarize_channel_otsu(grayscale_images[i])
        closed = ski.morphology.closing(binarized, footprint=ski.morphology.disk(5))
        filled = binary_fill_holes(closed, structure=np.ones((5,5))).astype(int)
        
        # Extract the largest contour and create a mask
        abm_contours[i] = find_largest_contour_channel(filled)
        masks[i] = create_mask_channel(grayscale_images[i].shape, abm_contours[i])
        segmented = segment_spheroid_channel(grayscale_images[i], masks[i])

        # Plot if the 'plot' argument is True
        if plot:
            fig, axs = plt.subplots(2, 3, figsize=(12, 12))
            axs = axs.ravel()
            [ax.axis('off') for ax in axs]
            axs[0].imshow(grayscale_images[i], cmap='gray')
            axs[1].imshow(binarized, cmap='gray')
            axs[2].imshow(masks[i], cmap='gray')
            plot_contour(grayscale_images[i], abm_contours[i], axs[3])
            axs[4].imshow(segmented, cmap='gray')
            fig.subplots_adjust(wspace=0.2, hspace=0.1)
            plt.show()
        
        # Calculate radial distances and peaks for the contours
        if i == 0:
            radial_distances, median_radial_distance = calculate_radial_distance(abm_contours[i])
        else:
            radial_distances, _ = calculate_radial_distance(abm_contours[i])
        
        peaks, properties = calculate_peaks(radial_distances, median_radial_distance)
        
        # Calculate contour properties and store them
        image_properties[f"timepoint-{simtime}"] = calc_contour_props(masks[i], radial_distances, median_radial_distance, peaks)

    return image_properties

def plot_radial_distance(contour, radial_distances, median_radial_distance, peaks):
    """
    Plots the radial distances of contour points and identifies peaks.

    This function visualizes the radial distances of points along a contour 
    from the center of mass. It displays the radial distances, the median 
    radial distance, and highlights the peaks on the plot.

    Parameters:
    contour (numpy.ndarray): A 2D array representing the contour points, 
                             typically of shape (n, 2) where `n` is the number 
                             of points.
    radial_distances (numpy.ndarray): An array of radial distances calculated 
                                       from the contour points.
    median_radial_distance (float): The median of the radial distances to be plotted.
    peaks (numpy.ndarray): An array of indices indicating the locations of peaks 
                           in the radial distances.

    Returns:
    None: This function does not return any value but generates a plot.

    Example:
    >>> plot_radial_distance(contour, radial_distances, median_radial_distance, peaks)
    """
    # Reshape the contour array
    contour = contour.reshape(-1, 2)

    # radian measurements
    radians = np.linspace(0, 2 * np.pi, num=len(radial_distances))
    
    # Create a plot
    plt.figure(figsize=(8, 6))

    # Plot the radial distances
    plt.plot(radians, radial_distances, label='Radial distance')

    # Plot the median radial distance as a dashed horizontal line
    plt.axhline(median_radial_distance, color='r', linestyle='--', label='Median radial distance')

    # Plot the peaks
    plt.scatter(radians[peaks], radial_distances[peaks], color='g', label='Peaks')

    # Set the title and labels
    plt.title('Radial Distance of Contour Points')
    plt.xlabel('Radians')
    plt.ylabel('Distance from Center of Mass')

    # Set x ticks at every pi/2 radians
    xticks = np.arange(0, 2.5 * np.pi, np.pi / 2)
    xticklabels = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
    plt.xticks(xticks, xticklabels)

    # Show the legend and plot
    plt.legend()
    plt.show()

def plot_contour(image, contour, ax):
    """
    Plot the largest contour and its convex hull overlaid on the original image.

    Parameters:
        image (numpy.ndarray): The original image, typically in grayscale.
        contour (numpy.ndarray): The largest contour points as an Nx1x2 array.

    Returns:
        None: This function displays the plot using matplotlib but does not return any values.
    """
    # Convert the grayscale image to BGR color space for color drawing
    oimage = ski.color.gray2rgb(image) #cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    oimage = oimage[:,:,::-1] # reverse color channels

    # Display the output image with contour and hull on the first subplot
    ax.imshow(oimage)
    ax.plot(contour[:, :, 0], contour[:, :, 1], linewidth=3)
    ax.axis('off')  # Hide the axis