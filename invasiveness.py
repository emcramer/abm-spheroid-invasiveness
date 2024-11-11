import skimage as ski
import numpy as np
import scipy as sp

from image_processing import *

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