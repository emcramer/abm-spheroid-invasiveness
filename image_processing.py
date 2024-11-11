import skimage as ski
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import scipy as sp

def binarize_channel_local_otsu(image, radius:int=15):
    img = ski.util.img_as_ubyte(image)
    radius = 15
    footprint = ski.morphology.disk(radius)
    local_otsu = ski.rank.otsu(img, footprint)
    return(local_otsu)

def binarize_channel_local(image, block_size:int=155, offset:int=5):
    local_thresh = ski.filters.threshold_local(image, block_size=block_size, offset=offset)
    binary_local = image > local_thresh
    return binary_local

def binarize_channel_gmm(image):
    # Flatten the image to 1D array for GMM
    reshaped = image.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(reshaped)
    gmm_labels = gmm.predict(reshaped)
    # Threshold image
    background_label = np.argmin(gmm.means_)
    return (gmm_labels != background_label).reshape(image.shape).astype(np.uint8)

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

def find_largest_contour(cleaned_image):
    contours = []
    n_channels = cleaned_image.shape[0]
    for channel in range(n_channels):
        contours.append(find_largest_contour_channel(cleaned_image[channel,:,:]))
    return contours
        
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