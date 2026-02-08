import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def get_wrinkle_pixel_ratio(rgb, mask):
    
    rgb = cv2.resize(rgb, (128, 128))
    #mask =  cv2.resize(mask, (128, 128)) 
    

    if mask.dtype != np.uint8:  # Ensure mask has a valid data type (uint8)
        mask = mask.astype(np.uint8)


    # Use cv2 edge detection to get the wrinkle ratio.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # plt.imshow(edges)
    # plt.show()

    masked_edges = cv2.bitwise_and(edges, mask)
    # plt.imshow(masked_edges)
    # plt.show()

    wrinkle_ratio = np.sum(masked_edges) / np.sum(mask)

    return wrinkle_ratio

def get_canonical_IoU(mask, canonical_mask):
    intersection = np.sum(np.logical_and(mask, canonical_mask))
    union = np.sum(np.logical_or(mask, canonical_mask))
    return intersection/union

def get_canonical_hausdorff_distance(mask, canonical_mask):
    hausdorff_distance = directed_hausdorff(mask, canonical_mask)[0]

    return hausdorff_distance