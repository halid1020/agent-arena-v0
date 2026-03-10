import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F


from actoris_harena.utilities.networks.utils import np_to_ts, ts_to_np

def gaussian_kernel(kernel_size, sigma):
    x = torch.linspace(-sigma, sigma, kernel_size)
    kernel_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    return kernel_2d / kernel_2d.sum()

def zoom_image(image, zoom_factor, center=None):
    """
    Zoom in/out an image without changing its size by cropping.

    Args:
    - image: PyTorch tensor representing the image (shape: C x H x W)
    - zoom_factor: Zoom factor (>1 for zooming in, <1 for zooming out)
    - center: Coordinates of the center of the zoomed region (default: None, which zooms from the image center)

    Returns:
    - zoomed_image: PyTorch tensor representing the zoomed image
    """

    ## if image is in numpy
    convert = False
    if isinstance(image, np.ndarray):
        convert = True
        image = np_to_ts(image, 'cpu')
    
    permute = False
    if image.shape[2] <= 3:
        permute = True
        image = image.permute(2, 0, 1)


    C, H, W = image.shape

    #print('zoom shape', image.shape)

    # Calculate the new dimensions
    new_H = int(H * zoom_factor)
    new_W = int(W * zoom_factor)

    # Resize the image
    resized_image = F.interpolate(image.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)

    if zoom_factor > 1:
        # Zooming in: Crop the center of the resized image
        if center is None:
            center = (new_W // 2, new_H // 2)
        left = max(0, center[0] - W // 2)
        top = max(0, center[1] - H // 2)
        zoomed_image = resized_image[:, top:top + H, left:left + W]
    else:
        # Zooming out: Place the resized image in the center of a black canvas
        zoomed_image = torch.zeros_like(image)
        if center is None:
            center = (W // 2, H // 2)
        left = max(0, center[0] - new_W // 2)
        top = max(0, center[1] - new_H // 2)
        zoomed_image[:, top:top + new_H, left:left + new_W] = resized_image
    
    if permute:
        zoomed_image = zoomed_image.permute(1, 2, 0)
    
    if convert:
        zoomed_image = ts_to_np(zoomed_image)

    return zoomed_image

def inverse_normalize(array, mean, std):
    """Apply the inverse of Normalize transform.
    
    Args:
        array (torch.Tensor or np.ndarray): The normalized tensor or array.
        mean (list): The mean used for normalization (for each channel).
        std (list): The standard deviation used for normalization (for each channel).
    
    Returns:
        torch.Tensor or np.ndarray: The array after applying the inverse normalization.
    """
    if isinstance(array, torch.Tensor):
        mean = torch.as_tensor(mean, dtype=array.dtype, device=array.device)
        std = torch.as_tensor(std, dtype=array.dtype, device=array.device)
        if array.ndim == 3:  # If a single image tensor, add a batch dimension
            array = array.unsqueeze(0)
        array.mul_(std[:, None, None]).add_(mean[:, None, None])
    elif isinstance(array, np.ndarray):
        mean = np.array(mean, dtype=array.dtype)
        std = np.array(std, dtype=array.dtype)
        if array.ndim == 3:  # If a single image array, add a batch dimension
            array = np.expand_dims(array, axis=0)
        array = (array * std) + mean
    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray")
    
    return array
## Expecting tensor rgb
