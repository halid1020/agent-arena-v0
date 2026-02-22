import cv2
import numpy as np
import os
import json

def save_colour(img, filename='color', directory=".", rgb2bgr=False):
    os.makedirs(directory, exist_ok=True)
    if rgb2bgr:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    cv2.imwrite('{}/{}.png'.format(directory, filename), img_bgr)

def save_depth(depth, filename='depth', directory=".", colour=False):
    # Note: This function normalizes data, losing real-world scale.
    # To save absolute values, consider saving as .npy or 16-bit PNG instead.
    if np.max(depth) - np.min(depth) == 0:
        norm_depth = depth
    else:
        norm_depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
        
    if colour:
        norm_depth = cv2.applyColorMap(np.uint8(255 * norm_depth), cv2.COLORMAP_JET)
    else:
        norm_depth = np.uint8(255 * norm_depth)
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite('{}/{}.png'.format(directory, filename), norm_depth)

def save_mask(mask, filename='mask', directory="."):
    os.makedirs(directory, exist_ok=True)
    mask = (mask > 0).astype(np.uint8) * 255 
    cv2.imwrite(f'{directory}/{filename}.png', mask)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_): # <--- ADDED THIS LINE
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_action_json(action, filename, directory="."):
    """Saves action to JSON. Handles nested numpy arrays."""
    os.makedirs(directory, exist_ok=True)
    
    with open(f'{directory}/{filename}.json', 'w') as f:
        # cls=NumpyEncoder handles the conversion recursively
        json.dump(action, f, cls=NumpyEncoder, indent=4)

# --- NEW: Corresponding Load Functions ---

def load_colour(filename, directory=".", bgr2rgb=False):
    path = f'{directory}/{filename}.png'
    if not os.path.exists(path):
        return None
    
    img = cv2.imread(path)
    if img is None:
        return None
        
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_depth(filename, directory="."):
    path = f'{directory}/{filename}.png'
    if not os.path.exists(path):
        return None

    # Read as grayscale (unchanged depth)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None
        
    # Normalize back to 0-1 float to match typical depth usage
    return img.astype(np.float32) / 255.0

def load_mask(filename, directory="."):
    path = f'{directory}/{filename}.png'
    if not os.path.exists(path):
        return None

    # Read as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # Convert back to 0 or 1 integers
    return (img > 127).astype(np.uint8)

def load_action_json(filename, directory="."):
    path = f'{directory}/{filename}.json'
    if not os.path.exists(path):
        return None
        
    with open(path, 'r') as f:
        data = json.load(f)
    return np.array(data)