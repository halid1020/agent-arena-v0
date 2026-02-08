import cv2
import numpy as np
import os
def save_colour(img, filename='color', directory=".", rgb2bgr=True):
    os.makedirs(directory, exist_ok=True)
    if rgb2bgr:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
        
    cv2.imwrite('{}/{}.png'.format(directory, filename), img_bgr)

def save_depth(depth, filename='depth', directory=".", colour=False):
    depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
    if colour:
        depth = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
    else:
        depth = np.uint8(255 * depth)
    os.makedirs(directory, exist_ok=True)
    #print('save')
    cv2.imwrite('{}/{}.png'.format(directory, filename), depth)

def save_mask(mask, filename='mask', directory="."):
    os.makedirs(directory, exist_ok=True)
    mask = (mask > 0).astype(np.uint8) * 255  # convert to 0 or 255 uint8
    cv2.imwrite(f'{directory}/{filename}.png', mask)