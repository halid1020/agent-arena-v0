import os

import math
import numpy as np

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from moviepy.editor import ImageSequenceClip



def draw_pick_and_place_noise_actions(
        input_obs, noise_actions, 
        filename='noise_actions.png', directory='./tmp',
        draw_noise=True,
        draw_trj=False,
        draw_arrow=True):
    # Make a copy of the input image to avoid modifying the original
    image = input_obs.copy()
    H, W = image.shape[:2]
    ## if H and W is too small, we need to scale it up
    if H < 500:
        ratio = int(500 / H)
        image = cv2.resize(image, (W*ratio, H*ratio))
        H, W = image.shape[:2]
    noise_actions = ((noise_actions + 1.0)/2  * np.asarray([H, W, H, W]).reshape(1, 4)).astype(int)
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Define start and end colors for the gradients
    place_start_color = np.array([255, 255, 255])  # light red
    place_end_color = np.array([255, 0, 0])  # Red for pick points
    pick_start_color = np.array([255, 255, 255])   # light green
    pick_end_color = np.array([0, 128, 0])  # Green for place points
    
    # Sample actions with even intervals
    if draw_noise:
        num_total_actions = len(noise_actions)
        plt_actions_num = 20
        if num_total_actions <= plt_actions_num:
            sampled_actions = noise_actions
        else:
            indices = np.linspace(0, num_total_actions - 1, plt_actions_num, dtype=int)
            sampled_actions = [noise_actions[i] for i in indices]
        
        # Adjust sizes based on the image dimensions
        image_size = min(image.shape[0], image.shape[1])
        circle_radius = max(3, int(image_size / 100))
        cross_length = max(5, int(image_size / 50))
        
        if draw_trj:
            images = []
            sampled_actions = noise_actions[-100:]
            for i, action in enumerate(sampled_actions):
                # Calculate color based on the action index
                #t = i / (plt_actions_num-1)
                pick_color = tuple(map(int, pick_end_color))
                place_color = tuple(map(int, place_end_color))
                
                # Assuming action is in the format [start_x, start_y, end_x, end_y]
                pick_point = (int(action[0]), int(action[1]))
                place_point = (int(action[2]), int(action[3]))
                
                tmp_image = image.copy()
                # Draw a circle for the pick point
                cv2.circle(tmp_image, pick_point, circle_radius, pick_color, -1)
                
                # Draw a cross for the place point
                draw_cross(tmp_image, place_point, cross_length, place_color)

                ## Daw number i with white color on the top left corner
                cv2.putText(tmp_image, str(100-i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                images.append(tmp_image)
            return np.stack(images)
        
        else:
        # Iterate through sampled actions
            for i, action in enumerate(sampled_actions):
                # Calculate color based on the action index
                t = i / (plt_actions_num-1)
                pick_color = tuple(map(int, (1-t)*pick_start_color + t*pick_end_color))
                place_color = tuple(map(int, (1-t)*place_start_color + t*place_end_color))
                
                # Assuming action is in the format [start_x, start_y, end_x, end_y]
                pick_point = (int(action[0]), int(action[1]))
                place_point = (int(action[2]), int(action[3]))
                
                # Draw a circle for the pick point
                cv2.circle(image, pick_point, circle_radius, pick_color, -1)
                
                # Draw a cross for the place point
                draw_cross(image, place_point, cross_length, place_color)
    
    ### draw the last action with draw_pick_and_place
    if draw_arrow:
        image = draw_pick_and_place(
            image, 
            tuple(noise_actions[-1][:2]), 
            tuple(noise_actions[-1][2:]),
            (0, 150, 255)
        ).get().clip(0, 255).astype(np.uint8)

    # Save the image
    output_path = os.path.join(directory, filename)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    #print('Finish')
    
    
    return image

def draw_cross(image, center, length, color):
    x, y = center
    cv2.line(image, (x - length, y), (x + length, y), color, 2)
    cv2.line(image, (x, y - length), (x, y + length), color, 2)

# input image is in [0, 255] numpy int type, H, W, 3
# start is two element tuple (x, y)
# end is two element tuple (x, y)
def draw_pick_and_place(image, start, end, color=(143, 201, 58), get_ready=False, swap=False):
    ## adjust thickness regarding to the image size
    thickness = max(1, int(image.shape[0] / 100))
    # print('start', start, 'end', end)
    # print('image shape', image.shape)

    start = (int(start[1]), int(start[0]))
    end = (int(end[1]), int(end[0]))
    
    if swap:
        start = (int(start[1]), int(start[0]))
        end = (int(end[1]), int(end[0]))
    
    image = cv2.arrowedLine(
        cv2.UMat(image), 
        start, 
        end,
        color, 
        thickness)
    
    image = cv2.circle(
        cv2.UMat(image), 
        start, 
        thickness*2,
        color, 
        thickness)
    
    if get_ready:
        return image.get().clip(0, 255).astype(np.uint8)

    return image #.get().astype(np.int8).clip(0, 255)

def plot_pick_and_place_trajectory(obs, acts, 
    info=None,
    info_color='white',
    info_font=20,
    save_png=True, save_path='', 
    action_color= [(143, 201, 58), (0, 114, 187)], 
    title='trajectory', show=False, col = 10):

    row = math.ceil(len(obs)/col)
    T, H, W, C = obs.shape
    #print('obs shape', obs.shape)
    fig = plt.figure(figsize=(5*col, 5*row))
    outer = fig.add_gridspec(ncols=1, nrows=1)
    inner = gridspec.GridSpecFromSubplotSpec(row, col, # TODO: magic number
                subplot_spec=outer[0], wspace=0, hspace=0)
    
    #print('acts', acts)
    
    act_len = acts.shape[0]
    acts =  acts.reshape(act_len, -1, 4)
    pick_num = acts.shape[1]

    pixel_actions = ((acts + 1.0)/2  * np.asarray([H, W, H, W]).reshape(1, 1, 4)).astype(int)
    
    
    # if acts1 is not None:
    #     pixel_actions_1 = ((acts1 + 1)/2  * np.asarray([H, W, H, W])).astype(int)
    # if acts2 is not None:
    #     pixel_actions_2 = ((acts2 + 1)/2  * np.array([H,W, H, W])).astype(int)

    thickness = 2

    for i in range(len(obs)):

        ####### 
        ax0 = plt.Subplot(fig, inner[i])
        
        image = obs[i]

        if C >= 3:
            image = image[:, :, :3].astype(int).clip(0, 255)
            if i < len(obs) - 1:
                if acts is not None:
                    #print('image shape', image.shape)
                    image = draw_pick_and_place(
                        image[:, :, :3],
                        tuple(pixel_actions[i][0][:2]), 
                        tuple(pixel_actions[i][0][2:]),
                        get_ready=True
                    )
                    if pick_num == 2:
                        image = cv2.arrowedLine(
                            cv2.UMat(image[:, :, :3].astype(int)), 
                            tuple(pixel_actions[i][1][:2]), 
                            tuple(pixel_actions[i][1][2:]),
                            action_color[1], 
                            thickness)
                        image = image.get().astype(int).clip(0, 255)
        
        ax0.axis('on')
        ax0.imshow(image)
        ax0.set_xticks([])
       
        # write the info as text on the corresponding image
        if info is not None:
            ax0.text(0.05, 0.95, info[i], fontsize=info_font, color=info_color, 
             transform=ax0.transAxes, verticalalignment='top')


        ax0.set_yticks([])
        fig.add_subplot(ax0)
    
    if show:
        plt.show()

    if save_png:
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        plt.savefig(os.path.join(save_path, '{}.png'.format(title)), bbox_inches='tight')
    
    plt.close()

## Purely polit image trajectory no action involved
def plot_image_trajectory(obs, 
    save_png=True, save_path='.',
    title='trajectory', show=False, col=10,
    row_lables=None):

    #col = col
    row = math.ceil(len(obs)/col)
    #T, H, W, C = obs.shape
    fig = plt.figure(figsize=(5*col, 5*row))
    outer = fig.add_gridspec(ncols=1, nrows=1)
    inner = gridspec.GridSpecFromSubplotSpec(row, col, # TODO: magic number
                subplot_spec=outer[0], wspace=0, hspace=0)
    
    for i in range(len(obs)):

        ####### 
        ax0 = plt.Subplot(fig, inner[i])
        
        image = obs[i]
        H, W, C = image.shape

        if C >= 3:
            image = image[:, :, :3].astype(int).clip(0, 255)
        
        ax0.axis('on')
        ax0.imshow(image)
        ax0.set_xticks([])
        ax0.set_yticks([])
        fig.add_subplot(ax0)

    # if row_lables is not None:
    #     for i in range(len(row_lables)):
    #         fig.text(0.14, 0.87 - i*0.128, row_lables[i], 
    #                  ha='left',
    #                  va='center', 
    #                  fontsize=25, 
    #                  color='red', fontweight='bold')
    
    if show:
        plt.show()

    if save_png:
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        plt.savefig(os.path.join(save_path, '{}.png'.format(title)), bbox_inches='tight')
    
    plt.close()


### frames: S * H * W * 3 in RGB numpy, or list of H*W*3 RGB numpys
def save_video(frames, path='', title='default'):

    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)  # shape (T, H, W, C)

    frames = frames.clip(0, 255).astype(np.uint8)
    bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

    if not os.path.exists(path):
        os.makedirs(path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    _, H, W, _ = frames.shape
    writter = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), fourcc, 30, (W, H))
    for frame in bgr_frames:
        writter.write(frame)
    writter.release()

def show_image(img, window_name=''):
    # Input image has to be displayable
    # The input type is either np int8 [0, 255] or np float [0, 1.0]
    # H*W*3 or H*W
    cv2.imshow(window_name, img)
    cv2.waitKey(1)



def save_numpy_as_gif(frames, path, filename, fps=200, scale=1.0, select_frames=1000):
    # from https://github.com/Xingyu-Lin/softgym/blob/master/softgym/utils/visualization.py
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    #fname, _ = os.path.splitext(filename)
    filename = filename + '.gif'
    filename = os.path.join(path, filename)
    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)  # shape (T, H, W, C)

    # copy into the color dimension if the images are black and white
    if frames.ndim == 3:
        frames = frames[..., np.newaxis] * np.ones(3)
    ## select 1000 frames across the whole frames includig the beginning and the end
    select_frames = min(select_frames, len(frames))
    frames = frames[::len(frames)//select_frames]


    # make the moviepy clip
    clip = ImageSequenceClip(list(frames), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip



def make_grid(array, nrow=1, padding=0, pad_value=120):
    # from https://github.com/Xingyu-Lin/softgym/blob/master/softgym/utils/visualization.py
    """ numpy version of the make_grid function in torch. Dimension of array: NHWC """
    if len(array.shape) == 3:  # In case there is only one channel
        array = np.expand_dims(array, 3)
    N, H, W, C = array.shape
    assert N % nrow == 0
    ncol = N // nrow
    idx = 0
    grid_img = None
    for i in range(nrow):
        row = np.pad(array[idx], [[padding if i == 0 else 0, padding], [padding, padding], [0, 0]], constant_values=pad_value)
        for j in range(1, ncol):
            idx += 1
            cur_img = np.pad(array[idx], [[padding if i == 0 else 0, padding], [0, padding], [0, 0]], constant_values=pad_value)
            row = np.hstack([row, cur_img])
        if i == 0:
            grid_img = row
        else:
            grid_img = np.vstack([grid_img, row])
    return grid_img


import open3d as o3d
def save_pointclouds(points):        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.random.rand(points.shape[0], 3)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud("pointcloud.pcd", pcd)

def filter_small_masks(mask, min_area):
    """
    Filter out small masks from a given input mask.
    
    Parameters:
    mask (numpy.ndarray): Input mask. Can be bool, uint8, or float.
    min_area (int): Minimum area threshold for keeping a contour.
    
    Returns:
    numpy.ndarray: Filtered mask of the same type and shape as input.
    """
    # Check mask type and convert if necessary
    org_mask = mask.copy()
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.dtype == np.float32 or mask.dtype == np.float64:
        mask = (mask * 255).astype(np.uint8)
    elif mask.dtype != np.uint8:
        raise ValueError("Unsupported mask dtype. Use bool, uint8, or float.")

    # Ensure mask is 2D
    if len(mask.shape) > 2:
        if mask.shape[2] != 1:
            raise ValueError("Mask must be 2D or have only one channel.")
        mask = mask.squeeze()

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask
    filtered_mask = np.zeros_like(mask)
    
    # Filter contours based on area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Convert back to original dtype
    if org_mask.dtype == bool:
        filtered_mask = (filtered_mask/255).astype(bool)
    elif org_mask.dtype == np.float32 or org_mask.dtype == np.float64:
        filtered_mask = filtered_mask.astype(org_mask.dtype) / 255.0
    
    return filtered_mask

def save_color(img, filename='color', directory="."):
    cv2.imwrite('{}/{}.png'.format(directory, filename), img)

def save_depth(depth, filename='depth', directory="."):
    depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
    depth = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
    cv2.imwrite('{}/{}.png'.format(directory, filename), depth)