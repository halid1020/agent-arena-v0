## Python tutorial https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
## Elanation https://dev.intelrealsense.com/docs/post-processing-filters
########################################################################
########################################################################
## To run this script you need to have the following packages installed in advanced:
## 1. install ag_ar package using the conda environment and activate it.
## 2. pip install segment-anything==1.0.0
## 3. pip install pyrealsense2==2.54.2.5684
## 4. run `python advanced_model.py` to load the pre-tuned configuration `depth_camera_config.json`
## 5. run python realsense_test_display_rgbd.py to display the camera feed and make sure the depth is stable and smooth
##    otherwise, one need to open realsense_viewer to fine-tune the depth camera settings and reload the configuration.
## 6. make sure the model checkpoint of the agent is saved in the log directory.



import os
from tqdm import tqdm
import pyrealsense2 as rs
import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from matplotlib import pyplot as plt
import api as ag_ar
from gym.spaces import Box

import torch
from segment_anything import sam_model_registry
from src.utilities.visualisation_utils import draw_pick_and_place
from src.utilities.visualisation_utils import plot_image_trajectory


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### Camera Macros ###
colorizer = rs.colorizer()
decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude, 4) #4
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)
spatial = rs.spatial_filter()
spatial.set_option(rs.option.holes_fill, 3) #3
spatial.set_option(rs.option.filter_magnitude, 5) #5
spatial.set_option(rs.option.filter_smooth_alpha, 1) #1
spatial.set_option(rs.option.filter_smooth_delta, 50) #50
hole_filling = rs.hole_filling_filter()
temporal = rs.temporal_filter()

### Masking Model Macros ###
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint='sam_vit_h_4b8939.pth')
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

## Depth Mapping Macros ##
Depth_max = 1
Depth_min = 0.7

depth_base = 0.91 ## This value needs to be fine-tuned based on the camera setting
depth_top = 0.785 ## This value needs to be fine-tuned based on the camera setting

thick_scale = 0.04 # decided by the input distribution of planet-clothpick
depth_target_min = 1.5 - thick_scale # decided by the trianing setting of planet-clothpick

def bilinear_interpolation(x, y, x1, y1, x2, y2, q11, q21, q12, q22):
    """
    Perform bilinear interpolation.
    
    Parameters:
        x, y: Coordinates of the target point.
        x1, y1, x2, y2: Coordinates of the four corners.
        q11, q21, q12, q22: Values at the four corners.
        
    Returns:
        Interpolated value at the target point.
    """
    denom = (x2 - x1) * (y2 - y1)
    w11 = (x2 - x) * (y2 - y) / denom
    w21 = (x - x1) * (y2 - y) / denom
    w12 = (x2 - x) * (y - y1) / denom
    w22 = (x - x1) * (y - y1) / denom
    
    interpolated_value = q11 * w11 + q21 * w21 + q12 * w12 + q22 * w22
    return interpolated_value


def interpolate_image(height, width, corner_values):
    interpolated_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            x = i/height
            y = j/width
            x1 = int(x)
            y1 = int(y)
            x2 = x1 + 1
            y2 = y1 + 1
            q11 = corner_values[(x1, y1)]
            q21 = corner_values[(x2, y1)]
            q12 = corner_values[(x1, y2)]
            q22 = corner_values[(x2, y2)]
            interpolated_image[i, j] = \
                bilinear_interpolation(x, y, x1, y1, x2, y2, q11, q21, q12, q22)
    return interpolated_image


def plane_transform(color_data, depth_data, mask):

    color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
    depth_data = depth_data.astype(np.float32)/1000.0

    


    H, W = depth_data.shape
    
    ## get the 4 corners of the depth data (x, y, z)
    top_left = [0, 0, depth_data[0, 0]]
    top_right = [1, 0, depth_data[-1, 0]]
    bottom_left = [0, 1, depth_data[0, -1]]
    bottom_right = [1, 1, depth_data[-1, -1]]

    ## get the average depth of the 4 corners
    average_depth = (top_left[2] + top_right[2] + bottom_left[2] + bottom_right[2])/4.0
    print('avearge_depth', average_depth)
    print('min raw depth:', np.min(depth_data))
    print('max raw depth:', np.max(depth_data))

    
    ## create a ground truth depth, where x, y has depth top_left + (1-x) * (top_right - top_left) + y * (bottom_left - top_left)
    corner_values = {(0, 0): top_left[2], (1, 0): top_right[2], (0, 1): bottom_left[2], (1, 1): bottom_right[2]}
    ground_depth = interpolate_image(H, W, corner_values)

    if args.store_interm:
        ground_depth_ = (ground_depth - np.min(ground_depth))/(np.max(ground_depth) - np.min(ground_depth))
        ground_depth_ = cv2.applyColorMap(np.uint8(255 * ground_depth_), cv2.COLORMAP_JET)
        cv2.imwrite('depth_ground.png', ground_depth_)

    transform_depth = (depth_data + 0.01) - ground_depth + average_depth
    print('min transform depth:', np.min(transform_depth))
    print('max transform depth:', np.max(transform_depth))

    transform_depth[(mask == False)] = depth_base
    
    return transform_depth

def postprocess_camera(render=False, steps=20, re_mask=True):
    for i in range(steps):
        frames = pipeline.wait_for_frames()

        color_data = np.asanyarray(frames.get_color_frame().get_data())
        H, W = color_data.shape[:2]
        # Convert images to numpy arrays
        depth_frame = frames.get_depth_frame()
        raw_depth = np.asanyarray(depth_frame.get_data())

        depth_frame = decimation.process(depth_frame)
        depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        depth_data = np.asanyarray(depth_frame.get_data())
        depth_data = cv2.resize(depth_data, (W, H))
        #depth_post = depth_data.copy()
        
        
        CH = int(H*0.65)
        CW = int(W*0.65)
        BH = 85
        BW = 85
        raw_depth = raw_depth[BH:BH+CH, BW:BW+CW]
        depth_data = depth_data[BH:BH+CH, BW:BW+CW]
        depth_data = cv2.resize(depth_data, (W, H))
        aligned_depth = depth_data.copy()
        
        mask = None
        adjusted_depth = None

        if re_mask and i == steps - 1:
            result = mask_generator.generate(color_data)
            
            fmask = None
            max_val = 0
            ## print all the stablity scores
            for r in result:
                mask = r['segmentation'].copy()
                tmp_mask = mask[20:-20, 20:-20].copy()
                mask = mask.reshape(*color_data.shape[:2], -1)
                #print('mask 0 0', mask[0][0])
                corners = (not tmp_mask[0][0]) + (not tmp_mask[0][-1]) + (not tmp_mask[-1][0]) + (not tmp_mask[-1][-1])
                
                if np.sum(corners) >= 2 and np.sum(corners) <= 4:
                    if max_val < np.sum(tmp_mask):
                        fmask = mask
                        max_val = np.sum(tmp_mask)
                    
            mask = fmask.reshape(*color_data.shape[:2])
            adjusted_depth = plane_transform(color_data, depth_data, mask)
    
    if args.store_interm:
        ## normalisd depth and give it a color map
        raw_depth = raw_depth.clip(np.min(aligned_depth), np.max(aligned_depth))
        raw_depth = (raw_depth - np.min(raw_depth))/(np.max(raw_depth) - np.min(raw_depth))
        raw_depth = cv2.applyColorMap(np.uint8(255 * raw_depth), cv2.COLORMAP_JET)
        cv2.imwrite('depth_raw.png', raw_depth)

        cv2.imwrite('color_raw.png', color_data)
        
        adjusted_depth_ = (adjusted_depth - np.min(adjusted_depth))/(np.max(adjusted_depth) - np.min(adjusted_depth))
        adjusted_depth_ = cv2.applyColorMap(np.uint8(255 * adjusted_depth_), cv2.COLORMAP_JET)
        cv2.imwrite('depth_adjust.png', adjusted_depth_)


        aligned_depth = (aligned_depth - np.min(aligned_depth))/(np.max(aligned_depth) - np.min(aligned_depth))
        aligned_depth = cv2.applyColorMap(np.uint8(255 * aligned_depth), cv2.COLORMAP_JET)
        cv2.imwrite('depth_post.png', aligned_depth)

        # depth_post = (depth_post - np.min(depth_post))/(np.max(depth_post) - np.min(depth_post))
        # depth_post = cv2.applyColorMap(np.uint8(255 * depth_post), cv2.COLORMAP_JET)
        # cv2.imwrite('depth_post.png', depth_post)

        cv2.imwrite('mask.png', mask.astype(np.uint8)*255)



    return color_data, adjusted_depth, mask, depth_data

def get_quasi_static_observation(render=False, resolution=(128, 128)):
    
    # Wait for a coherent pair of frames: depth and color
    rgb, depth, mask, raw_depth = postprocess_camera(render=render, steps=5)
    

    ## crop the image with the given resolution

    H, W = rgb.shape[:2]
    min_val = min(H, W)
    mid_x = W//2
    mid_y = H//2
    rgb = rgb[mid_y - min_val//2:mid_y + min_val//2, mid_x - min_val//2:mid_x + min_val//2]
    depth = depth[mid_y - min_val//2:mid_y + min_val//2, mid_x - min_val//2:mid_x + min_val//2]
    mask = mask[mid_y - min_val//2:mid_y + min_val//2, mid_x - min_val//2:mid_x + min_val//2]
    raw_depth = raw_depth[mid_y - min_val//2:mid_y + min_val//2, mid_x - min_val//2:mid_x + min_val//2]



    rgb = cv2.resize(rgb, resolution)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.resize(depth, resolution)   
    raw_depth = cv2.resize(raw_depth, resolution)
    mask = cv2.resize(mask.astype(np.float), resolution)
    mask = (mask > 0.2).astype(np.bool8)

    #min_depth = 0.786
    depth = depth.clip(depth_top, depth_base)
    new_depth = (depth - depth_top)/(depth_base - depth_top)*thick_scale + depth_target_min



    return {
        'rgb': rgb,
        'depth': new_depth.reshape(*resolution, 1),
        'mask': mask,
        'raw_depth': raw_depth
    }

import argparse
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--arena', default='softgym|domain:rainbow-rect-fabric,initial:crumple,action:pixel-pick-and-place(1),task:flattening')
    parser.add_argument('--agent', default='planet-clothpick')
    parser.add_argument('--config', default='D2M')
    parser.add_argument('--log_dir', default='/data/fast-ldm-fabric-shaper')
    parser.add_argument('--store_interm', action='store_true', help='store intermediate results')
    parser.add_argument('--eval_checkpoint', default=-1, type=int)


    return parser.parse_args()

def run_experiments(args, agent):
    real_depth_images = []
    real_rgb_images = []
    real_action_images = []
    real_mask_images = []
    real_input_obs = []
    real_raw_depth_images = []
    k = -1
    while True:
        user_input = input("Press [y/n] to continue for step {}: ".format(k+1))
        if user_input == 'n':
            observation= get_quasi_static_observation(render=True, resolution=resolution)
            break
        elif user_input != 'y':
            continue
        else:
            k += 1


        print('Getting the quasi-static depth...')
        resolution = (256, 256)
        observation= get_quasi_static_observation(render=True, resolution=resolution)


        state = {
            'observation': {
                'rgb': observation['rgb'].copy(),
                'depth': observation['depth'].copy(),
                'mask': observation['mask'].copy()
            },
            'action_space': Box(
                -np.ones((1, 4)).astype(np.float32),
                np.ones((1, 4)).astype(np.float32),
                dtype=np.float32),
        }
        if k == 0:
            agent.set_eval()
            agent.reset()
            agent.init(state)
        else:
            agent.update(state, action)
        
        print('Getting Action ...')
        action = agent.act(state)
        int_res = agent.get_state()
        input_obs = int_res['input_obs'].copy().astype(np.uint8)
        ## convert bgr to rgb
        #input_obs = cv2.cvtColor(input_obs, cv2.COLOR_BGR2RGB)
        
        real_input_obs.append(input_obs)
        pixel_actions = ((action + 1)/2*resolution[0]).astype(int).reshape(4)


        image = draw_pick_and_place(
            observation['rgb'], 
            tuple(pixel_actions[:2]), 
            tuple(pixel_actions[2:]),
            color=(255, 0, 0))

        H, W = observation['rgb'].shape[:2]
        real_depth_images.append(observation['depth'].reshape(H, W, 1))
        real_rgb_images.append(observation['rgb'])
        real_mask_images.append(observation['mask'].reshape(H, W, 1))
        real_action_images.append(image)
        real_raw_depth_images.append(observation['raw_depth'].reshape(H, W, 1))
        
        ### plot mask, rgb, depth, and the action annotation together in one plot
        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(observation['rgb'])
        ax[0, 0].set_title('RGB')
        ax[0, 1].imshow(observation['depth'])
        ax[0, 1].set_title('Processed Depth')
        ax[1, 0].imshow(observation['mask'])
        ax[1, 0].set_title('Mask')
        ax[1, 1].imshow(image)
        ax[1, 1].set_title('Action')
        ax[0, 2].imshow(input_obs)
        ax[0, 2].set_title('Input Obs')
        ax[1, 2].imshow(observation['raw_depth'])
        ax[1, 2].set_title('Raw Depth')
        ### get rid of the axis for all
        for i in range(2):
            for j in range(3):
                ax[i, j].axis('off')
        plt.show()

    # images = real_rgb_images + real_raw_depth_images+ real_mask_images + \
    #     real_depth_images +  real_input_obs + real_action_images
    real_action_images.append(observation['rgb'])
    save_dir = os.path.join(args.log_dir, 'real-trials', args.agent, args.config)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    traj_num = int(input("Enter the trajectory number: "))
    
    # plot_image_trajectory(images,
    #     save_path=save_dir,
    #     title='trajectory_{}'.format(str(traj_num)), 
    #     col=len(real_rgb_images),
    #     row_lables=['RGB', 'Raw Depth', 'Mask', 'Depth', 'Input Obs', 'Action'])

    images = real_action_images + real_input_obs 
    
    plot_image_trajectory(images,
        save_path=save_dir,
        title='trajectory_{}'.format(str(traj_num)), 
        col=len(real_action_images),
        row_lables=['Action', 'Input'])


if __name__ == "__main__":
    
    
    ### Initialize the camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    
    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
        pipeline.wait_for_frames()

    args = parse_arguments()
    ### Initialise Agent ####
    agent_config = ag_ar.retrieve_config(
        args.agent, 
        args.arena, 
        args.config,
        args.log_dir)
    
    agent = ag_ar.build_agent(
        args.agent,
        config=agent_config)

    if args.eval_checkpoint != -1:
        agent.load_checkpoint(args.eval_checkpoint)
    else:
        agent.load()

    
    
    try:
        while True:

            continue_exp = input("Do you want to run an experiment? [y/n]: ")
            if continue_exp == 'n':
                break
            elif continue_exp != 'y':
                continue
            run_experiments(args, agent)

                
    finally:
        # Stop streaming
        pipeline.stop()