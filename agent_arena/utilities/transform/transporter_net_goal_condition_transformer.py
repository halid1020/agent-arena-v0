import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import random

from agent_arena.agent.utilities.torch_utils import np_to_ts, ts_to_np
from agent_arena.utilities.transform.utils import *
from agent_arena.utilities.visual_utils import draw_pick_and_place

class TransporterNetGoalConditionTransformer:
   
    def __init__(self,  config=None):
        self.config = config
        if 'color_jitter' in config:
            self.color_jitter = torchvision.transforms.ColorJitter(
                **self.config.color_jitter
            )
        if 'grayscale' in config and config.grayscale:
            self.to_gray = torchvision.transforms.Grayscale(num_output_channels=1)
    

    def __call__(self, sample, train=True):
        print ('sample', sample.keys())
        
        sample = {k: np_to_ts(v, self.config.device) for k, v in sample.items()}
        for k, v in sample.items():
            if k in ['depth', 'mask', 'contour'] and len(v.shape) == 2:
                sample[k] = v.unsqueeze(2)
            if k in ['goal_depth', 'goal_mask', 'goal_contour'] and len(v.shape) == 2:
                sample[k] = v.unsqueeze(2)
        
        

        for k, v in sample.items():
            if k in ['rgb', 'color' 'depth', 'mask',
                     'goal_rgb', 'goal_depth', 'goal_mask',
                     ] and tuple(v.shape[:2]) != tuple(self.config.img_dim):
                
                # v is in shape (H, W, C), I want to resize it to (H', W', C)
                sample[k] = F.interpolate(
                    v.permute(2, 0, 1).unsqueeze(0), 
                    self.config.img_dim, 
                    mode='bilinear', 
                    align_corners=False).squeeze(0).permute(1, 2, 0)
        
        

        if 'rgb' not in sample and 'color' in sample:
            sample['rgb'] = sample['color']
        
        if 'goal_rgb' not in sample and 'goal_color' in sample:
            sample['goal_rgb'] = sample['goal_color']

        # if 'depth' in sample and len(sample['depth'].shape) == 2:
        #     sample['depth'] = sample['depth'][..., None]
        
        if ('maskout' in self.config) and self.config.maskout:
            sample['mask'] = (sample['mask'].float() > 0.5).float()
            #print(sample['mask'].shape, sample['rgb'].shape)
            sample['rgb'] = sample['rgb'] * sample['mask']
            sample['depth'] = sample['depth'] * sample['mask']

            sample['goal_mask'] = (sample['goal_mask'].float() > 0.5).float()
            sample['goal_rgb'] = sample['goal_rgb'] * sample['goal_mask']
            sample['goal_depth'] = sample['goal_depth'] * sample['goal_mask']


         # # Vertical Flip
        if self.config.vertical_flip and train:
            
            # Generate random vertical flip decisions

            vertical_flip_decision = random.choice([True, False])

            if vertical_flip_decision:
                for obs in ['rgb', 'depth', 'mask', 'gray', 'contour', 'goal_rgb', 'goal_depth', 'goal_mask', 'goal_contour']:
                    if obs in sample:
                        sample[obs] = \
                            torchvision.transforms.functional.vflip(sample[obs])
        
            if ('action' in sample.keys()) and vertical_flip_decision:
                action_to_flip = sample['action'].reshape(-1, 2)
                action_to_flip[:, 0] = -action_to_flip[:, 0]
                sample['action'] = action_to_flip.reshape(*sample['action'].shape)

        # Random Rotation
        if self.config.random_rotation and train:
            #B = sample['rgb'].shape[0]
            #print('rotation')
            
            ### Generate torch version of the follow code:
            B = 1
            degree = self.config.rotation_degree * \
                torch.randint(int(360 / self.config.rotation_degree), size=(B,))
            thetas = torch.deg2rad(degree)

            # B * 2 * 2
            rot = torch.tensor([[
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)]]
                for theta in thetas]
            ).to(self.config.device)  # Move to the correct device
            
            ## Rotation Action
            if ('action' in sample.keys()):
                
                rotation_matrices_tensor = rot.squeeze(0)

                #print('rotaton metrics shape', rotation_matrices_tensor.shape)
                action_to_rotate = sample['action'].reshape(-1, 2)
                #print('action_to_rotate shape', action_to_rotate.shape)

                rotated_action = torch.matmul(action_to_rotate, rotation_matrices_tensor)\
                    .reshape(*sample['action'].shape)
                #print('rotated_action', rotated_action)
                
                ## if any of the action absolute value is greater than 1 sample the degree again
                while (abs(rotated_action) > 1).any():
                    print('resample')
                    degree = self.config.rotation_degree * \
                        torch.randint(int(360 / self.config.rotation_degree), size=(B,))
                    thetas = torch.deg2rad(degree)

                    # B * 2 * 2
                    rot = torch.tensor([[
                        [torch.cos(theta), -torch.sin(theta)],
                        [torch.sin(theta), torch.cos(theta)]]
                        for theta in thetas]
                    ).to(self.config.device)

                    rotation_matrices_tensor = rot.squeeze(0)
                    action_to_rotate = sample['action'].reshape(-1, 2)

                    rotated_action = torch.matmul(action_to_rotate, rotation_matrices_tensor)\
                        .reshape(*sample['action'].shape)

                sample['action'] = rotated_action

            ## Rotation Observation
            for obs in ['rgb', 'depth', 'mask', 'gray', 'contour', 'goal_rgb', 'goal_depth', 'goal_mask', 'goal_contour']:
                if obs in sample:
                    C, H, W = sample[obs].shape

                    
                    new_obs = sample[obs].permute(2, 0, 1).unsqueeze(0)

                    #print('new_obs and rot shape', new_obs.shape, rot.shape)

                    # Expand dimensions of rot to match the trajectory_images shape
                    rotation_matrices_expanded = rot.view(B, 1, 2, 2)

                    # Create an affine transformation matrix
                    # This includes a 2x2 rotation matrix and 2x1 translation matrix
                    affine_matrix = torch.zeros(B, 1, 2, 3, device=self.config.device)
                    affine_matrix[:, :, :2, :2] = rotation_matrices_expanded
                    affine_matrix[:, :, :, 2] = 0  # No translation for rotation

                    # Create grid using F.affine_grid
                    #new_obs = sample[obs].reshape(B, C, H, W)
                    grid = F.affine_grid(
                        affine_matrix.reshape(B, 2, 3), 
                        new_obs.size(), 
                        align_corners=True)

                    # Apply grid_sample for rotation
                    rotated_images = F.grid_sample(
                        new_obs, grid, align_corners=True)

                    sample[obs] = rotated_images.squeeze(0).permute(1, 2, 0)

            

       
        # Preprocess RGB
        if 'rgb' in sample:
        

            rgb_noise_factor = self.config.rgb_noise_factor if train else 0.0

            sample['rgb'] = preprocess_rgb(
                sample['rgb'],
                normalise={
                    'mode': self.config.rgb_norm_mode,
                    'param': self.config.rgb_norm_param
                },
                noise_factor=rgb_noise_factor)
            
        if 'goal_rgb' in sample:
            rgb_noise_factor = self.config.rgb_noise_factor if train else 0.0

            sample['goal_rgb'] = preprocess_rgb(
                sample['goal_rgb'],
                normalise={
                    'mode': self.config.rgb_norm_mode,
                    'param': self.config.rgb_norm_param
                },
                noise_factor=rgb_noise_factor)


            
        
        # Preprocess Depth
        if 'depth' in sample:
            sample['depth'] = preprocess_depth(
                sample['depth'],
                normalise={
                    'mode': self.config.depth_norm_mode,
                    'param': self.config.depth_norm_param
                }
            )
        
        if 'goal_depth' in sample:
            sample['goal_depth'] = preprocess_depth(
                sample['goal_depth'],
                normalise={
                    'mode': self.config.depth_norm_mode,
                    'param': self.config.depth_norm_param
                }
            )


        if ('action' in sample.keys()) and ('swap_action' in self.config) and self.config.swap_action:
            #print('sample aciotn shape', sample['action'].shape)
            sample['action'] = sample['action'][:, [1, 0, 3, 2]]
            #print('swap aciton')

        for k, v in sample.items():
            sample[k] = ts_to_np(v)

        return sample
       