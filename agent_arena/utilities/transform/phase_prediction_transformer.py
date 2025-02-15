import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from agent_arena.agent.utilities.torch_utils import np_to_ts, ts_to_np
from agent_arena.utilities.transform.utils import \
    preprocess_rgb, preprocess_depth, postprocess_rgb

class PhasePredictionTransformer:
    def __init__(self,  config=None):
        self.config = config
        if 'grayscale' in config and config.grayscale:
            self.to_gray = torchvision.transforms.Grayscale(num_output_channels=1)
    
    def postprocess(self, sample):
        for k, v in sample.items():
            sample[k] = np_to_ts(v, self.config.device)
            

        if 'rgb' in sample:
            sample['rgb'] = postprocess_rgb(
                sample['rgb'], 
                normalise={
                    'mode': self.config.rgb_norm_mode,
                    'param': self.config.rgb_norm_param
                }
            )
            return sample
        
        if 'gray' in sample:
            sample['gray'] = postprocess_rgb(
                sample['gray'], 
                normalise={
                    'mode': self.config.gray_norm_mode,
                    'param': self.config.gray_norm_param
                }
            )
            return sample
           

    def __call__(self, sample, augment=True, to_tensor=True, single=False):
        # batch is assumed to have the shape B*T*C*H*W
        for k, v in sample.items():
            sample[k] = np_to_ts(v, self.config.device)
            if single:
                sample[k] = sample[k].unsqueeze(0)

        for obs in ['rgb', 'depth', 'mask', 'rgbd', 'contour']:
            if obs in sample:
                #print('shape', sample[obs].shape)
                B, C, H, W = sample[obs].shape
                # print(obs, sample[obs].shape)
                sample[obs] = F.interpolate(
                    sample[obs].reshape(B, C, H, W),
                    size=self.config.img_dim, mode='bilinear', align_corners=False)\
                        .reshape(B, C, *self.config.img_dim)
        
        if 'depth' in sample:
            #from matplotlib import pyplot as plt
            if 'thickness_augmentation' in self.config and self.config.thickness_augmentation and augment:
                
                # generate a random scaling factor for each trajectory
                B = sample['depth'].shape[0]
                # plt.imshow(sample['depth'][0, 0].squeeze(0).cpu().numpy())
                # plt.savefig('depth.png')
                thickness_scales = torch.rand(B, device=self.config.device) *\
                      (self.config.thickness_scale[1] - self.config.thickness_scale[0]) + self.config.thickness_scale[0]
                # print('thickness_scales', thickness_scales.shape)
                # print('thicknes scale min', thickness_scales.min())
                # print('thicknes scale max', thickness_scales.max())
                
                # extract the region that needs to be scales using mask
                #depth_to_scale = 1.0*sample['depth'] * sample['mask']
                thick_to_scale = self.config.thinkness_base - sample['depth']
                
                # plt.imshow(thick_to_scale[0, 0].squeeze(0).cpu().numpy())
                # plt.savefig('thick_to_scale.png')
                # print('thick_to_scale', thick_to_scale.shape)
                thick_scaled = thick_to_scale * thickness_scales.view(B, 1, 1, 1)
                depth_scaled = -thick_scaled + self.config.thinkness_base
                sample['depth'] = depth_scaled

        if ('maskout' in self.config) and self.config.maskout:
            sample['mask'] = (sample['mask'].float() > 0.9).float()
            #print(sample['mask'].shape, sample['rgb'].shape)
            sample['rgb'] = sample['rgb'] * sample['mask']
            #from matplotlib import pyplot as plt
            # plt.imshow(sample['rgb'].cpu().numpy().astype(np.uint8))
            # plt.show()
            sample['depth'] = sample['depth'] * sample['mask']

        if 'rgb' in sample:
            

            if 'grayscale' in self.config and self.config.grayscale:
                sample['gray'] = self.to_gray(sample['rgb']/255.0)*255.0

                gray_noise_factor = self.config.gray_noise_factor if augment else 0.0
                sample['gray'] = preprocess_rgb(
                    sample['gray'],
                    normalise={
                        'mode': self.config.gray_norm_mode,
                        'param': self.config.gray_norm_param
                    },
                    noise_factor=gray_noise_factor)
                
                # from matplotlib import pyplot as plt

                # plt.imshow(sample['gray'][0].permute(1, 2, 0).cpu().numpy())
                # plt.show()

            
            sample['rgb'] = preprocess_rgb(
                sample['rgb'],
                normalise={
                    'mode': self.config.rgb_norm_mode,
                    'param': self.config.rgb_norm_param
                },
                noise_factor= (0.0 if not augment else self.config.rgb_noise_factor)
            )
        
        
                # plt.imshow(sample['depth'][0, 0].squeeze(0).cpu().numpy())
                # plt.savefig('depth_scaled.png')
        
        if 'depth' in sample:

            preprocess_depth(
                sample['depth'],
                normalise={
                    'mode': self.config.depth_norm_mode,
                    'param': self.config.depth_norm_param
                }
            )
            
       
        # Random Rotation
        if self.config.random_rotation and augment:
            B = sample['rgb'].shape[0]
            #print('rotation')
            
            ### Generate torch version of the follow code:
            degree = self.config.rotation_degree * torch.randint(int(360 / self.config.rotation_degree), size=(B,))
            thetas = torch.deg2rad(degree)

            # B * 2 * 2
            rot = torch.tensor([[
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)]]
                for theta in thetas]
            ).to(self.config.device)  # Move to the correct device

            ## Rotation Observation
            for obs in ['rgb', 'depth', 'mask', 'gray', 'contour']:
                if obs in sample:
                    B, C, H, W = sample[obs].shape

                    # Expand dimensions of rot to match the trajectory_images shape
                    rotation_matrices_expanded = rot.view(B, 1, 2, 2)

                    # Create an affine transformation matrix
                    # This includes a 2x2 rotation matrix and 2x1 translation matrix
                    affine_matrix = torch.zeros(B, 1, 2, 3, device=self.config.device)
                    affine_matrix[:, :, :2, :2] = rotation_matrices_expanded
                    affine_matrix[:, :, :, 2] = 0  # No translation for rotation

                    # Create grid using F.affine_grid
                    new_obs = sample[obs].reshape(B, C, H, W)
                    grid = F.affine_grid(
                        affine_matrix.reshape(B, 2, 3), 
                        new_obs.size(), 
                        align_corners=True)

                    # Apply grid_sample for rotation
                    rotated_images = F.grid_sample(
                        new_obs, grid, align_corners=True)

                    sample[obs] = rotated_images.reshape(B, C, H, W)


        # Vertical Flip
        if self.config.vertical_flip and augment:
            
            # Generate random vertical flip decisions
            vertical_flip_decision = torch.randint(0, 2, size=(B,)).bool()


            for obs in ['rgb', 'depth', 'mask', 'gray', 'contour']:
                if obs in sample:
                    sample[obs][vertical_flip_decision] = \
                        torchvision.transforms.functional.vflip(sample[obs][vertical_flip_decision])

        for obs in ['mask', 'contour']:
            if obs in sample:
                sample[obs] = (sample[obs] > 0.1).float()
                # from matplotlib import pyplot as plt
                # plt.imshow(sample[obs][0].permute(1, 2, 0).cpu().numpy())
                # plt.show()


        if not to_tensor:
            for k, v in sample.items():
                sample[k] = ts_to_np(v)
        
        return sample