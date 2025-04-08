import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from agent_arena.agent.utilities.torch_utils import np_to_ts, ts_to_np
from .utils import preprocess_rgb, postprocess_rgb, gaussian_kernel


def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(torch.arange(-m, m+1), torch.arange(-n, n+1), indexing='ij')
    h = torch.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h

class PickAndPlaceTransformer:
    def __init__(self,  config=None):
        self.config = config
    

    def __call__(self, sample, train=True, to_tensor=True, 
                 single=False):
        # batch is assumed to have the shape B*T*C*H*W
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                sample[k] = np_to_ts(v.copy(), self.config.device)
            else:
                sample[k] = v.to(self.config.device)
            if single:
                sample[k] = sample[k].unsqueeze(0)
            sample[k] = sample[k].float()

        # plot pre process action on image
        # if True:
        #     from agent_arena.utilities.visual_utils import draw_pick_and_place
        #     import cv2
        #     rgb = sample['rgb'][0, 0].squeeze(0).cpu().numpy()
        #     print('rgb shape', rgb.shape)
        #     if rgb.shape[0] == 3:
        #         rgb = rgb.transpose(1, 2, 0)
        #     H, W = rgb.shape[:2]
        #     action = sample['action'][0, 0].cpu().numpy().reshape(1, 4)
        #     print('action', action)
        #     pick = (action[:, :2] + 1)/2 * np.array([W, H])
        #     pick = (int(pick[0, 0]), int(pick[0, 1]))
        #     place = (action[:, 2:] + 1)/2 * np.array([W, H])
        #     place = (int(place[0, 0]), int(place[0, 1]))
        #     pnp_rgb = draw_pick_and_place(
        #         rgb, pick, place, get_ready=True, swap=False
        #     )
        #     cv2.imwrite('tmp/pre_pnp_rgb.png', pnp_rgb)

            # next_rgb = sample['rgb'][0, 1].squeeze(0).cpu().numpy()
            # print('next_rgb shape', next_rgb.shape)
            # if next_rgb.shape[0] == 3:
            #     next_rgb = next_rgb.transpose(1, 2, 0)
            # cv2.imwrite('tmp/pre_pnp_next_rgb.png', next_rgb)
        

        
        
        if 'rgbm' in sample:
            sample['rgb'] = sample['rgbm'][:, :, :3]
            sample['mask'] = sample['rgbm'][:, :, 3:]

        if 'gc-depth' in sample:
            sample['depth'] = sample['gc-depth'][:, :, :1]
            sample['goal-depth'] = sample['gc-depth'][:, :, 1:]

        for obs in ['rgb', 'depth', 'mask', 'rgbd', 'goal-rgb', 'goal-depth', 'goal-mask']:
            
            if obs in sample:
                if len(sample[obs].shape) == 4:
                    sample[obs] = sample[obs].unsqueeze(0)

                if sample[obs].shape[-1] <= 4:
                    sample[obs] = sample[obs].permute(0, 1, 4, 2, 3)
                #print('obs', obs, sample[obs].shape)
                B, T, C, H, W = sample[obs].shape

                sample[obs] = F.interpolate(
                    sample[obs].view(B*T, C, H, W),
                    size=self.config.img_dim, mode='bilinear', align_corners=False)\
                        .view(B, T, C, *self.config.img_dim)

                if obs == 'mask':
                    sample[obs] = (sample[obs] > 0.5).float()
        
        if 'rgbd' in sample:
            #print('Were!!!!!!!!!!!!!!')
            sample['rgb'] = sample['rgbd'][:, :, :3]
            sample['depth'] = sample['rgbd'][:, :, 3:]
            # print('rgbd', sample['rgbd'].shape)
            # print('rgb', sample['rgb'].shape)
            # print('depth', sample['depth'].shape)

        # if single:
        #     for k, v in sample.items():
        #         sample[k] = v.squeeze(0)

        # return sample
        if 'rgb' in sample:
            sample['rgb'] = preprocess_rgb(
                sample['rgb'], 
                normalise={'mode': self.config.rgb_norm_mode, 
                           'param': self.config.rgb_norm_param}, 
                noise_factor=(self.config.rgb_noise_factor if train else 0))
        
        if 'goal-rgb' in sample:
            sample['goal-rgb'] = preprocess_rgb(
                sample['goal-rgb'], 
                normalise={'mode': self.config.rgb_norm_mode, 
                           'param': self.config.rgb_norm_param}, 
                noise_factor=(self.config.rgb_noise_factor if train else 0))

        # from matplotlib import pyplot as plt
        # plt.imshow(sample['depth'][0, 0].squeeze(0).cpu().numpy())
        # plt.savefig('raw-depth.png')

        # from matplotlib import pyplot as plt
        # plt.imshow(sample['mask'][0, 0].squeeze(0).cpu().numpy())
        # plt.savefig('raw-mask.png')

        if 'depth' in sample:
            sample['depth'] = self._process_depth(
                sample['depth'], 
                sample.get('mask', None), train)
        if 'goal-depth' in sample:
            sample['goal-depth'] = self._process_depth(
                sample['goal-depth'], 
                sample.get('goal-mask', None), train)

     
            
        if self.config.reward_scale and train:
            sample['reward'] *= self.config.reward_scale
        
        # we assume the action is in the correct form
        if 'action' in sample.keys():
            sample['action'] = sample['action'][:, :, [1, 0, 3, 2]]

            if 'swap_action' in self.config and self.config.swap_action:
            # print('swap action')
                sample['action'] = sample['action'][:, :, [1, 0, 3, 2]]

        

    
        # Random Rotation
        if self.config.random_rotation and train:
            B = sample['action'].shape[0]
            #print('rotation')
            
            ### Generate torch version of the follow code:
            while True:
                
                degree = self.config.rotation_degree * \
                    torch.randint(int(360 / self.config.rotation_degree), size=(B,))
                thetas = torch.deg2rad(degree)
                cos_theta = torch.cos(thetas)
                sin_theta = torch.sin(thetas)

                # B * 2 * 2
                # rot = torch.tensor([[
                #     [torch.cos(theta), -torch.sin(theta)],
                #     [torch.sin(theta), torch.cos(theta)]]
                #     for theta in thetas]
                # ).to(self.config.device)  # Move to the correct device

                rot = torch.stack([
                    torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1).reshape(B, 2, 2)
                ], dim=1).to(self.config.device)

                # rotation_matrices_tensor = rot.unsqueeze(1).expand(B, (T-1)*2, 2, 2).reshape(-1, 2, 2)

                # rotated_action = torch.matmul(sample['action'].reshape(-1, 1, 2), rotation_matrices_tensor)\
                #     .reshape(*sample['action'].shape)

                # Rotate actions
                rotation_matrices_tensor = rot.expand(B, (T-1)*2, 2, 2).reshape(-1, 2, 2)
                rotation_action = sample['action'].reshape(-1, 1, 2)
                # print('rotation_action', rotation_action.shape)
                # print('rotation_matrices_tensor', rotation_matrices_tensor.shape)
                rotated_action = torch.bmm(rotation_action, rotation_matrices_tensor).reshape(*sample['action'].shape)
                #sample['action'] = rotated_action

                #print('rotated_action', rotated_action)
                # if torch.any(torch.abs(rotated_action) > 1):
                #     continue

                # Rotate observations
                affine_matrix = torch.zeros(B, T, 2, 3, device=self.config.device)
                affine_matrix[:, :, :2, :2] = rot.expand(B, T, 2, 2)

                ## Rotation Observation
                for obs in ['rgb', 'depth', 'mask']:
                    if obs in sample:
                        B, T, C, H, W = sample[obs].shape

                        new_obs = sample[obs].reshape(B*T, C, H, W)
                        grid = F.affine_grid(affine_matrix.reshape(B*T, 2, 3), new_obs.size(), align_corners=True)
                        rotated_images = F.grid_sample(new_obs, grid, align_corners=True)
                        sample[obs] = rotated_images.reshape(B, T, C, H, W)

                ## Rotation Action
               
                sample['action'] = rotated_action
                break

        # Vertical Flip
        if self.config.vertical_flip and train:
            
            # Generate random vertical flip decisions
            vertical_flip_decision = torch.randint(0, 2, size=(B,)).bool()


            for obs in ['rgb', 'depth', 'mask']:
                if obs in sample:
                    sample[obs][vertical_flip_decision] = \
                        torch.flip(sample[obs][vertical_flip_decision], dims=[-2])
                        
                        #torchvision.transforms.functional.vflip(sample[obs][vertical_flip_decision])



            B, T, _ = sample['action'].shape
            new_actions = sample['action'][vertical_flip_decision].reshape(-1, 2)
            new_actions[:, 1] = -new_actions[:, 1]
            sample['action'][vertical_flip_decision] = new_actions.reshape(-1, T, 4)
            # sample['action'] = new_actions.reshape(*sample['action'].shape)
        
        if 'action' in sample.keys() and 'swap_action' in self.config and self.config.swap_action:
            # print('swap action')
            sample['action'] = sample['action'][:, :, [1, 0, 3, 2]]



        if 'maskout' in self.config and self.config.maskout:
            bg_value = self.config.bg_value \
                if 'bg_value' in self.config else 0
            # print('rgb', sample['rgb'].shape)
            # print('mask', sample['mask'].shape)
            if 'rgb' in sample:
                sample['rgb'] = sample['rgb'] * sample['mask'] + \
                    bg_value * (1 - sample['mask'])
            if 'depth' in sample:
                sample['depth'] = sample['depth'] * sample['mask'] + \
                    bg_value * (1 - sample['mask'])
            
            if 'goal-rgb' in sample:
                sample['goal-rgb'] = sample['goal-rgb'] * sample['goal-mask'] + \
                    bg_value * (1 - sample['goal-mask'])
            if 'goal-depth' in sample:
                sample['goal-depth'] = sample['goal-depth'] * sample['goal-mask'] + \
                    bg_value * (1 - sample['goal-mask'])

        

        if 'rgbd' in sample:
            #print('Here!!!!!!!!!!!!!!')
            sample['rgbd'][:, :, :3] = sample['rgb']
            sample['rgbd'][:, :, 3:] = sample['depth']

        if 'gc-depth' in sample:
            sample['gc-depth'] = torch.cat([sample['depth'], sample['goal-depth']], dim=2)
        
        if 'rgbm' in sample:
            sample['rgbm'][:, :, :3] = sample['rgb']
            sample['rgbm'][:, :, 3:] = sample['mask']
        
        if 'action' in sample:
            sample['action'] = sample['action'].clip(-1, 1)
        
            if 'action2map' in self.config and self.config.action2map:

                if self.config.action2map_encoding in ['normal', 'normal-average', 'normal-map']:
                    pick_idx = ((sample['action'][:, :, :2] + 1) / 2 * torch.tensor([H, W]).to(self.config.device)).long()
                    place_idx = ((sample['action'][:, :, 2:] + 1) / 2 * torch.tensor([H, W]).to(self.config.device)).long()
                    
                    action_heatmap = torch.zeros(B, T, 2, H, W)

                    # Generate Gaussian kernel
                    sigma = min(H, W) / 64.  # You can adjust this value
                    kernel_size = int(6 * sigma)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    gaussian_kernel_ = gaussian_2d((kernel_size, kernel_size), sigma).to(sample['action'].device)

                    for b in range(B):
                        for t in range(T):
                            # Pick action
                            y, x = pick_idx[b, t]
                            if 0 <= y < H and 0 <= x < W:
                                top = max(0, y - kernel_size // 2)
                                left = max(0, x - kernel_size // 2)
                                bottom = min(H, y + kernel_size // 2 + 1)
                                right = min(W, x + kernel_size // 2 + 1)
                                
                                kernel_top = max(0, kernel_size // 2 - y)
                                kernel_left = max(0, kernel_size // 2 - x)
                                kernel_bottom = min(kernel_size, kernel_size // 2 + H - y)
                                kernel_right = min(kernel_size, kernel_size // 2 + W - x)
                                
                                action_heatmap[b, t, 0, top:bottom, left:right] = gaussian_kernel_[kernel_top:kernel_bottom, kernel_left:kernel_right]

                            # Place action
                            y, x = place_idx[b, t]
                            if 0 <= y < H and 0 <= x < W:
                                top = max(0, y - kernel_size // 2)
                                left = max(0, x - kernel_size // 2)
                                bottom = min(H, y + kernel_size // 2 + 1)
                                right = min(W, x + kernel_size // 2 + 1)
                                
                                kernel_top = max(0, kernel_size // 2 - y)
                                kernel_left = max(0, kernel_size // 2 - x)
                                kernel_bottom = min(kernel_size, kernel_size // 2 + H - y)
                                kernel_right = min(kernel_size, kernel_size // 2 + W - x)
                                
                                action_heatmap[b, t, 1, top:bottom, left:right] = gaussian_kernel_[kernel_top:kernel_bottom, kernel_left:kernel_right]

                    # Normalize each heatmap
                    action_heatmap = F.normalize(action_heatmap.view(B*T*2, -1), p=1, dim=1).view(B, T, 2, H, W)

                    if self.config.action2map_encoding == 'normal-average':
                        action_heatmap = action_heatmap - 0.5
                    elif self.config.action2map_encoding == 'normal-map':
                        action_heatmap = action_heatmap * 2 - 1

                    #sample['action_heatmap'] = action_heatmap

                else:
                    pick_idx = ((sample['action'][:, :, :2] + 1) / 2 * torch.tensor([H, W])).long()
                    place_idx = ((sample['action'][:, :, 2:] + 1) / 2 * torch.tensor([H, W])).long()
                    
                    action_heatmap = torch.zeros(B, T, 2, H, W)

                    batch_indices = torch.arange(B).view(B, 1).repeat(1, T)
                    time_indices = torch.arange(T).repeat(B, 1)

                    action_heatmap[batch_indices, time_indices, 0, pick_idx[..., 0], pick_idx[..., 1]] = 1.0
                    action_heatmap[batch_indices, time_indices, 1, place_idx[..., 0], place_idx[..., 1]] = 1.0

                
                sample['action_heatmap'] = action_heatmap.to(sample['action'].device)
        if 'action' in sample.keys():
            sample['action'] = sample['action'][:, :, [1, 0, 3, 2]]
        ## plot pre process action on image
        # if True:
        #     from agent_arena.utilities.visual_utils import draw_pick_and_place
        #     import cv2
        #     rgb = sample['rgb'][0, 0].squeeze(0).cpu().numpy().transpose(1, 2, 0)
        #     rgb = (rgb * 255).astype(np.uint8)
        #     print('rgb shape', rgb.shape)
        #     H, W = rgb.shape[:2]
        #     action = sample['action'][0, 0].cpu().numpy().reshape(1, 4)
        #     print('action', action)
        #     pick = (action[:, :2] + 1)/2 * np.array([W, H])
        #     pick = (int(pick[0, 0]), int(pick[0, 1]))
        #     place = (action[:, 2:] + 1)/2 * np.array([W, H])
        #     place = (int(place[0, 0]), int(place[0, 1]))
        #     pnp_rgb = draw_pick_and_place(
        #         rgb, pick, place, get_ready=True, swap=False
        #     )
        #     cv2.imwrite('tmp/post_pnp_rgb.png', pnp_rgb)

        # if True:
        #     import matplotlib.pyplot as plt
        #     import cv2
        #     ## save rgb, depth, mask
        #     plt.imshow(sample['rgb'][0, 0].squeeze(0).cpu().numpy().transpose(1, 2, 0))
        #     plt.savefig('tmp/process-rgb.png')
        #     plt.imshow(sample['depth'][0, 0].squeeze(0).cpu().numpy())
        #     plt.savefig('tmp/process-depth.png')
        #     plt.imshow(sample['mask'][0, 0].squeeze(0).cpu().numpy())
        #     plt.savefig('tmp/process-mask.png')

        #     plt.imshow(sample['goal-depth'][0, 0].squeeze(0).cpu().numpy())
        #     plt.savefig('process-goal-depth.png')
        #     plt.imshow(sample['goal-mask'][0, 0].squeeze(0).cpu().numpy())
        #     plt.savefig('process-goal-mask.png')
        #     exit()
            # plt.imshow(sample['rgb'][0, 0].squeeze(0).cpu().numpy()\
            #            .transpose(1, 2, 0))
            # plt.savefig('process-rgb.png')
            # from matplotlib import pyplot as plt
            # plt.imshow(sample['action_heatmap'][0, 0, 0].cpu().numpy())
            # plt.savefig('process-action-heatmap.png')

        if single:
            for k, v in sample.items():
                sample[k] = v.squeeze(0)
        
        ## check if there is any nan value
        for k, v in sample.items():
            if torch.isnan(v).any():
                print('Transform nan value in', k)
                #print(v)
                raise ValueError('nan value in the transform data {k}')

      
        if not to_tensor:
            for k, v in sample.items():
                sample[k] = ts_to_np(v)
        
        # print all output shape
        # print('HELLOO')
        # for k, v in sample.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape)
        # exit()
        
        return sample
    
    def postprocess(self, sample):
        
        res = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                res[k] = ts_to_np(v)
            else:
                res[k] = v   
            #print(k, res[k].shape)

        if 'rgb' in res:
            res['rgb'] = postprocess_rgb(
                res['rgb'], 
                normalise={'mode': self.config.rgb_norm_mode, 
                           'param': self.config.rgb_norm_param}) 
        
        if 'rgbd' in res:
            if len(res['rgbd'].shape) == 3:
                rgb = postprocess_rgb(res['rgbd'][:3, :, :], 
                            normalise={'mode': self.config.rgb_norm_mode, 
                           'param': self.config.rgb_norm_param}).astype(np.float32)
                depth = res['rgbd'][3:, :, :].astype(np.float32)

                if self.config.z_norm:
                    depth = depth * self.config.z_norm_std + \
                        self.config.z_norm_mean
                
                if self.config.min_max_norm:
                    depth = \
                        depth * (self.config.depth_max - self.config.depth_min) \
                        + self.config.depth_min
        
                res['rgbd'] = np.concatenate([rgb, depth], axis=0).astype(np.float32)

            else:
                res['rgbd'] = postprocess_rgb(res['rgbd'][:, :3, :, :],
                            normalise={'mode': self.config.rgb_norm_mode, 
                           'param': self.config.rgb_norm_param}) 

        if 'rgbm' in res:
            res['rgbm'] = postprocess_rgb(res['rgbm'][:, :3, :, :])
            
        if 'depth' in res:
            
            if self.config.z_norm:
                res['depth'] = res['depth'] * self.config.z_norm_std + \
                      self.config.z_norm_mean
            
            if self.config.min_max_norm:
                res['depth'] = \
                    res['depth'] * (self.config.depth_max - self.config.depth_min) \
                    + self.config.depth_min
        
        if 'action_heatmap' in res:
            # convert action heatmap to action
            #print('action heatmap shape', res['action_heatmap'].shape)
            B, T, _, H, W = res['action_heatmap'].shape
            action_heatmap = res['action_heatmap'].reshape(B*T, 2, H, W)
            
            # For pick action
            pick_idx = np.argmax(action_heatmap[:, 0].reshape(B*T, -1), axis=1)
            pick_idx = np.stack([pick_idx // W, pick_idx % W], axis=1).astype(float)
            
            # For place action
            place_idx = np.argmax(action_heatmap[:, 1].reshape(B*T, -1), axis=1)
            place_idx = np.stack([place_idx // W, place_idx % W], axis=1).astype(float)
            
            # Combine pick and place actions
            action = np.concatenate([pick_idx, place_idx], axis=1)
            res['action'] = action.reshape(B, T, 4)

            res['action'] = res['action'] / np.array([H, W, H, W]) * 2 - 1

            res['action'] = res['action'].reshape(B, T, 4)

            #print('action', res['action'])

        if 'action' in res:
            res['action'] = res['action'].clip(-1, 1)

        return res
    

    def _process_depth(self, depth, mask=None, train=False):
        B, T, C, H, W = depth.shape
      
        if 'thickness_augmentation' in self.config and self.config.thickness_augmentation and train:
            
            # generate a random scaling factor for each trajectory
            B = depth.shape[0]
            # plt.imshow(sample['depth'][0, 0].squeeze(0).cpu().numpy())
            # plt.savefig('depth.png')
            thickness_scales = torch.rand(B, device=self.config.device) *\
                    (self.config.thickness_scale[1] - self.config.thickness_scale[0]) + self.config.thickness_scale[0]
            # print('thickness_scales', thickness_scales.shape)
            # print('thicknes scale min', thickness_scales.min())
            # print('thicknes scale max', thickness_scales.max())
            
            # extract the region that needs to be scales using mask
            #depth_to_scale = 1.0*sample['depth'] * sample['mask']
            thick_to_scale = self.config.thinkness_base - depth
            
            # plt.imshow(thick_to_scale[0, 0].squeeze(0).cpu().numpy())
            # plt.savefig('thick_to_scale.png')
            # print('thick_to_scale', thick_to_scale.shape)
            thick_scaled = thick_to_scale * thickness_scales.view(B, 1, 1, 1, 1)
            depth_scaled = -thick_scaled + self.config.thinkness_base
            depth = depth_scaled
            # plt.imshow(sample['depth'][0, 0].squeeze(0).cpu().numpy())
            # plt.savefig('depth_scaled.png')


        #print('depth')
        #obs[:,-1,:, :] += torch.randn(obs[:,-1,:, :].shape, device=self.config.device) * (self.config.depth_noise_var if train else 0)
        if self.config.depth_clip:
            depth = depth.clip(self.config.depth_clip_min, self.config.depth_clip_max)
        
        if self.config.z_norm:
            depth = (depth - self.config.z_norm_mean) / self.config.z_norm_std

        elif self.config.min_max_norm:
            # get the min and max of each trajectory
            depth_min = depth.view(B, T, -1).min(dim=2, keepdim=True).values
            depth_max = depth.view(B, T, -1).max(dim=2, keepdim=True).values

            ## depth_min compare with self.config.depth_min and get the max for each trajectory

            if 'depth_hard_interval' in self.config and self.config.depth_hard_interval:
                #print('depth_hard_interval', self.config.depth_hard_interval)
                depth_min = self.config.depth_min
                depth_max = self.config.depth_max
            else:
                depth_min = torch.max(
                    depth_min, 
                    torch.tensor(self.config.depth_min).to(depth_min.device)).view(B, T, 1, 1, 1)
                
                depth_max = torch.min(
                    depth_max, 
                    torch.tensor(self.config.depth_max).to(depth_max.device)).view(B, T, 1, 1, 1)



            depth = (depth-depth_min) / (depth_max-depth_min+1e-3)
            if 'depth_flip' in self.config and self.config.depth_flip:
                depth = 1 - depth
        
        
        depth_noise = \
            torch.randn(depth.shape, device=self.config.device) \
                * (self.config.depth_noise_var if train else 0)

        if 'depth_blur' in self.config and self.config.depth_blur and train:
            ### apply gaussian blur on each image
            B, T, C, H, W = depth.shape
            kernel_size = self.config.depth_blur_kernel_size
            sigma = 1.0

            # Create Gaussian kernel
            kernel = gaussian_kernel(kernel_size, sigma).to(depth.device)
            kernel = kernel.expand(C, 1, kernel_size, kernel_size)

            # Reshape the tensor to apply blur to each image individually
            depth_reshaped = depth.view(B * T, C, H, W)

            # Apply Gaussian blur using convolution
            padding = (kernel_size - 1) // 2
            blurred_depth = F.conv2d(depth_reshaped, kernel, padding=padding, groups=C)

            # Reshape back to original dimensions
            depth = blurred_depth.view(B, T, C, H, W)

        if 'apply_depth_noise_on_mask' in self.config and self.config.apply_depth_noise_on_mask \
            and (mask is not None):
            depth_noise *= mask
            
        depth += depth_noise
        
        depth = depth.clip(0, 1)
        
        if self.config.depth_map:
            # At this point we assume depth is between [0, 1]
            map_diff = self.config.depth_map_range[1] - self.config.depth_map_range[0]
            depth = depth*map_diff + self.config.depth_map_range[0]
        
        return depth