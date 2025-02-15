import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import random
import cv2
from agent_arena.agent.utilities.torch_utils import np_to_ts, ts_to_np
from agent_arena.utilities.transform.utils import *
from agent_arena.utilities.visual_utils import draw_pick_and_place

class TransporterNetTransformer:
    def __init__(self, config=None):
        self.config = config
        if 'color_jitter' in config:
            self.color_jitter = torchvision.transforms.ColorJitter(**self.config.color_jitter)
        if 'grayscale' in config and config.grayscale:
            self.to_gray = torchvision.transforms.Grayscale(num_output_channels=1)

    def __call__(self, sample, train=True, sim2real=False):
        #print('config device', self.config.device)
        # if action is dict
        if 'action' in sample.keys() and isinstance(sample['action'], dict):
            sample['action'] = \
                np.concatenate([sample['action']['pick_0'], sample['action']['place_0']]).reshape(-1, 4)
        
        if 'action' in sample.keys() and 'pre_swap_action' in self.config and self.config.pre_swap_action:
            sample['action'] = sample['action'][:, [1, 0, 3, 2]]

        # if True:
        #     rgb = sample['color']
        #     print('rgb shape', rgb.shape)
        #     H, W = rgb.shape[:2]
        #     action = sample['action']
        #     print('action', action)
        #     pick = (action[:, :2] + 1)/2 * np.array([W, H])
        #     pick = (int(pick[0, 0]), int(pick[0, 1]))
        #     place = (action[:, 2:] + 1)/2 * np.array([W, H])
        #     place = (int(place[0, 0]), int(place[0, 1]))
        #     pnp_rgb = draw_pick_and_place(
        #         rgb, pick, place, get_ready=True, swap=True
        #     )
        #     cv2.imwrite('tmp/pre_pnp_rgb.png', pnp_rgb)


                

        sample = {k: np_to_ts(v, self.config.device) for k, v in sample.items()}
        for k, v in sample.items():
            if k in ['depth', 'mask', 'contour'] and len(v.shape) == 2:
                sample[k] = v.unsqueeze(2)

        for k, v in sample.items():
            if k in ['rgb', 'color', 'depth', 'mask'] and tuple(v.shape[:2]) != tuple(self.config.img_dim):
                sample[k] = F.interpolate(v.permute(2, 0, 1).unsqueeze(0), self.config.img_dim, mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

        if 'rgb' not in sample and 'color' in sample:
            sample['rgb'] = sample['color']
        if 'depth' in sample and len(sample['depth'].shape) == 2:
            sample['depth'] = sample['depth'][..., None]

        # save rgb and depth images
        # if True:
        #     from matplotlib import pyplot as plt
        #     if 'rgb' in sample:
        #         plt.imshow(sample['rgb'].cpu().numpy()/255.0)
        #         plt.axis('off')
        #         plt.savefig('tmp/rgb.png', bbox_inches='tight', pad_inches=0)
        #     if 'depth' in sample:
        #         depth = sample['depth'].cpu().numpy()
        #         norm_depth = (depth - depth.min()) / (depth.max() - depth.min())
        #         color_depth = cv2.applyColorMap((norm_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
        #         cv2.imwrite('tmp/depth.png', color_depth)
            #plt.close()

        # from matplotlib import pyplot as plt
        # plt.imshow(sample['depth'].cpu().numpy())
        # plt.savefig('pre-depth.png')

        if 'depth' in sample:
            min_depth = sample['depth'].min().item()
            max_depth = sample['depth'].max().item()
        #     print('min depth', min_depth)
        #     print('max depth', max_depth)

        if 'thickness_augmentation' in self.config and self.config.thickness_augmentation and train:
            B = 1
            thickness_scales = torch.rand(B, device=self.config.device) * (self.config.thickness_scale[1] - self.config.thickness_scale[0]) + self.config.thickness_scale[0]
            thick_to_scale = self.config.thinkness_base - sample['depth']
            thick_scaled = thick_to_scale * thickness_scales.view(B, 1, 1)
            depth_scaled = -thick_scaled + self.config.thinkness_base
            sample['depth'] = depth_scaled

        if self.config.vertical_flip and train:
            vertical_flip_decision = random.choice([True, False])
            if vertical_flip_decision:
                for obs in ['rgb', 'depth', 'mask', 'gray', 'contour']:
                    if obs in sample:
                        sample[obs] = torchvision.transforms.functional.vflip(sample[obs])
                if 'action' in sample.keys():
                    action_to_flip = sample['action'].reshape(-1, 2)
                    action_to_flip[:, 0] = -action_to_flip[:, 0]
                    sample['action'] = action_to_flip.reshape(*sample['action'].shape)

        if self.config.random_rotation and train:
            B = 1
            degree = self.config.rotation_degree * torch.randint(int(360 / self.config.rotation_degree), size=(B,))
            thetas = torch.deg2rad(degree)
            rot = torch.tensor([[[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]] for theta in thetas]).to(self.config.device)

            if 'action' in sample.keys():
                rotation_matrices_tensor = rot.squeeze(0)
                action_to_rotate = sample['action'].reshape(-1, 2)
                rotated_action = torch.matmul(action_to_rotate, rotation_matrices_tensor).reshape(*sample['action'].shape)
                cnt = 0
                while (abs(rotated_action) > 1).any() and cnt < 10:
                    degree = self.config.rotation_degree * torch.randint(int(360 / self.config.rotation_degree), size=(B,))
                    thetas = torch.deg2rad(degree)
                    rot = torch.tensor([[[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]] for theta in thetas]).to(self.config.device)
                    rotation_matrices_tensor = rot.squeeze(0)
                    rotated_action = torch.matmul(action_to_rotate, rotation_matrices_tensor).reshape(*sample['action'].shape)
                    cnt += 1
                if cnt >= 10:
                    print('Failed to rotate action')
                else:
                    sample['action'] = rotated_action

            for obs in ['rgb', 'depth', 'mask', 'gray', 'contour']:
                if obs in sample:
                    C, H, W = sample[obs].shape
                    new_obs = sample[obs].permute(2, 0, 1).unsqueeze(0)
                    rotation_matrices_expanded = rot.view(B, 1, 2, 2)
                    affine_matrix = torch.zeros(B, 1, 2, 3, device=self.config.device)
                    affine_matrix[:, :, :2, :2] = rotation_matrices_expanded
                    affine_matrix[:, :, :, 2] = 0
                    grid = F.affine_grid(affine_matrix.reshape(B, 2, 3), new_obs.size(), align_corners=True)
                    rotated_images = F.grid_sample(new_obs, grid, align_corners=True)
                    sample[obs] = rotated_images.squeeze(0).permute(1, 2, 0)
                    if obs == 'depth':
                        sample[obs] = sample[obs].clip(min_depth, max_depth)

        if 'scale_range' in self.config and 'action' in sample.keys() and train:
            max_scale_factor = (1.0 / (abs(ts_to_np(sample['action'])))).min()
            scale_factor = np.random.uniform(self.config.scale_range[0], min(self.config.scale_range[1], max_scale_factor))
            sample['rgb'] = zoom_image(sample['rgb'].permute(2, 0, 1), scale_factor).permute(1, 2, 0)
            if 'depth' in sample:
                sample['depth'] = zoom_image(sample['depth'].permute(2, 0, 1), scale_factor).permute(1, 2, 0)
            if 'mask' in sample:
                sample['mask'] = zoom_image(sample['mask'].permute(2, 0, 1), scale_factor).permute(1, 2, 0)
            if 'action' in sample.keys():
                sample['action'] = sample['action'] * scale_factor

        if 'rgb' in sample:
            if 'color_jitter' in self.config and train:
                sample['rgb'] = self.color_jitter((sample['rgb'].float() / 255.0).permute(2, 0, 1)).permute(1, 2, 0) * 255.0
            rgb_noise_factor = self.config.rgb_noise_factor if train else 0.0
            #print('noise factor', rgb_noise_factor)
            sample['rgb'] = preprocess_rgb(
                sample['rgb'], 
                normalise={'mode': self.config.rgb_norm_mode, 'param': self.config.rgb_norm_param}, 
                noise_factor=rgb_noise_factor)

        if 'grayscale' in self.config and self.config.grayscale:
            rgb = sample['rgb'].permute(2, 0, 1)
            sample['gray'] = self.to_gray(rgb).permute(1, 2, 0) * 255.0
            gray_noise_factor = self.config.gray_noise_factor if train else 0.0
            sample['gray'] = preprocess_rgb(sample['gray'], normalise={'mode': self.config.gray_norm_mode, 'param': self.config.gray_norm_param}, noise_factor=gray_noise_factor)

        if 'depth' in sample:
            #print('transform sim2real', sim2real)
            if not sim2real:
                #print('preprocess depth !!!')
                sample['depth'] = preprocess_depth(
                    sample['depth'], 
                    normalise={
                        'mode': self.config.depth_norm_mode,
                        'param': self.config.depth_norm_param}, 
                    noise_factor=self.config.depth_noise_factor if train else 0.0)
            
            if 'blur_depth' in self.config and self.config.blur_depth and train:
                sample['depth'] = cv2.GaussianBlur(sample['depth'].squeeze(2).unsqueeze(0).cpu().numpy(), (11,11), 0)
                sample['depth'] = np_to_ts(sample['depth'], self.config.device).squeeze(0).unsqueeze(2)
            if 'depth_flip' in self.config and self.config.depth_flip:
                sample['depth'] = 1.0 - sample['depth']
                sample['depth'] = sample['depth'].clip(0, 1)
            
            # if nan entry in sample['depth']: exit()

            if torch.isnan(sample['depth']).any():
                print('nan in depth transform')
                ## get the indices of the nan
                #nan_indices = torch.isnan(sample['depth'])
                # save the plot of the rgb and depth
                if 'rgb' in sample:
                    rgb = (sample['rgb'].cpu().numpy()*255).astype(np.uint8)
                    cv2.imwrite('tmp/nan_rgb.png', rgb)
                if 'depth' in sample:
                    depth = (sample['depth'].cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite('tmp/nan_depth.png', depth)
                

        if 'maskout' in self.config:
            sample['mask'] = (sample['mask'].float() > 0.99).float()
            bg_value = self.config['bg_value'] if 'bg_value' in self.config else 0.0
            #print('bg_value', bg_value)
            sample['rgb'] = sample['rgb'] * sample['mask'] + (1 - sample['mask']) * bg_value
            if 'depth' in sample:
                sample['depth'] = sample['depth'] * sample['mask']
        
        # if True:
        #     from matplotlib import pyplot as plt
        #     if 'rgb' in sample:
        #         plt.imshow(sample['rgb'].cpu().numpy())
        #         plt.axis('off')
        #         plt.savefig('tmp/mask-rgb.png', bbox_inches='tight', pad_inches=0)
        #     if 'depth' in sample:
        #         depth = sample['depth'].cpu().numpy()
        #         norm_depth = (depth - depth.min()) / (depth.max() - depth.min())
        #         color_depth = cv2.applyColorMap((norm_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
        #         cv2.imwrite('tmp/mask-depth.png', color_depth)
            
        #     import time
        #     time.sleep(1)
                #plt.savefig('tmp/mask-depth.png')
            #plt.close()

        # from matplotlib import pyplot as plt
        # plt.imshow(sample['depth'].cpu().numpy())
        # plt.savefig('mask-depth.png')

        # plt.imshow(sample['mask'].cpu().numpy())
        # plt.savefig('mask.png')

        # if True:
        #     rgb = (sample['rgb'].cpu().numpy() * 255).astype(np.uint8)
        #     print('post rgb shape', rgb.shape)
        #     H, W = rgb.shape[:2]
        #     action = sample['action'].cpu().numpy()
        #     print('action', action)
        #     pick = (action[:, :2] + 1)/2 * np.array([W, H])
        #     pick = (int(pick[0, 0]), int(pick[0, 1]))
        #     place = (action[:, 2:] + 1)/2 * np.array([W, H])
        #     place = (int(place[0, 0]), int(place[0, 1]))
        #     pnp_rgb = draw_pick_and_place(
        #         rgb, pick, place, get_ready=True, swap=True
        #     )
        #     cv2.imwrite('tmp/post_pnp_rgb.png', pnp_rgb)


        if 'action' in sample.keys() and 'swap_action' in self.config and self.config.swap_action:
            sample['action'] = sample['action'][:, [1, 0, 3, 2]]

        for k, v in sample.items():
            sample[k] = ts_to_np(v)

        return sample

    def postprocess(self, sample):
        if 'rgb' in sample:
            #print('here!!!')
            shapes = sample['rgb'].shape
            sample['rgb'] = postprocess_rgb(
                sample['rgb'], 
                normalise={
                    'mode': self.config.rgb_norm_mode, 
                    'param': self.config.rgb_norm_param})
            if sample['rgb'].shape != shapes:
                sample['rgb'] = sample['rgb'].reshape(shapes)
        return sample