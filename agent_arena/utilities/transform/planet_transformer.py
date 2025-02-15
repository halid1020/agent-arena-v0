import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from agent_arena.agent.utilities.torch_utils import np_to_ts, ts_to_np


def preprocess_rgb(observation, bit_depth=5, noise_factor=0.0):
    device = observation.device
    
    #if noise_factor != 0:
    #print('noise factor', noise_factor)
    observation = torch.floor_divide(observation, 2 ** (8 - bit_depth)).float() / (2 ** bit_depth) - 0.5

    noise_tensor = torch.randn(observation.shape, device=device)*noise_factor / (2 ** bit_depth)
    observation = observation + noise_tensor

    observation = observation.clamp(-0.5, 0.5)
    # else:
    #     observation = observation/255.0

    # if remap_image is not None:

    #     observation = observation * (remap_image[1] - remap_image[0]) + remap_image[0]
        
    return observation

def postprocess_rgb(observation):
    return (observation + 0.5)*255

class PlaNetTransformer:
    def __init__(self,  config=None):
        self.config = config
    

    def __call__(self, sample, train=True, to_tensor=True, single=False):
        # batch is assumed to have the shape B*T*C*H*W
        for k, v in sample.items():
            sample[k] = np_to_ts(v, self.config.device)
            if single:
                sample[k] = sample[k].unsqueeze(0)
        
        obs = 'rgb'
        B, T, C, H, W = sample[obs].shape
        #print('shape', sample[obs].shape)
        # print(obs, sample[obs].shape)
        sample[obs] = F.interpolate(
            sample[obs].reshape(B*T, C, H, W),
            size=self.config.img_dim, mode='bilinear', align_corners=False)\
                .reshape(B, T, C, *self.config.img_dim)

        
        sample['rgb'] = preprocess_rgb(
            sample['rgb'],
            self.config.bit_depth, 
            noise_factor=(self.config.rgb_noise_var if train else 0))
        
                
            
        if self.config.reward_scale and train:
            sample['reward'] *= self.config.reward_scale
    
        if single:
            for k, v in sample.items():
                sample[k] = v.squeeze(0)

        if not to_tensor:
            for k, v in sample.items():
                sample[k] = ts_to_np(v)
        
        return sample
    
    def post_transform(self, sample):
        
        res = {}
        for k, v in sample.items():
            res[k] = ts_to_np(v)
            #print(k, res[k].shape)

        res['rgb'] = postprocess_rgb(res['rgb'])
        
        return res