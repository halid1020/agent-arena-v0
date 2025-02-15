import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from agent_arena.agent.utilities.torch_utils import np_to_ts, ts_to_np

from agent_arena.utilities.transform.utils import *


class ContrastiveLearningTransformer:
    def __init__(self,  config=None):
        self.config = config
    
    # We assume the shape B*C*H*W, RGB images
    def __call__(self, images):

        images = np_to_ts(images, self.config.device)
        ret_images = F.interpolate(
                    images,
                    size=self.config.img_dim, 
                    mode='bilinear', align_corners=False).to(self.config.device)

        ## Add blurring, the value has to be between 0 and 255
        if self.config.rgb_blur:
            ret_images = ret_images/255.0
            guassian_blur = torchvision.transforms.GaussianBlur(
                    kernel_size=self.config.rgb_blur_kernel_size,
                    sigma=self.config.rgb_blur_sigma,
                )
            
            ret_images = guassian_blur(ret_images)*255.0
            # print('ret image shape', ret_images.shape)
            # print('ret image range', ret_images.min(), ret_images.max())
        
        # Add color jitter, the value has to be between 0 and 255
        if self.config.rgb_color_jitter:
            ret_images = ret_images/255.0
            
            jitter = torchvision.transforms.ColorJitter(
                    brightness=self.config.rgb_color_jitter_brightness,
                    contrast=self.config.rgb_color_jitter_contrast,
                    saturation=self.config.rgb_color_jitter_saturation,
                    hue=self.config.rgb_color_jitter_hue)
            

            ret_images = jitter(ret_images)

            ret_images = ret_images*255.0

            #print('ret range', ret_images.min(), ret_images.max())

        ## Plot the first image
        # import matplotlib.pyplot as plt
        # for i in range(3):
        #     plt.imshow(ret_images[i].cpu().numpy().transpose(1, 2, 0)/255)
        #     plt.show()




        ret_images = preprocess_rgb(
            ret_images,
            normalise={
                'mode': self.config.rgb_norm_mode,
                'param': self.config.rgb_norm_param
            },
            noise_factor=self.config.rgb_noise_var,
        )
        
        
        return ret_images