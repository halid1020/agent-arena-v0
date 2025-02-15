import numpy as np
from scipy import ndimage

from  agent_arena import Agent

class GarmentBorderBiasedPixelPickAndPlacePolicy(Agent):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'garment-border-biased-pixel-pick-and-place'

    def reset(self, arena_ids):
        pass

    def init(self, infos):
        pass

    def update(self, infos, actions):
        pass

    def act(self, infos, update=False):
        actions = []
        for info in infos:
            action = self._border_biased_random_pick_and_place(info)
            actions.append(action)
        return actions
    
    def _border_biased_random_pick_and_place(self, info):
        mask = info['observation']['mask']
        eroded_mask = ndimage.binary_erosion(mask, iterations=2)
        border = mask^eroded_mask
        ## plot the original mask and the border

        # from matplotlib import pyplot as plt
        # plt.imsave('mask.png', mask)
        # plt.imsave('border.png', border)
        
        border_coords = np.argwhere(border)
        if len(border_coords) == 0:
            return {
                'norm-pixel-pick-and-place': {
                    'pick_0': np.random.uniform(-1, 1, 2),
                    'place_0': np.random.uniform(-1, 1, 2)
                }
            }
        
        pick_pixel = border_coords[np.random.randint(len(border_coords))].astype(np.float32)
        pick_pixel[0] = pick_pixel[0] / mask.shape[0] * 2 - 1
        pick_pixel[1] = pick_pixel[1] / mask.shape[1] * 2 - 1
        pick_pixel = pick_pixel[::-1]

        return {
            'norm-pixel-pick-and-place': {
                'pick_0': pick_pixel,
                'place_0': np.random.uniform(-1, 1, 2)
            }
        }

    
