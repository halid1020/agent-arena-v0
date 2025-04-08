import numpy as np
from scipy import ndimage

from  agent_arena import Agent

class GarmentMaskBiasedPixelPickAndPlacePolicy(Agent):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'garment-mask-biased-pixel-pick-and-place'

    def reset(self, arena_ids):
        self.internal_states = {arena_id: {} for arena_id in arena_ids}

    def init(self, infos):
        pass

    def update(self, infos, actions):
        pass

    def act(self, infos, update=False):
        actions = []
        for info in infos:
            action = self._mask_biased_random_pick_and_place(info)
            actions.append(action)
        return actions
    
    def _mask_biased_random_pick_and_place(self, info):
        mask = info['observation']['mask']
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        rgb = info['observation']['rgb']
        
        mask_coords = np.argwhere(mask)
        if len(mask_coords) == 0:
            return {
                'norm-pixel-pick-and-place': {
                    'pick_0': np.random.uniform(-1, 1, 2),
                    'place_0': np.random.uniform(-1, 1, 2)
                }
            }
        
        pick_pixel = mask_coords[np.random.randint(len(mask_coords))]
        pick_pixel = pick_pixel.astype(np.float32)
        pick_pixel[0] = pick_pixel[0] / mask.shape[0] * 2 - 1
        pick_pixel[1] = pick_pixel[1] / mask.shape[1] * 2 - 1
        
        action = {
            'norm-pixel-pick-and-place': {
                'pick_0': pick_pixel,
                'place_0': np.random.uniform(-1, 1, 2)
            }
        }
        action['pick_0'] = action['norm-pixel-pick-and-place']['pick_0']
        action['place_0'] = action['norm-pixel-pick-and-place']['place_0']
        
        return action

    
