import numpy as np
from scipy import ndimage
from actoris_harena import Agent

class MaskBiasedPixelPickAndPlacePolicy(Agent):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'mask-biased-pixel-random-pick-and-place'

    def reset(self, arena_ids):
        self.internal_states = {arena_id: {} for arena_id in arena_ids}

    def init(self, infos):
        pass

    def update(self, infos, actions):
        pass

    def act(self, infos, update=False):
        # Return a list of numpy arrays (vectorized)
        actions = [self.single_act(info) for info in infos]
        return actions
    
    def single_act(self, info, update=False):
        mask = info['observation']['mask']
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        mask_coords = np.argwhere(mask)
        
        # 1. Determine Pick Pixel (Normalized -1 to 1)
        if len(mask_coords) == 0:
            pick_pixel = np.random.uniform(-1, 1, 2)
        else:
            # Pick a random point where the mask is positive
            pick_idx = np.random.randint(len(mask_coords))
            pick_raw = mask_coords[pick_idx].astype(np.float32)
            
            # Normalize to [-1, 1]
            # pick_raw[0] is Row (Y), pick_raw[1] is Col (X)
            py = (pick_raw[0] / mask.shape[0]) * 2 - 1
            px = (pick_raw[1] / mask.shape[1]) * 2 - 1
            pick_pixel = np.array([py, px], dtype=np.float32)
        
        # 2. Determine Place Pixel (Random -1 to 1)
        place_pixel = np.random.uniform(-1, 1, 2).astype(np.float32)
        
        # 3. Concatenate into a single vector [py, px, ty, tx]
        vector_action = np.concatenate([pick_pixel, place_pixel])
        
        return vector_action