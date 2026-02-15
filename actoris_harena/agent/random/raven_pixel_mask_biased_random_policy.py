import numpy as np
from actoris_harena import Agent

class RavenPixelMaskBiasedRandomPolicy(Agent):
    """
    Random Policy for RavenPixelEnvAdapter.
    
    Logic:
    1. Pick: Biased towards valid pixels in obs['mask'].
    2. Place: Completely random pixel in the workspace.
    3. Rotation: Random rotation.
    
    Output:
    Normalized 5-dim action: [pick_u, pick_v, place_u, place_v, theta]
    Range: [-1, 1]
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "raven-pixel-random"
        
    def get_name(self):
        return self.name

    def single_act(self, info, update=False):
        # 1. Retrieve Observation
        obs = info['observation']
        
        # Initialize a completely random action first [u, v, u, v, theta]
        action = np.random.uniform(-1, 1, size=5).astype(np.float32)

        # 2. Pick Logic (Mask Biased)
        if 'mask' in obs:
            mask = obs['mask'] # Shape: (H, W), usually binary 0 or 1
            height, width = mask.shape

            # Find all pixel coordinates (row, col) where mask > 0
            # np.where returns tuple of arrays (row_indices, col_indices)
            valid_pixels = np.where(mask > 0)
            num_valid = len(valid_pixels[0])
            
            if num_valid > 0:
                # Select a random valid pixel index
                idx = np.random.randint(num_valid)
                
                pick_v = valid_pixels[0][idx] # Row (y-axis)
                pick_u = valid_pixels[1][idx] # Col (x-axis)
                
                # Normalize to [-1, 1]
                # u corresponds to width (x), v corresponds to height (y)
                n_pick_u = (pick_u / width * 2) - 1.0
                n_pick_v = (pick_v / height * 2) - 1.0
                
                # Overwrite the random pick coordinates
                action[0] =  n_pick_v 
                action[1] = n_pick_u

        # 3. Return Action
        # Place and Rotation remain completely random from initialization
        return np.clip(action, -1.0, 1.0)
        
    def init(self, infos):
        pass

    def update(self, infos, actions):
        pass

    def get_state(self):
        return {}