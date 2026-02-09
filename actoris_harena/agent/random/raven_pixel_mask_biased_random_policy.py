
import numpy as np
from actoris_harena import Agent

class RavenPixelMaskBiasedRandomPolicy(Agent):
    """
    Random Policy for RavenPixelEnvAdapter.
    
    Logic:
    1. Pick: Biased towards object masks (random pixel on a random object).
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
        # 1. Retrieve Segmentation Mask
        # The RavenPixelEnvAdapter provides 'segm' in the observation dict.
        # segm contains object IDs. 0 is usually background/table.
        obs = info['observation']
        if 'segm' not in obs:
            # Fallback if segm isn't available (shouldn't happen with correct adapter)
            return np.zeros(5, dtype=np.float32)

        segm = obs['segm'] # Shape: (H, W)
        height, width = segm.shape

        # --- PICK LOGIC (Mask Biased) ---
        
        # Get unique object IDs excluding background (0)
        valid_obj_ids = np.unique(segm)
        valid_obj_ids = valid_obj_ids[valid_obj_ids > 0]
        
        if len(valid_obj_ids) > 0:
            # Select a random object
            target_id = np.random.choice(valid_obj_ids)
            
            # Find all pixel coordinates (v, u) = (row, col) belonging to this object
            # np.where returns tuple of arrays (row_indices, col_indices)
            obj_pixels = np.where(segm == target_id)
            
            # Select a random pixel index
            idx = np.random.randint(len(obj_pixels[0]))
            
            pick_v = obj_pixels[0][idx] # Row (y-axis in image)
            pick_u = obj_pixels[1][idx] # Col (x-axis in image)
            
            # Normalize to [-1, 1]
            # Formula: (coord / size * 2) - 1
            n_pick_u = (pick_u / width * 2) - 1.0
            n_pick_v = (pick_v / height * 2) - 1.0
            
        else:
            # Scene is empty: Pick completely random
            n_pick_u = np.random.uniform(-1, 1)
            n_pick_v = np.random.uniform(-1, 1)

        # --- PLACE LOGIC (Random) ---
        
        # Place anywhere in the image bounds [-1, 1]
        n_place_u = np.random.uniform(-1, 1)
        n_place_v = np.random.uniform(-1, 1)

        # --- ROTATION LOGIC (Random) ---
        
        # Random rotation in normalized range [-1, 1] (representing -pi to pi)
        n_theta = np.random.uniform(-1, 1)

        # --- CONSTRUCT ACTION ---
        action = np.array([n_pick_u, n_pick_v, n_place_u, n_place_v, n_theta], dtype=np.float32)
        
        # Clip to be safe
        return np.clip(action, -1.0, 1.0)
        
    def init(self, infos):
        pass

    def update(self, infos, actions):
        pass

    def get_state(self):
        return {}