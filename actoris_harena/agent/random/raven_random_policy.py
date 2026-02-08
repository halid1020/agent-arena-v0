import numpy as np
from actoris_harena import Agent
from actoris_harena.agent.bc.transporter.utils import utils

class RavenRandomPolicy(Agent):

    def __init__(self, config):
        super().__init__(config)
        
    def single_act(self, info, update=False):
        arena = info['arena']
        env = arena._env
        task = arena._task

        # 1. Get Scene Data (Heightmap & Segmentation)
        # We use the task helper to get the orthographic projection (top-down)
        # which makes coordinate conversion (Pixel -> World) strictly accurate.
        cmap, hmap, obj_mask = task.get_true_image(env)
        
        # --- PICK LOGIC (Mask Biased) ---
        
        # Get all object IDs present in the scene (excluding background 0)
        valid_obj_ids = np.unique(obj_mask)
        valid_obj_ids = valid_obj_ids[valid_obj_ids > 0]
        
        # Fallback if scene is empty
        if len(valid_obj_ids) == 0:
            return np.zeros(14)

        # Select one random object from the scene
        target_id = np.random.choice(valid_obj_ids)
        
        # Create a specific probability mask for this object
        target_mask = (obj_mask == target_id).astype(np.float32)
        
        # Sample a pixel (u, v) strictly from this object's mask
        pick_pix = utils.sample_distribution(target_mask)
        
        # Convert pixel to World Position (x, y, z)
        pick_pos = utils.pix_to_xyz(pick_pix, hmap, task.bounds, task.pix_size)
        
        # Default top-down pick orientation (Identity quaternion)
        pick_rot = (0, 0, 0, 1)


        # --- PLACE LOGIC (Random Scene) ---
        
        # Generate random X, Y within the workspace bounds
        # bounds shape: [[minX, maxX], [minY, maxY], [minZ, maxZ]]
        bounds = task.bounds
        place_x = np.random.uniform(bounds[0, 0], bounds[0, 1])
        place_y = np.random.uniform(bounds[1, 0], bounds[1, 1])
        
        # For Z, we can use the object's pick height or a safe drop height.
        # Here we reuse the pick height to avoid smashing into the floor.
        place_z = pick_pos[2] 
        place_pos = (place_x, place_y, place_z)

        # Generate random Z-axis rotation for placement
        theta = np.random.rand() * 2 * np.pi
        place_rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))


        # --- CONSTRUCT ACTION ---
        # 14 values: [Pick Pos (3), Pick Rot (4), Place Pos (3), Place Rot (4)]
        ret_action = np.concatenate([
            pick_pos, pick_rot, 
            place_pos, place_rot
        ])

        return ret_action
        
    def init(self, infos):
        # No internal state initialization required for random policy
        pass

    def update(self, infos, actions):
        pass

    def get_state(self):
        return {}