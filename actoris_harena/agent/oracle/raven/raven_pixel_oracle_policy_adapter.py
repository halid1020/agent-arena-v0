import os
import numpy as np
import pybullet as p
from actoris_harena import Agent

ENV_ASSETS_DIR = os.environ.get("RAVENS_ASSETS_DIR", "")

class RavenPixelOraclePolicyAdapter(Agent):
    """
    Oracle Policy that outputs normalized pixel-based actions (5-dim).
    Format: [pick_u_norm, pick_v_norm, place_u_norm, place_v_norm, theta_norm]
    Range: All values are in [-1, 1].
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "raven-pixel-oracle"
        self._policy = None

    def get_name(self):
        return self.name

    def _project_to_pixel(self, position, cam_config):
        """
        Projects a 3D World position (x, y, z) to 2D Pixel coordinates (u, v).
        Inverse of the Arena's deprojection logic.
        """
        # Unpack camera params
        intrinsics = cam_config['intrinsics']
        fx = intrinsics[0]
        fy = intrinsics[4]
        cx = intrinsics[2]
        cy = intrinsics[5]
        
        cam_pos = np.array(cam_config['position'])
        cam_rot_quat = cam_config['rotation']
        
        # 1. World -> Camera Frame
        # P_cam = R_inv * (P_world - T)
        # Calculate R (Rotation Matrix)
        rot_matrix = np.array(p.getMatrixFromQuaternion(cam_rot_quat)).reshape(3, 3)
        
        # P_world - T
        rel_pos = np.array(position) - cam_pos
        
        # Apply Inverse Rotation (Transpose for orthogonal matrix)
        p_cam = rot_matrix.T @ rel_pos
        
        x_c, y_c, z_c = p_cam

        # 2. Camera Frame -> Pixel Space (Pinhole Model)
        # u = (x_c * fx / z_c) + cx
        # v = (y_c * fy / z_c) + cy
        
        if z_c == 0: 
            return 0, 0 # Avoid division by zero
            
        u = (x_c * fx / z_c) + cx
        v = (y_c * fy / z_c) + cy
        
        return u, v

    def _normalize_action(self, pixel_action, img_res):
        """
        Converts pixel coordinates and radians to [-1, 1] range.
        Args:
            pixel_action: [pick_u, pick_v, place_u, place_v, theta_rad]
        Returns:
            norm_action: [pick_u_n, pick_v_n, place_u_n, place_v_n, theta_n]
        """
        pick_u, pick_v, place_u, place_v, theta_rad = pixel_action
        
        # Normalize Pixels: [0, img_res] -> [-1, 1]
        # Formula: (coord / img_res * 2) - 1
        
        n_pick_u = (pick_u / img_res * 2) - 1.0
        n_pick_v = (pick_v / img_res * 2) - 1.0
        n_place_u = (place_u / img_res * 2) - 1.0
        n_place_v = (place_v / img_res * 2) - 1.0 
        
        # Normalize Rotation: [-pi, pi] -> [-1, 1]
        n_theta = theta_rad / np.pi
        
        # Clip to ensure bounds
        return np.clip([n_pick_u, n_pick_v, n_place_u, n_place_v, n_theta], -1.0, 1.0)

    def single_act(self, info, update=False):
        arena = info['arena']
        arena_id = info['arena_id']
        
        # 1. Initialize Oracle if needed
        if arena_id not in self.internal_states:
            self.internal_states[arena_id] = {}
            self.internal_states[arena_id]['policy'] = arena._task.oracle(arena._env)
        
        policy = self.internal_states[arena_id]['policy']
        
        # 2. Get Oracle Action (World Space Dict)
        # {'pose0': (pos, rot), 'pose1': (pos, rot)}
        # Oracle uses internal env state, so 'obs' argument is often unused but required
        obs = info['observation'] if 'observation' in info else info
        action_dict = policy.act(obs, None)
        
        if action_dict is None:
            # No-op or Done: Return zeros
            return np.zeros(5, dtype=np.float32)

        # 3. Extract World Coordinates
        pick_pos = action_dict['pose0'][0]  # (x, y, z)
        place_pos = action_dict['pose1'][0] # (x, y, z)
        place_rot = action_dict['pose1'][1] # (qx, qy, qz, qw)
        
        # 4. Project World -> Pixel
        # We need the camera config used by the arena
        # Assuming the first camera is the agent's top-down camera
        cam_config = arena._env.agent_cams[0]
        img_res = cam_config['image_size'][0]
        
        pick_u, pick_v = self._project_to_pixel(pick_pos, cam_config)
        place_u, place_v = self._project_to_pixel(place_pos, cam_config)
        
        # 5. Convert Rotation -> Theta (Radians around Z)
        # Convert Place Quaternion to Euler [roll, pitch, yaw]
        place_euler = p.getEulerFromQuaternion(place_rot)
        theta_rad = place_euler[2] # Yaw (Z-axis rotation)
        
        # 6. Normalize
        pixel_action = [pick_u, pick_v, place_u, place_v, theta_rad]
        norm_action = self._normalize_action(pixel_action, img_res)
        
        return norm_action

    def init(self, infos):
        for info in infos:
            arena = info['arena']
            arena_id = info['arena_id']
            if arena_id not in self.internal_states:
                self.internal_states[arena_id] = {}
            self.internal_states[arena_id]['policy'] = arena._task.oracle(arena._env)

    def update(self, infos, actions):
        pass

    def get_state(self):
        return {}