import os
import numpy as np
import gym
import pybullet as p
import cv2

from .raven_env_adapter import RavenEnvAdapter 

class RavenPixelEnvAdapter(RavenEnvAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.last_obs = None
        
        # Cache camera configuration
        self.cam_config = self._env.agent_cams[0]
        self.img_res = self.cam_config['image_size'][0]
        
        # Pre-calculate camera parameters for deprojection
        intrinsics = self.cam_config['intrinsics']
        self.fx = intrinsics[0]
        self.fy = intrinsics[4]
        self.cx = intrinsics[2]
        self.cy = intrinsics[5]
        
        rotation = p.getMatrixFromQuaternion(self.cam_config['rotation'])
        self.cam_rotm = np.float32(rotation).reshape(3, 3)
        self.cam_pos = np.array(self.cam_config['position'])

        self.debug = config.get('debug', False)
        self.debug_dir = config.get('debug_dir', 'tmp/raven_debug')
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"[RavenPixel] Debugging enabled. Images will be saved to: {self.debug_dir}")

    def get_name(self):
        return "RavenPixelNormalized"
    
    def _visualize_action(self, pixel_action, step_idx):
        """Draws the pick and place action on the current RGB frame."""
        # 1. Get current RGB image (H, W, 3)
        img = self.last_obs['rgb'].copy() 
        if img.shape[-1] == 4:
            img = img[:, :, :3]
            
        # 2. Extract Coordinates (Integers for cv2)
        u1, v1 = int(pixel_action[0]), int(pixel_action[1]) # Pick
        u2, v2 = int(pixel_action[2]), int(pixel_action[3]) # Place
        theta = pixel_action[4]

        # 3. Draw Markers
        # Pick: Red Circle
        cv2.circle(img, (u1, v1), 4, (255, 0, 0), -1)
        
        # Place: Green Circle
        cv2.circle(img, (u2, v2), 4, (0, 255, 0), -1)
        
        # Trajectory: Yellow Arrow from Pick -> Place
        cv2.arrowedLine(img, (u1, v1), (u2, v2), (255, 255, 0), 1, tipLength=0.1)

        # Orientation: Blue Line at Place
        r_len = 15 
        end_x = int(u2 + r_len * np.cos(theta))
        end_y = int(v2 + r_len * np.sin(theta))
        cv2.line(img, (u2, v2), (end_x, end_y), (0, 0, 255), 2)

        # 4. Save Image (Convert RGB -> BGR for OpenCV)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        filename = os.path.join(self.debug_dir, f"step_{step_idx:04d}_action.png")
        cv2.imwrite(filename, img_bgr)

    def _deproject_pixel(self, u, v, depth_map):
        """Converts pixel coordinates (u, v) + depth to World (x, y, z)."""
        h, w = depth_map.shape
        
        # Ensure integer coordinates for array indexing
        u_clamped = int(np.clip(u, 0, w - 1))
        v_clamped = int(np.clip(v, 0, h - 1))
        
        z_cam = depth_map[v_clamped, u_clamped]

        # Pinhole Camera Model
        x_c = (u - self.cx) * z_cam / self.fx
        y_c = (v - self.cy) * z_cam / self.fy
        
        p_cam = np.array([x_c, y_c, z_cam])
        p_world = self.cam_pos + self.cam_rotm @ p_cam
        
        return p_world

    def _denormalize_action(self, norm_action):
        """
        Converts normalized action [-1, 1] to pixel space and radians.
        
        Args:
            norm_action: [pick_x, pick_y, place_x, place_y, theta] in range [-1, 1]
        Returns:
            pixel_action: [pick_u, pick_v, place_u, place_v, theta_rad]
        """
        # Clip to ensure safety
        norm_action = np.clip(norm_action, -1.0, 1.0)
        
        # 1. Pixel Coordinates: Map [-1, 1] -> [0, img_res]
        # Formula: (x + 1) / 2 * size
        pixel_coords = (norm_action[:4] + 1) / 2 * self.img_res
        
        # 2. Rotation: Map [-1, 1] -> [-pi, pi]
        # Formula: x * pi
        theta = norm_action[4] * np.pi
        
        return np.concatenate([pixel_coords, [theta]])

    def _convert_pixel_action_to_world(self, pixel_action):
        """
        Converts 5-dim [pick_u, pick_v, place_u, place_v, theta] 
        to 14-dim [pick_pos, pick_rot, place_pos, place_rot].
        """
        pick_u, pick_v, place_u, place_v, theta = pixel_action
        
        depth_map = self.last_obs['depth']
        
        # Calculate 3D Positions
        pick_pos = self._deproject_pixel(pick_u, pick_v, depth_map)
        place_pos = self._deproject_pixel(place_u, place_v, depth_map)
        
        # Calculate Rotations
        pick_rot = np.array([0, 0, 0, 1]) # Identity
        place_rot = np.array(p.getQuaternionFromEuler([0, 0, theta]))
        
        return np.concatenate([pick_pos, pick_rot, place_pos, place_rot])

    def reset(self, episode_config=None):
        info = super().reset(episode_config)
        self.last_obs = info['observation']
        return info

    def step(self, action):
        # 1. Check if action is the 5-dim normalized action
        if isinstance(action, (np.ndarray, list)) and len(action) == 5:
            # 2. Denormalize: [-1, 1] -> [Pixels, Radians]
            pixel_action = self._denormalize_action(action)
            
            if self.debug and self.last_obs is not None:
                self._visualize_action(pixel_action, self._step)

            # 3. Convert: [Pixels, Radians] -> [World XYZ, Quat]
            world_action = self._convert_pixel_action_to_world(pixel_action)
        else:
            raise ValueError
            
        info = super().step(world_action)
        self.last_obs = info['observation']
        return info

    def get_action_space(self):
        # Normalized Space: [-1, 1] for all 5 dimensions
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)