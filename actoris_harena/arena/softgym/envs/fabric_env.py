import cv2
import numpy as np
import logging
from softgym.registered_env import SOFTGYM_ENVS

from .cloth_env import ClothEnv

class Points:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class FabricEnv(ClothEnv):
    _name = 'softgym_fabric_env'


    def __init__(self, kwargs):
        super().__init__(kwargs)
        softgym_parameters = {
            'action_mode': "pickerpickplace",
            'action_repeat': 1,
            'picker_radius': 0.015,
            'camear_name': "default_camera",
            'render_mode': "cloth",
            'num_picker': 1,
            'render': False, ### Need to figure this out
            'cloth_dim': (0.4, 0.4),
            'particle_radius': 0.00625,


            ## Allow to Change Variabels but from kwargs
            'observation_mode': {"image": "cam_rgb"},
            'observation_image_shape': (256, 256, 3),
            'num_variations': 1000, 
            'random_seed': 0,
            'save_step_info': False, 
            #'save_image_dim': (256, 256),
            'use_cached_states': True,
            'headless': True,

            'context_cloth_colour': False
        }

        self._observation_image_shape = kwargs['observation_image_shape'] \
            if 'observation_image_shape' in kwargs.keys() else (256, 256, 3)

        for k, v in kwargs.items():
            softgym_parameters[k] = v

        
        self._env = SOFTGYM_ENVS["FabricEnv"](**softgym_parameters)
        self.camera_height = self._env.camera_height
        self.camera_intrinsic_matrix = self._env.camera_intrinsic_matrix
        self.camera_extrinsic_matrix = self._env.camera_extrinsic_matrix
        self.camera_size = self._env.camera_size
        self.camera_to_world_ratio = self._env.pixel_to_world_ratio
        self.horizon = self._env.control_horizon
        #logging.info('[softgym, fabric-velocity-control] action_space {}'.format(self._env.action_space))
        self.no_op = np.zeros(self.get_action_space().shape)
        self.no_op[-1] = -1.0
        self.info_keys = ['contour', 'cloth_size', 
                          'corner_positions', 'corner_visibility']

    def get_no_op(self):
        return self.no_op

    def observation_shape(self):
        return {'rgb': self._observation_image_shape, 
                'depth': self._observation_image_shape}
    
    def get_corner_positions(self):
        return self._env.get_corner_positions()
    

    def get_cloth_edge_mask(self, camera_name='default_camera', resolution=(64, 64)):
        edge_ids = self._env.get_edge_ids()
        positions = self._env.get_particle_positions()
        edge_positions = positions[edge_ids]
        #print('edge positions', edge_positions)
        visbility, proj_pos = self._get_visibility(edge_positions, resolution=(128, 128))
        visbility = visbility[0]
        proj_pos = proj_pos[0]

        edge_mask = np.zeros(resolution)
        for i in range(edge_positions.shape[0]):
            if visbility[i]:
                y = int((proj_pos[i][0] + 1)/2 * resolution[0])
                y = max(0, min(y, resolution[0]-1))
                x = int((proj_pos[i][1] + 1)/2 * resolution[1])
                x = max(0, min(x, resolution[1]-1))
                edge_mask[x, y] = 1
       
        return edge_mask

    def get_contour(self, camera_name='default_camera', resolution=(128, 128)):
        cloth_edge_mask = self.get_cloth_edge_mask(resolution=resolution)
        cloth_mask = self._get_cloth_mask(resolution=resolution)
        cloth_countor = np.zeros_like(cloth_mask)
        for i in range(1, cloth_mask.shape[0]-1):
            for j in range(1, cloth_mask.shape[1]-1):
                if cloth_mask[i, j] == 1:
                    if np.sum(cloth_mask[i-1:i+2, j-1:j+2]) == 9:
                        cloth_countor[i, j] = 0
                    else:
                        cloth_countor[i, j] = 1
        cloth_countor = (cloth_countor + cloth_edge_mask).clip(0, 1)
        return cloth_countor
    
    
    
    def get_edge_positions(self):
        return self._env.get_edge_positions()
    
    def get_corner_ids(self):
        return self._env.get_corner_ids()
    
    def get_edge_ids(self):
        return self._env.get_edge_ids()
    
    def get_cloth(self):
        positions = self._env.get_particle_positions()
        ret = []
        for p in positions:
            ret.append(Points(p[0], p[2], p[1]))
        
        return ret
    
    def get_cloth_dim(self):
        return self._env.get_cloth_dim()
    
    def get_name(self, episode_config=None):
        return self._name
    
    def get_id(self):
        return self.id
    
import os
import ray

@ray.remote(num_gpus=0.05)
class FabricEnvRay(FabricEnv):
    
    def __init__(self, config):
        super().__init__(config)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU ID
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Make CUDA calls synchronous