from softgym.registered_env import SOFTGYM_ENVS
import numpy as np

from agent_arena.arena.softgym.envs.cloth_env \
    import ClothVelocityControlEnv

class GarmentVelocityControlEnv(ClothVelocityControlEnv):
    _name = 'SoftGymGarmentEnv'


    def __init__(self, kwargs):
        #self.hoirzon = kwargs['horizon']
        super().__init__(kwargs)
        
        self.eval_para = {
            'eval_tiers': kwargs['eval_tiers'],
            'video_episodes': kwargs['video_episodes']
        }
        
        
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
            'action_mode': 'pickerpickplace',
            'reward_mode': 'hoque_ddpg',
            'num_variations': 1000, 
            'random_seed': 0,
            'save_step_info': False, 
            'save_image_dim': (256, 256),
            'use_cached_states': True,
            'action_horizon': None,
            'control_horizon': 1000,
            'motion_trajectory': 'triangle',
            'pick_height': 0.026, 
            'place_height': 0.06,
            'end_trajectory_move': True,
            'intermidiate_height': 0.15,
            # 'picker_low': [-1, -1, -1, -1],
            # 'picker_high': [1, 1, 1, 1],
            'headless': True,

            'context_cloth_colour': False
        }

        for k, v in kwargs.items():
            softgym_parameters[k] = v
        
        self._observation_image_shape = softgym_parameters['observation_image_shape']
        
        print('Observation shape', self._observation_image_shape)

        self._env = SOFTGYM_ENVS["Garment"](**softgym_parameters)
        self._name = 'SoftGymGarmentEnv'
        self.pixel_to_world_ratio = self._env.pixel_to_world_ratio
        self.camera_height = self._env.camera_height
        self.info_keys = []
        self.no_op = np.zeros(self.get_action_space().shape)
        self.no_op[-1] = -1.0
        #print('Finished Init Garment Velocity Control Env')

    def get_no_op(self):
        return self.no_op
    
    def is_flattened(self):
        ### TODO: I am not sure if this is the right way to check if the garment is flatten.
        return self.get_normalised_coverage() > 0.98 and self.get_wrinkle_ratio() < 0.02

    def get_name2keypoints(self):
        return self._env.get_name2keypoints()
    
    
    
    def get_flatten_keypoint_positions(self):
        return self._env.get_flatten_keypoint_positions()
    
    
    # def get_cloth_size(self):
    #     ### TODO: cloth size is hard to decide for different garments.
    #     return self._env.get_cloth_size()

    
    def get_flatten_corner_positions(self):
        ### TODO: we need to specify corner ids for each garment.
        return self._env.get_flatten_corner_positions()
    
    def observation_shape(self):
        return {'rgb': self._observation_image_shape, 
                'depth': self._observation_image_shape}

    def get_name(self):
        return self._name
