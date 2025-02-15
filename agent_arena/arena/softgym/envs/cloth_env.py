import cv2
import numpy as np
import torch
from gym.spaces import Box

from agent_arena.arena.arena import Arena


class ClothEnv(Arena):
    _name = 'SoftGymClothEnv'


    def __init__(self, config):
        #self.hoirzon = kwargs['horizon']
        self.eval_para = {
            'eval_tiers': config['eval_tiers'],
            'video_episodes': config['video_episodes']
        }
        
        self.eval_params = [
            {
                'eid': eid,
                'tier': tier,
                'save_video': (eid in config['video_episodes'])
            }

            for tier in config['eval_tiers'] for eid in config['eval_tiers'][tier]
        ]

        self.val_params = [
            {
                'eid': eid,
                'save_video': False
            }
            for eid in config['val_episodes']
        ]
        
        self._env = None
        self.action_mode = 'velocity-grasp'
        self._train = True
        from .pick_and_place_rect_fabric_single_task_logger \
            import PickAndPlaceRectFabricSingleTaskLogger
        self.logger = PickAndPlaceRectFabricSingleTaskLogger()
        self.set_id(0)

    def get_goal(self):
        return self.task.get_goal(self)
        
    def set_task(self, task):
        self.task = task 
    
    def set_id(self, id):
        self.id = id
        
    def get_num_picker(self):
        return self._env.get_num_picker()
    
    def set_to_flatten(self):
        info = self._env.set_to_flatten()
        return self._process_info(info)

    def _get_particle_positions(self):
        return self.get_object_positions()
    
    def get_picker_pos(self):
        return self._env.get_picker_pos()

    def set_action_tool(self, action_tool):
        self.action_tool = action_tool

    def get_picker_position(self):
        positions = self._env.get_picker_position()
        #swap y and z
        positions = positions[:, [0, 2, 1]]
        return positions

    def _get_cloth_area(self):
        w, h = self._env.get_cloth_dim()
        return w * h
    
    def set_disp(self, flag):
        self._env.set_gui(flag)

    def get_action_space(self):
        return self._env.get_action_space()
    
    def observation_shape(self): ### If image: H*W*C
        raise NotImplementedError
     
    
    def get_eval_configs(self):
        return self.eval_params
    
    def get_val_configs(self):
        return self.val_params
    
    def get_flattened_pos(self):
        return self._env.get_flattened_pos()

    
    def set_eval(self):
        self._train = False
        self._env.eval_flag = True
    
    def set_val(self):
        self.set_eval()

    def set_train(self):
        self._train = True
        self._env.eval_flag = False
    
    def get_num_episodes(self):
        return self._env.get_num_episodes()

    def get_mode(self):
        if self._train:
            return 'train'
        else:
            return 'eval'

    def clear_frames(self):
        self._env.reset_control_step_info()

    def get_frames(self):
        res = self._env.get_control_step_info()
        return res['rgb']
    
    def get_info(self):
        return self.info


    def is_falttened(self):
        return self._get_normalised_coverage() > 0.99
    
    
    def get_coverage(self):
        return self._env.get_coverage()
    
    
    def get_initial_coverage(self):
        return self._env.get_initial_coverage()
    
    def get_flattened_coverage(self):
        return self._env.get_flattened_coverage()

    def _get_cloth_mask(self, camera_name='default_camera', resolution=(64, 64)):
        return self._env.get_cloth_mask(camera_name=camera_name, resolution=resolution)

    def get_canonical_mask(self, resolution=(64, 64)):
        return self._env.get_canonical_mask(resolution=resolution)

    def get_cloth_size(self):
        return self._env.get_cloth_size()

    def set_save_control_step_info(self, flag):
        self._env.set_save_control_step_info(flag)

    def get_object_positions(self):

        positions = self._env.get_particle_positions()
        positions = positions[:, [0, 2, 1]]
        return positions

    def get_state(self):
        state = {
            'particle_pos': self.get_object_positions(),
            'picker_pos': self._env.get_picker_pos(),
            'control_step': self._env.control_step,
            'action_step': self._t
        }

        return state  

    def set_state(self, state, step=None):
        self._env.set_pos(state['particle_pos'], state['picker_pos'])
        self._t = state['action_step']
        self._env.control_step = state['control_step']

    def get_performance_value(self):
        return self._env.get_performance_value()
    
    def wait_until_stable(self, max_wait_step=200):
        info = self._env.wait_until_stable(max_wait_step=max_wait_step)
        return self._process_info(info)
    
    def _get_visibility(self, positions, cameras="default_camera", resolution=(128,128)):
        # TODO: need to refactor this, so bad.
        N = positions.shape[0]
        
        visibility = [False for _ in range(N)]

        camera_hight = self.camera_height
        # print('camera_hight', camera_hight)
        # print('pixel_to_world_ratio', self.pixel_to_world_ratio)
        depths = camera_hight - (positions[:, 1]+ self._env.cloth_particle_radius) #x, z, y
        #print('depths', depths)
        
        #print('positions', positions)

        self.pixel_to_world_ratio = self._env.pixel_to_world_ratio
        projected_pixel_positions_x = positions[:, 0]/(self.pixel_to_world_ratio*depths) #-1, 1
        projected_pixel_positions_y = positions[:, 2]/(self.pixel_to_world_ratio*depths) #-1, 1
        projected_pixel_positions = np.concatenate(
            [projected_pixel_positions_x.reshape(N, 1), projected_pixel_positions_y.reshape(N, 1)],
            axis=1)

        depth_images = self.render(mode='d', camera_name=cameras, resolution=resolution)

        for i in range(N):
            x, y = projected_pixel_positions[i][0],  projected_pixel_positions[i][1]
            
            ## if not a number, continue
            if np.isnan(x) or np.isnan(y):
                continue
            if x < -1 or x > 1 or y < -1 or y > 1:
                continue
            x_ = int((y + 1)/2 * resolution[0])
            y_ = int((x + 1)/2 * resolution[1])
            ## clip x_, y_ between 0 and resolution
            x_ = max(0, min(x_, resolution[0]-1))
            y_ = max(0, min(y_, resolution[1]-1))
            # print('depth i', depths[i])
            # print('depth image i',  depth_images[0][x_][y_])
            if depths[i] < depth_images[x_][y_] + 1e-4:
                visibility[i] = True

        #print('np.asarray(visibility', np.asarray(visibility))
        
        return [np.asarray(visibility)], [projected_pixel_positions]

    def get_flatten_observation(self):
        
        if self.flatten_obs is not None:
            return self.flatten_obs
        
        goal_ = self._env._goal
        flatten_obs = {}
        H, W = self.observation_shape()['rgb'][0], self.observation_shape()['rgb'][1]
        flatten_obs['rgb'] = cv2.resize(goal_['rgb'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        flatten_obs['depth'] = cv2.resize(goal_['depth'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        #print('goal_["mask"]', goal_['mask'].shape)
        flatten_obs['mask'] = cv2.resize(goal_['mask'].astype(float), (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        flatten_obs['mask'] = flatten_obs['mask'] > 0.5
        flatten_obs['particle_position'] = self._get_flatten_positions()
        self.flatten_obs = flatten_obs
        return self.flatten_obs 
        
    def reset(self, episode_config=None):
        #print('reset')
        self.evaluate_result = None
        if episode_config == None:
            episode_config = {
                'eid': None,
                'save_video': False
            }
        self.flatten_obs = None
        self._t = 0  # Reset internal timer
        self.last_info = None
        if 'save_video' not in episode_config:
            episode_config['save_video'] = False
        
        self.set_save_control_step_info(episode_config['save_video'])
        info = self._env.reset(episode_id=episode_config['eid'])
        self.episode_id = self._env.episode_id
        episode_config['eid'] = self.episode_id
        self.episode_config = episode_config.copy()
        self.info = self._process_info(info)
        self.info = self.action_tool.reset(self)
        self.info.update(self.task.reset(self))
        
        return self.info
    
    def get_episode_config(self):
        return self.episode_config
    
    def get_episode_id(self):
        return self.episode_id

    def get_pointcloud(self):
        particle_pos = self.get_object_positions()
        visibility, proj_pos = self._get_visibility(particle_pos)
        visible_particle_pos = particle_pos[tuple(visibility)]
        return visible_particle_pos

    def get_keypoint_positions(self):
        return self._env.get_keypoint_positions()

    def control_picker(self, action, process_info=True):
        
        self._t += 1
        action = action[:, [0, 2, 1, 3]]
        info = self._env.step(action)
        if process_info:
            #print('process info!!!!')
            info = self._process_info(info)
        info['arena'] = self
        self.info = info
        return info
    
    def step(self, action): ## get action for hybrid action primitive, action defined in the observation space
        self.last_info = self.info
        self.evaluate_result = None
        info = self.action_tool.step(self, action)
        self.info = self._process_info(info)
        return self.info

    def _process_info(self, info):
        #print('here process')
        assert 'observation' in info.keys()
        assert 'rgb' in info['observation'].keys()
        H, W = self.observation_shape()['rgb'][0], self.observation_shape()['rgb'][1]
        info['observation']['rgb'] = cv2.resize(info['observation']['rgb'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        info['observation']['depth'] = cv2.resize(info['observation']['depth'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        info['observation']['mask'] = self._get_cloth_mask(resolution=(H, W))
        info['observation']['particle_position'] = self._get_particle_positions()
        info['goal'] = self.task.get_goal(self)
        if 'contour' in self.info_keys:
            info['observation']['contour'] = self.get_contour(resolution=(H, W))
        info['no_op'] = self.get_no_op()
        if 'cloth_size' in self.info_keys:
            info['cloth_size'] = self.get_cloth_size()
        info['normalised_coverage'] = self._get_normalised_coverage()
        if 'corner_positions' in self.info_keys:
            info['corner_positions'] = self.get_corner_positions()
        #info['flatten_canonical_IoU'] = self.get_flatten_canonical_IoU()
        if 'corner_visibility' in self.info_keys:
            info['corner_visibility'], _ = self._get_visibility(info['corner_positions'])
            info['corner_visibility'] = info['corner_visibility'][0]
        #info['pointcloud'] = self.get_pointcloud()
        info['arena'] = self
        info['arena_id'] = self.id
        info['evaluation'] = self.task.evaluate(self)
        info['reward'] = self.task.reward(self.last_info, None, info)
        return info
    


    def render(self, mode='rgb', resolution=(720, 720), camera_name='default_camera'):
        return self._env.render(
            mode = mode,
            camera_name=camera_name, 
            resolution=resolution)

    def close(self):
        self._env.close()

    def get_action_horizon(self):
        return self.action_tool.action_horizon

    def get_flatten_corner_positions(self):
        return self._env.get_flatten_corner_positions()
    
    def get_flatten_edge_positions(self):
        return self._env.get_flatten_edge_positions()

    def _get_flatten_positions(self):
        # swap y and z
        positions = self.get_object_positions()
        positions = positions[:, [0, 2, 1]]
        return positions
    

    def _get_normalised_coverage(self):
        return max(0, min(1, self._env.get_normalised_coverage()))
    
    def _get_normalised_impovement(self):
        target_coverage = self.get_flattened_coverage()
        initial_coverage = self.get_initial_coverage()
        current_coverage = self.get_coverage()

        res = (current_coverage - initial_coverage) / (max(target_coverage - initial_coverage, 0) + 1e-3)
        return np.clip(res, 0, 1)
    
    def get_wrinkle_ratio(self):
        return self._env.get_wrinkle_ratio()


    
    
    @property
    def observation_space(self):
        if self.symbolic:
            return self._env.observation_space
        else:
            return Box(low=-np.inf, high=np.inf, 
                shape=(self.observation_shape()['image'][0], self.observation_shape()['image'][1], self.observation_space()[2]), dtype=np.float64)

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] \
            if self._symbolic else (self._image_c, self._image_dim, self._image_dim)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action_ts(self):
        if self._env.action_tool._step_mode == "pixel_pick_and_place":
            return torch.from_numpy(self._env.action_tool.sample())
        return torch.from_numpy(self._env.action_space.sample())
    
    def sample_random_action(self):
        # if self._env.action_tool._step_mode == "pixel_pick_and_place":
        #     return self._env.action_tool.sample()
        return self._env.action_space.sample()

    # def get_flatten_observation(self):
    #     obs = self._env.get_flatten_observation()
    #     H, W = self.observation_shape()['rgb'][0], self.observation_shape()['rgb'][1]
    #     obs['rgb'] = cv2.resize(obs['rgb'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
    #     obs['depth'] = cv2.resize(obs['depth'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
    #     obs['mask'] = self._get_cloth_mask(resolution=(H, W))
    #     return obs
    
    def get_no_op(self):
        raise NotImplementedError
    
    def evaluate(self):
        if self.evaluate_result is None:
            self.evaluate_result = self.task.evaluate(self)
        return self.evaluate_result