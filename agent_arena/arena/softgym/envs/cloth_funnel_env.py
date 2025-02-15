import os
import h5py
import numpy as np
import cv2
import json

from softgym.action_space.action_space import Picker
from softgym.utils.env_utils import get_coverage
import pyflex
from agent_arena import Arena
from tqdm import tqdm


from ..picker_action_wrappers.hybrid_action_primitive import HybridActionPrimitive
from .cloth_funnel_logger import ClothFunnelLogger
from .clothfunnel_utils import set_scene, \
    get_max_IoU, calculate_iou
from .camera_utils import get_camera_matrix

import ray

global CLOTH_FUNNEL_ENV_NUM
CLOTH_FUNNEL_ENV_NUM = 0

# @ray.remote
class ClothFunnelEnv(Arena):
    name = 'cloth_funnel_env'
    
    def __init__(self, config):
        ## Agent Arena setups, TODO: eval_params and val_params.
        self.config = config
        self.info = {}
        self.sim_step = 0
        self.save_video = False
        self.logger = ClothFunnelLogger()
        self.random_reset = False
        self.set_id(0)
        self.name =f'clothfunnels-{config.object}'
        
        # Softgym Setup
        self._get_sim_config()
        self._setup_camera()
        
        self.scene_config = self.default_config['scene_config']
        self.workspace_mask = None
        
        #print('disp', config.disp)
        headless = not config.disp
        #print('headless', headless)
        pyflex.init(headless, 
                    True, 
                    self.camera_config['cam_size'][0], 
                    self.camera_config['cam_size'][1],
                    )
        self.pickers = Picker(
            2, picker_radius=self.config.picker_radius, 
            particle_radius=self.scene_config['radius'],
            picker_threshold=self.config.picker_threshold,
            picker_low=self.config.picker_low, 
            picker_high=self.config.picker_high,
            grasp_mode=(self.config.grasp_mode if 'grasp_mode' in self.config.keys() else {'closest': 1.0}),
        )
        self.particle_radius = self.scene_config['radius']
        self.action_tool = HybridActionPrimitive(
            action_horizon=self.config.horizon,
            drag_vel=0.01)
        self._observation_image_shape = config.observation_image_shape \
            if 'observation_image_shape' in config else (480, 480, 3)

        
        self._get_init_state_keys()

        
    

    def _setup_camera(self):
        
        self.default_camera = self.default_config['camera_name']
        self.camera_config = self.default_config['camera_params'][self.default_camera]
        camera_pos = self.camera_config['cam_position'].copy()
        # swap y and z
        camera_pos[1], camera_pos[2] = camera_pos[2], camera_pos[1]
        #print('camera_pos', camera_pos)
        self.camera_height = camera_pos[2]
        camera_angle = self.camera_config['cam_angle'].copy()
        camera_angle[1], camera_angle[2] = camera_angle[2], camera_angle[1]
        camera_angle[0] = np.pi + camera_angle[0]
        camera_angle[2] = 4*np.pi/2 - camera_angle[2]
        #print('camera_angle', camera_angle)
        self.picker_initial_pos = self.default_config['picker_initial_pos']
        self.camera_intrinsic_matrix, self.camera_extrinsic_matrix = \
            get_camera_matrix(
                camera_pos, 
                camera_angle, 
                self.camera_config['cam_size'], 
                self.camera_config['cam_fov'])
        
        self.camera_size = self.camera_config['cam_size']

    def _get_sim_config(self):
        from .clothfunnel_utils import get_default_config
        self.default_config = get_default_config()

    def set_task(self, task):
        self.task = task

    def set_logdir(self, log_dir):
        self.logger.set_log_dir(log_dir)

    ## TODO: put this into the interface
    def get_action_horizon(self):
        return self.action_tool.get_action_horizon()
       
    ## TODO: if eid is out of range, we need to raise an error.   
    def reset(self, episode_config=None):
        #print('reset')
        if episode_config == None:
            episode_config = {
                'eid': None,
                'save_video': False
            }
        if 'save_video' not in episode_config:
            episode_config['save_video'] = False
        
        if 'eid' not in episode_config or episode_config['eid'] is None:

            # randomly select an episode whose 
            # eid equals to the number of episodes%CLOTH_FUNNEL_ENV_NUM = self.id
            if self.mode == 'train':
                episode_config['eid'] = np.random.randint(self.num_train_tasks)
            else:
                episode_config['eid'] = np.random.randint(self.num_eval_tasks)
           
        init_state_params = self._get_init_state_params(episode_config['eid'])



        self.sim_step = 0
        self.video_frames = []
        self.save_video = episode_config['save_video']

        self.episode_config = episode_config

        init_state_params['scene_config'] = self.scene_config
        init_state_params.update(self.default_config)
        set_scene(
            config=init_state_params, 
            state=init_state_params)
        #print('set scene done')
        self.pickers.reset(self.picker_initial_pos)
        #print('picker reset done')

        self.init_coverae = self._get_coverage()
        self.goal_obs = None
        self.get_goal()
        #self.flatten_coverage = init_state_params['flatten_area']
        
        self.info = {}
        self.action_tool.reset(self) # get out of camera view, and open the gripper
        self._step_sim()

        self.evaluate_result = None
        self.last_info = None
        
        self.info = self._process_info({})

        
        return self.info
    
    def get_episode_config(self):
        return self.episode_config

    def _get_max_IoU(self):

        cur_mask = self._get_cloth_mask()
        goal_mask = self.get_goal()['rgb'].sum(axis=2) > 0

        # print('cur_mask', cur_mask.shape)
        # print('goal_mask', goal_mask.shape)

        IoU, matched_IoU = get_max_IoU(cur_mask, goal_mask)
        
        return IoU

    def _get_canon_IoU(self):
        cur_mask = self._get_cloth_mask()
        goal_mask = self.get_goal()['rgb'].sum(axis=2) > 0
        IoU = calculate_iou(cur_mask, goal_mask)
        return IoU

    
    def get_id(self):
        return self.id

    def get_num_episodes(self) -> np.int:
        if self.mode == 'eval':
            return 30
        elif self.mode == 'train':
            return self.num_train_tasks
        else:
            raise NotImplementedError

    def get_info(self):
        return self.info
    
    def _process_info(self, info):
        info.update({
            'evaluation': self.evaluate(),
            'success': self.success(),
            'observation': self._get_obs(),
            'goal': self.get_goal(),
            'arena': self,
            'arena_id': self.id,
            'action_space': self.get_action_space(),
        })

        ## plot observation rgb
        # rgb = info['observation']['rgb']   
        # import matplotlib.pyplot as plt
        # plt.imsave('obs.png', rgb)

        for k, v in info['goal'].items():
            info['observation'][f'goal_{k}'] = v
            #print(f'goal_{k}', v.shape)

        if 'done' not in info:
            info['done'] = False
        info['reward'] = self.task.reward(self.last_info, None, info)
        return info
    
    def step(self, action): ## get action for hybrid action primitive, action defined in the observation space
        self.last_info = self.info
        self.evaluate_result = None
        info = self.action_tool.step(self, action)
        self.info = self._process_info(info)
        return self.info
    

    def clear_frames(self):
        self.video_frames = []
    
    def get_action_space(self):
        return self.action_tool.get_action_space()
    
    def get_frames(self):
        return self.video_frames.copy()
    
    def get_goal(self):
        
        if self.goal_obs == None:
            current_particl_pos = pyflex.get_positions()
            pyflex.set_positions(self.episode_params['init_particle_pos'].flatten())
            self.wait_until_stable()
            self.goal_obs = self._get_obs()
            self.flatten_coverage = self._get_coverage()
            pyflex.set_positions(current_particl_pos)
            self.wait_until_stable()
        
        return self.goal_obs
    
    def get_eval_configs(self):
        eval_configs = [
            {'eid': eid, 'tier': 0, 'save_video': True}
            for eid in range(10)
        ]
        eval_configs += [
            {'eid': eid, 'tier': 1, 'save_video': False}
            for eid in range(10, 30)]
        return eval_configs

    
    def get_val_configs(self):
        return {} #TODO
    
    def get_no_op(self):
        return self.action_tool.get_no_op()
    
    def set_disp(self, disp):
        # This function is disabled for this environment
        pass

    def evaluate(self):
        if self.evaluate_result is None:
            self.evaluate_result = self.task.evaluate(self)
        return self.evaluate_result
        

    def success(self):
        return self.task.success(self)
    
    def observation_shape(self):
        return {'rgb': self._observation_image_shape, 
                'depth': self._observation_image_shape}
#
    def sample_random_action(self):
        return self.action_tool.sample_random_action()

    # TODO: we may need to modify this.
    def set_id(self, id):
        self.id = id
    
    # these funcitons is required by the action_tool
    def get_picker_position(self):
        p = self._get_picker_position()
        # swap y and z
        p[:, [1, 2]] = p[:, [2, 1]]
        return p

    def get_particle_positions(self):
        p = self._get_particle_positions()
        p[:, [1, 2]] = p[:, [2, 1]]
        return p
    
    def control_picker(self, signal, process_info=True):
        
        signal = signal[:, [0, 2, 1, 3]]
        self.pickers.step(signal)
        self._step_sim()
        info = {
            'observation': self._get_obs(),
        }

        if process_info:
            info = self._process_info_(info)
        self.info = info
        return info
    
    def wait_until_stable(self, max_wait_step=200, stable_vel_threshold=0.0006):
        wait_steps = self._wait_to_stabalise(max_wait_step=max_wait_step, stable_vel_threshold=stable_vel_threshold)
        # print('wait steps', wait_steps)
        obs = self._get_obs()
        return {
            'observation': obs,
            'done': False,
            'wait_steps': wait_steps
        }
    
    ## Helper Functions
    def _wait_to_stabalise(self, max_wait_step=300, stable_vel_threshold=0.0006,
            target_point=None, target_pos=None):
        t = 0
        stable_step = 0
        #print('stable vel threshold', stable_vel_threshold)
        last_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
        for j in range(0, max_wait_step):
            t += 1

           
            cur_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
            curr_vel = np.linalg.norm(cur_pos - last_pos, axis=1)
            if target_point != None:
                cur_poss = pyflex.get_positions()
                curr_vell = pyflex.get_velocities()
                cur_poss[target_point * 4: target_point * 4 + 3] = target_pos
                
                curr_vell[target_point * 3: target_point * 3 + 3] = [0, 0, 0]
                pyflex.set_positions(cur_poss.flatten())
                pyflex.set_velocities(curr_vell)
                curr_vel = curr_vell

            self._step_sim()
            # if self.save_control_step_info:
            #     if 'control_signal' not in self.control_step_info:
            #         self.control_step_info['control_signal'] = []
            #     self.control_step_info['control_signal'].append(np.zeros((self.num_picker, 4)))
                
            if stable_step > 10:
                break
            if np.max(curr_vel) < stable_vel_threshold:
                stable_step += 1
            else:
                stable_step = 0

            last_pos = cur_pos
            
        #print('wait steps', t)
        return t
    
    def _get_normalised_coverage(self):
        res = self._get_coverage() / self.flatten_coverage
        #print('flatten coverage', self.flatten_coverage)
        #print('coverage', self._get_coverage())
        # clip between 0 and 1
        return np.clip(res, 0, 1)


    def _get_normalised_impovement(self):
        res = (self._get_coverage() - self.init_coverae) / (max(self.flatten_coverage - self.init_coverae, 0) + 1e-3)
        return np.clip(res, 0, 1)
    
    
    
    def _set_to_flatten(self, config, cloth_particle_radius=0.00625):
        cloth_dimx, cloth_dimz = config['cloth_size']
        N = cloth_dimx * cloth_dimz
        px = np.linspace(
            0, cloth_dimx * cloth_particle_radius, cloth_dimx)
        py = np.linspace(
            0, cloth_dimz * cloth_particle_radius, cloth_dimz)
        xx, yy = np.meshgrid(px, py)
        new_pos = np.empty(shape=(N, 4), dtype=np.float)
        new_pos[:, 0] = xx.flatten()
        new_pos[:, 1] = cloth_particle_radius
        new_pos[:, 2] = yy.flatten()
        new_pos[:, 3] = 1.
        new_pos[:, :3] -= np.mean(new_pos[:, :3], axis=0)
        pyflex.set_positions(new_pos.flatten())
        return self._get_coverage()

    def _get_coverage(self):
        particle_positions = self._get_particle_positions()
        # swap y and z
        particle_positions[:, [1, 2]] = particle_positions[:, [2, 1]]
        return get_coverage(particle_positions, self.particle_radius)

    def _get_particle_positions(self):
        pos = pyflex.get_positions().reshape(-1, 4)[:, :3].copy()
        # swap y and z
        pos[:, [1, 2]] = pos[:, [2, 1]]
        #print('pos', pos[0])
        return pos
    
    def _get_picker_pos(self):
        return self.pickers.get_picker_pos()
    
    def _get_picker_position(self):
        pos = self._get_picker_pos()
        return pos[:, :3].copy()
    
    def _reset_end_effectors(self):
        self.action_tool.movep(
            self,
            [[0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]], 
            vel=0.3)
        
    def _step_sim(self):
        pyflex.step()
        if self.save_video:
            self.video_frames.append(self._render('rgb', background=True))
        self.sim_step += 1


    def _process_info_(self, info):
        #print('here process')
        assert 'observation' in info.keys()
        assert 'rgb' in info['observation'].keys()
        H, W = self.observation_shape()['rgb'][0], self.observation_shape()['rgb'][1]
        info['observation']['rgb'] = cv2.resize(info['observation']['rgb'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        info['observation']['depth'] = cv2.resize(info['observation']['depth'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        info['observation']['mask'] = self._get_cloth_mask(resolution=(H, W))
        
        info['normalised_coverage'] = self._get_normalised_coverage()
        return info

    
    def _render(self, mode='rgb', camera_name='default_camera', resolution=None, background=False):
        #self._update_camera(camera_name)
        #pyflex.step()
        #print('pyflex render start')
        
        img, depth_img = pyflex.render()

        if not background:
            img, _ = pyflex.render_cloth()
        ## print statistics of depth image
        #print('depth stats', depth_img.min(), depth_img.max(), depth_img.mean(), depth_img.std())
        #print('pyflex render done')#
        CAMERA_WIDTH = self.camera_config['cam_size'][0]
        CAMERA_HEIGHT = self.camera_config['cam_size'][1]

        img = img.reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 4)[::-1, :, :3]  # Need to reverse the height dimension
        depth_img = depth_img.reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 1)[::-1, :, :1]
        

        if mode == 'rgbd':
            img =  np.concatenate((img, depth_img), axis=2)

        elif mode == 'rgb':
            pass
        elif mode == 'd':
            img = depth_img
        else:
            raise NotImplementedError
        
        if resolution is None:
            return img
        
        if CAMERA_HEIGHT != resolution[0] or CAMERA_WIDTH != resolution[1]:
            #print('resizing asked resolution', resolution)
            img = cv2.resize(img, resolution)

        return img
    
    def _get_obs(self):
        obs = {}
        obs['rgb'] = self._render(mode='rgb')
        obs['depth'] = self._render(mode='d')
        obs['mask'] = self._get_cloth_mask()
        obs['particle_position'] = self._get_particle_positions()
        return obs

    def _get_cloth_mask(self, camera_name='default_camera', resolution=None):
        rgb = self._render(camera_name=camera_name, mode='rgb', resolution=resolution)
        
        return rgb.sum(axis=2) > 0
    
    def _get_init_keys_helper(self, hdf5_path, key_file, difficulties=['hard', 'easy']):
        # print('hdf5_path', hdf5_path)
        # print('key_file', key_file)

        if os.path.exists(key_file):
            with open(key_file, 'r') as f:
                return json.load(f)
        else:
            with h5py.File(hdf5_path, 'r') as tasks:
                eval_keys = \
                    [key for key in tasks if tasks[key].attrs['task_difficulty'] in difficulties]
                print('total evalal keys', len(eval_keys))
            keys = []
            for key in tqdm(eval_keys, desc='Filtering keys'):
                with h5py.File(hdf5_path, 'r') as tasks:
                    group = tasks[key]
                    episode_params = dict(group.attrs)
                    for dataset_name in group.keys():
                        episode_params[dataset_name] = group[dataset_name][()]
                    
                    episode_params['scene_config'] = self.scene_config
                    episode_params.update(self.default_config)
                    set_scene(
                        config=episode_params, 
                        state=episode_params)
                    # self.pickers.reset(self.picker_initial_pos)
                    # self.action_tool.reset(self) # get out of camera view, and open the gripper
                    pyflex.step()
                    cloth_mask = self._get_cloth_mask()
                    if self.workspace_mask is not None:
                        cloth_mask = cloth_mask & self.workspace_mask
                    if cloth_mask.sum() > 500:
                        
                        keys.append(key)
                        # if 6 <= len(keys) <= 10:
                        #     print(f"eid {len(keys) - 1}, keys {keys[-1]}, sum of mask {cloth_mask.sum()}")
                        #     from matplotlib import pyplot as plt
                        #     plt.imsave(f'cloth_mask_{len(keys) - 1}.png', cloth_mask)
            # save the keys
            with open(key_file, 'w') as f:
                json.dump(keys, f)
        return keys

    def _get_init_state_keys(self):
        
        eval_path = os.path.join(self.config.init_state_path, f'multi-{self.config.object}-eval.hdf5')
        train_path = os.path.join(self.config.init_state_path, f'multi-{self.config.object}-train.hdf5')

        eval_key_file = os.path.join(self.config.init_state_path, f'{self.name}-eval.json')
        train_key_file = os.path.join(self.config.init_state_path, f'{self.name}-train.json')

        self.eval_keys = self._get_init_keys_helper(eval_path, eval_key_file, difficulties=['hard'])
        self.train_keys = self._get_init_keys_helper(train_path, train_key_file)
        
        # print len of keys
        self.num_eval_tasks = len(self.eval_keys)
        self.num_train_tasks = len(self.train_keys)
        print(f'Number of eval trials: {self.num_eval_tasks}')
        print(f'Number of train trials: {self.num_train_tasks}')

    def _get_init_state_params(self, eid):
        if self.mode == 'train':
            keys = self.train_keys
            hdf5_path = os.path.join(self.config.init_state_path, f'multi-{self.config.object}-train.hdf5')
        else:
            keys = self.eval_keys
            hdf5_path = os.path.join(self.config.init_state_path, f'multi-{self.config.object}-eval.hdf5')

        print('load initial state from', hdf5_path)
        print('eid', eid)
        key = keys[eid]
        print('key', key)
        with h5py.File(hdf5_path, 'r') as init_states:
            # print(hdf5_path, key)
            # Convert group to dict
            group = init_states[key]
            episode_params = dict(group.attrs)
            
            # If there are datasets in the group, add them to the dictionary
            #print('group keys', group.keys())
            for dataset_name in group.keys():
                episode_params[dataset_name] = group[dataset_name][()]

            self.episode_params = episode_params
            #print('episode_params', episode_params.keys())

        return episode_params
    
    def get_canon_particle_position(self):
        pos = self.episode_params['init_particle_pos'].reshape(-1, 4)[:,:3].copy()
        # swap y and z
        pos[:, [1, 2]] = pos[:, [2, 1]]
        return pos

    def get_cloth_area(self):
        return self.episode_params['cloth_height'] * self.episode_params['cloth_width']

@ray.remote(num_gpus=0.05)
class ClothFunnelEnvRay(ClothFunnelEnv):
    
    def __init__(self, config):
        super().__init__(config)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU ID
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Make CUDA calls synchronous