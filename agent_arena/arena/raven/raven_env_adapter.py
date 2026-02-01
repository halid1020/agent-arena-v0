import os
import numpy as np

from agent_arena.arena.arena import Arena

from .environments.environment import Environment
from . import tasks
from .utils.video_recorder import VideoRecorder
from agent_arena import StandardLogger
import pybullet as p

ENV_ASSETS_DIR = os.environ["RAVENS_ASSETS_DIR"]

class RavenEnvAdapter(Arena):

    def __init__(self, config):
        super().__init__(config)
        task = config.task
        disp = config.get('disp', False)
        
        # Camera Configuration
        img_res = config.get('img_res', None)
        view_mode = config.get('view_mode', 'standard')
        
        custom_cams = None
        hide_arm = False  # Default to showing the arm

        if view_mode == 'top_down':
            # 1. Enable arm hiding for top-down views
            hide_arm = True
            
            if img_res is not None:
                img_res = int(img_res)
                scale = img_res / 480.0
                focal_len = 450.0 * scale
                cx, cy = img_res / 2.0, img_res / 2.0
                rotation = p.getQuaternionFromEuler([0, np.pi, -np.pi/2])
                
                custom_cams = [{
                    'image_size': (img_res, img_res),
                    'position': np.array([0.5, 0, 1.0]), 
                    'rotation': rotation,
                    'zrange': (0.1, 2.0),
                    'noise': False,
                    'intrinsics': (focal_len, 0, cx, 0, focal_len, cy, 0, 0, 1)
                }]

        # 2. Pass hide_arm_rgb to Environment
        self._env = Environment(
            ENV_ASSETS_DIR,
            disp=disp,
            shared_memory=False,
            hz=240,
            agent_cams=custom_cams,
            hide_arm_rgb=hide_arm) 
        
        self._control_step_info = {'frame': []}
        self.disp = disp
        
        self._task = tasks.names[task]()
        self.action_horizon = config.get('action_horizon', self._task.max_steps)
        self._vid_rec = VideoRecorder(
            save_dir='.',
            episode_idx=None,
            record_mp4=True,
            display=self.disp,
            verbose=False)
        self._task.primitive._set_video_recorder(self._vid_rec)
        self.set_train()
        
        # Standard logging setup...
        self.num_eval_trials = 30
        self.num_val_trials = 10
        self.num_train_trials = 1000

        self.eval_params = [{'eid': i, 'save_video': True} for i in range(self.num_eval_trials)]
        self.val_params = [{'eid': i, 'save_video': True} for i in range(self.num_val_trials)]
        self.train_params = [{'eid': i, 'save_video': False} for i in range(self.num_train_trials)]

        self.logger = StandardLogger()

    def get_name(self):
        return "Raven"

    def get_action_horizon(self):
        return self.action_horizon

    def get_mode(self):
        return self.mode

    def get_action_space(self):
        return self._env.action_space

    def set_eval(self):
        self._task._set_mode('test')
        self.mode = 'eval'

    def set_train(self):
        self._task._set_mode('train')
        self.mode = 'train'
    
    def sample_random_action(self):
        ## sample random action from action space
        return self._env.action_space.sample()

    def get_no_op(self):
        # Access position_bounds through the internal _env object
        return {
            'pose0': (self._env.position_bounds.high, 
                      np.array([0., 0., 0., 1.], dtype=np.float32)),
            'pose1': (self._env.position_bounds.high,
                      np.array([0., 0., 0., 1.], dtype=np.float32))
        }

    def _get_binary_mask(self, raw_segm):
        """Helper to create a binary mask (scene objects vs background)."""
        # Gather IDs of all task-relevant objects (excluding table/plane/robot)
        valid_ids = []
        for cat in ['fixed', 'rigid', 'deformable']:
            valid_ids.extend(self._env.obj_ids[cat])
        
        # Create binary mask: 1 where ID matches a valid object, 0 otherwise
        return np.isin(raw_segm, valid_ids).astype(np.uint8)

    def reset(self, episode_config=None):
        self._step = 0
        self._total_reward = 0
        if 'save_video' not in episode_config.keys(): episode_config['save_video'] = False
        
        if episode_config == None:
            self.episode_id = np.random.randint(0, 1000)
            save_video = False
        else:
            self.episode_id = episode_config['eid']
            save_video = episode_config['save_video']

        if save_video:
            self.clear_frames()
            self._vid_rec.record_mp4 = True
        else:
            self._vid_rec.record_mp4 = False

        config_id = self.episode_id # eval: 0 - 100
        if self.mode == 'val': # 100 - 200
            config_id += 100
        elif self.mode == 'train': # 200 ->
            config_id += 200

        self._env.seed(config_id)
        self._env.set_task(self._task)
        obs = self._env.reset()

        # Process segmentation
        raw_segm = obs['mask'][0]
        binary_mask = self._get_binary_mask(raw_segm)

        info = {}
        info['observation'] = {
            'color': obs['color'],
            'depth': obs['depth'],
            'segm': raw_segm,       # Original segmentation (object IDs)
            'mask': binary_mask,    # Binary mask (0/1)
            'rgb': obs['color'][0],
        }
        info['done'] = False
        info['arena'] = self
        info['arena_id'] = self.id
        info['evaluation'] = self.evaluate()
        info['action_space'] = self.get_action_space()
        return info

    def step(self, action):
        
        if isinstance(action, dict):
            # Expects structure: {'pose0': (pos, rot), 'pose1': (pos, rot)}
            # pose0/1: ((x,y,z), (qx,qy,qz,qw))
            p0_pos, p0_rot = action['pose0']
            p1_pos, p1_rot = action['pose1']
            
            # Concatenate into flat [pos0, rot0, pos1, rot1] array
            action = np.concatenate([
                p0_pos, p0_rot, 
                p1_pos, p1_rot
            ])

        obs, reward, done, other_info = self._env.step(action)
        self._step  += 1
        self._total_reward += reward

        # Process segmentation
        raw_segm = obs['mask'][0]
        binary_mask = self._get_binary_mask(raw_segm)

        info = {}
        info['observation'] = {
            'color': obs['color'],
            'depth': obs['depth'],
            'segm': raw_segm,       # Original segmentation (object IDs)
            'mask': binary_mask,    # Binary mask (0/1)
            'rgb': obs['color'][0]
        }
        info['done'] = self._step >= self.action_horizon
        info['reward'] = reward
        info['others'] = other_info
        info['arena'] = self
        info['arena_id'] = self.id
        info['evaluation'] = self.evaluate()
        info['action_space'] = self.get_action_space()
        if reward >= 0.99: 
            info['success'] = True
        else: 
            info['success'] = False
        return info
    
    def get_episode_id(self):
        return self.episode_id
    
    def get_step(self):
        return self._step
    
    def get_goal(self):
        return {}

    def evaluate(self):
        res = {
            'total_reward': self._total_reward,
        }
        return res
    
    def get_eval_configs(self):
        return self.eval_params
    
    def get_val_configs(self):
        return self.val_params
    
    def get_train_configs(self):
        return self.train_params

    def get_frames(self):
        return np.stack(self._vid_rec.frames)
    
    def clear_frames(self):
        self._vid_rec.__enter__()

    def set_disp(self, flg):
        self._env = Environment(
            # ENV_ASSETS_DIR,
            disp=flg,
            # shared_memory=True,
            hz=480)
        self.disp = flg
    
    def success(self):
        return self._total_reward > 1-1e-6

    def get_control_step_info(self): 
            self._control_step_info['frame'] = np.stack(self._vid_rec.frames)
            return self._control_step_info
    
    def reset_control_step_info(self, flg=True):
        if flg:
            self._control_step_info['frame'] = []
            self._vid_rec.record_mp4 = True
            self._vid_rec.__enter__()
        else:
            self._vid_rec.record_mp4 = False
    
    def __getattr__(self, attr):
        if attr in ["obj_ids", "render_camera", "add_object"]: return getattr(self._env, attr)
        else: raise AttributeError(f"'EnvWrapper' object has no attribute '{attr}'")