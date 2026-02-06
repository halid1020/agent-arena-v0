import os
import numpy as np
import pybullet as p
import pickle
import matplotlib.pyplot as plt
import random

from agent_arena.arena.arena import Arena
from agent_arena import StandardLogger

from .environments.environment import Environment
from . import tasks
from .utils.video_recorder import VideoRecorder
from .tasks import cameras

ENV_ASSETS_DIR = os.environ["RAVENS_ASSETS_DIR"]

class RavenEnvAdapter(Arena):

    def __init__(self, config):
        super().__init__(config)
        task = config.task
        disp = config.get('disp', False)
        
        # --- NEW: Goal Configuration ---
        self.goal_dir = config.get('goal_dir', 'tmp/raven_goals')
        self.add_final_goal_to_obs = config.get('add_final_goal_to_obs', False)
        self.goals = [] # Store current episode's goal trajectory

        self.goal_cam_config = None
        
        self.use_default_goal_cam = config.get('use_default_goal_cam', False)
        if self.use_default_goal_cam:
            # The default RealSenseD415 config is a list; we take the first one.
            self.goal_cam_config = cameras.RealSenseD415.CONFIG[0]

        # -------------------------------

        # Camera Configuration
        img_res = config.get('img_res', None)
        view_mode = config.get('view_mode', 'standard')
        
        custom_cams = None
        hide_arm = False  

        if view_mode == 'top_down':
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

    # ---------------------------------------------------------------------------
    # Updated Reset and Step
    # ---------------------------------------------------------------------------

    def reset(self, episode_config=None):
        self._step = 0
        self._total_reward = 0
        if episode_config is None:
            # Random seed if not specified
            self.episode_id = np.random.randint(0, 1000)
            episode_config = {'eid': self.episode_id, 'save_video': False}
        else:
            if 'save_video' not in episode_config: episode_config['save_video'] = False
            self.episode_id = episode_config['eid']

        save_video = episode_config['save_video']
        if save_video:
            self.clear_frames()
            self._vid_rec.record_mp4 = True
        else:
            self._vid_rec.record_mp4 = False

        # Determine Seed based on Train/Val/Eval mode
        config_id = self.episode_id 
        if self.mode == 'val': config_id += 100
        elif self.mode == 'train': config_id += 200

        # --- STEP 1: Goal Generation/Loading ---
        # We do this BEFORE the actual agent reset, so we can restore the state after.
        self.goals = self._load_or_generate_goal(config_id)

        if save_video:
            self.clear_frames()
            self._vid_rec.record_mp4 = True
        else:
            self._vid_rec.record_mp4 = False

        # --- STEP 2: Actual Agent Reset ---
        # We must re-seed and reset to ensure the agent starts from the exact same state
        # as the goal demonstration started.
        np.random.seed(config_id)
        random.seed(config_id)
        self._env.seed(config_id)
        self._env.set_task(self._task)
        obs = self._env.reset()

        # Process Observation
        obs_dict = self._process_obs(obs)

        info = {}
        info['observation'] = obs_dict
        info['done'] = False
        info['arena'] = self
        info['arena_id'] = self.id
        info['evaluation'] = self.evaluate()
        info['action_space'] = self.get_action_space()

        # --- STEP 3: Inject Goal Info ---
        info = self._inject_goal_info(info)

        return info

    def step(self, action):
        
        # Handle Action Flattening (Dict -> Array)
        if isinstance(action, dict):
            p0_pos, p0_rot = action['pose0']
            p1_pos, p1_rot = action['pose1']
            action = np.concatenate([p0_pos, p0_rot, p1_pos, p1_rot])

        obs, reward, done, other_info = self._env.step(action)
        self._step += 1
        self._total_reward += reward

        # Process Observation
        obs_dict = self._process_obs(obs)

        info = {}
        info['observation'] = obs_dict
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

        # --- STEP 4: Inject Goal Info ---
        info = self._inject_goal_info(info)

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
        if len(self._vid_rec.frames):
            return []
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

    
    def _inject_goal_info(self, info):
        """Helper to inject goal data into the info dictionary."""
        # print('[debug] inject goal info function')
        if len(self.goals) > 0:
            # print('[debug] injecting')
            final_goal_step = self.goals[-1]
            
            # 1. info['goal'] contains the final state of the demonstration
            info['goal'] = {}
            for k, v in final_goal_step['observation'].items():
                info['goal'][k] = v
            
            # 2. info['goals'] contains the full trajectory
            info['goals'] = self.goals

            # 3. Flattened goal obs if requested
            if self.add_final_goal_to_obs:
                for k, v in final_goal_step['observation'].items():
                    #print('v.shape', v.shape)
                    info['observation'][f'goal_{k}'] = v
                    # Duplicate key with hyphen if needed for compatibility
                    info['observation'][f'goal-{k}'] = v 
        return info
    
    def _process_obs(self, obs):
        """Helper to process raw env observation into standard dict format."""
        raw_segm = obs['mask'][0]
        binary_mask = self._get_binary_mask(raw_segm)
        return {
            'color': obs['color'],
            'depth': obs['depth'][0],
            'segm': raw_segm,
            'mask': binary_mask,
            'rgb': obs['color'][0],
        }
    
    def _generate_goal(self, config_id):
        """Generates a goal trajectory using the Oracle policy."""
        goal_traj = []

        # Fix Determinism
        np.random.seed(config_id)
        random.seed(config_id)
        
        # 1. Reset Env for the Oracle
        self._env.seed(config_id)
        self._env.set_task(self._task)
        obs = self._env.reset()
        
        # --- NEW: Check if we need to render from a different camera for the goal ---
        init_step_obs = self._process_obs(obs)
        if self.goal_cam_config is not None:
             print('Goal Camera!!')
             # Manually render the specific goal view
             g_color, g_depth, g_segm = self._env.render_camera(self.goal_cam_config)
             # Update the processed observation with this view
             init_step_obs['color'] = (g_color,)
             init_step_obs['rgb'] = g_color
             init_step_obs['depth'] = g_depth
             init_step_obs['segm'] = g_segm
             init_step_obs['mask'] = self._get_binary_mask(g_segm)

        goal_traj.append({'observation': init_step_obs})

        # 2. Initialize Oracle
        oracle = self._task.oracle(self._env)
        
        done = False
        
        # 3. Run Episode
        while not done:
            action = oracle.act(obs, None) 
            
            if action is None: break
            
            if isinstance(action, dict):
                p0_pos, p0_rot = action['pose0']
                p1_pos, p1_rot = action['pose1']
                action = np.concatenate([p0_pos, p0_rot, p1_pos, p1_rot])
                
            obs, _, done, _ = self._env.step(action)
            
            # --- NEW: Process step observation with custom camera check ---
            step_obs = self._process_obs(obs)
            if self.goal_cam_config is not None:
                 g_color, g_depth, g_segm = self._env.render_camera(self.goal_cam_config)
                 step_obs['color'] = (g_color,)
                 step_obs['rgb'] = g_color
                 step_obs['depth'] = g_depth
                 step_obs['segm'] = g_segm
                 step_obs['mask'] = self._get_binary_mask(g_segm)

            goal_traj.append({'observation': step_obs})
            
        return goal_traj
    
    def _load_or_generate_goal(self, config_id):
        """Loads goal from disk or generates it if missing."""
        print('[RavenEnvAdapter, _load_or_generate_goal]', config_id)
        task_name = self._task.__class__.__name__
        episode_goal_dir = os.path.join(self.goal_dir, task_name, f"ep_{config_id}")
        
        goal_traj = []

        # Check if exists
        if os.path.exists(episode_goal_dir) and len(os.listdir(episode_goal_dir)) > 0:
            # Load existing
            # We assume steps are named strictly step_0.pkl, step_1.pkl, etc.
            steps = sorted([f for f in os.listdir(episode_goal_dir) if f.endswith('.pkl')])
            for step_file in steps:
                with open(os.path.join(episode_goal_dir, step_file), 'rb') as f:
                    goal_traj.append(pickle.load(f))
        else:
            # Generate
            print(f"Generating goal for {task_name} episode {config_id}...")
            goal_traj = self._generate_goal(config_id)
            
            # Save
            os.makedirs(episode_goal_dir, exist_ok=True)
            for i, step_data in enumerate(goal_traj):
                # Save full data as pickle
                with open(os.path.join(episode_goal_dir, f"step_{i:03d}.pkl"), 'wb') as f:
                    pickle.dump(step_data, f)
                
                # Save visualization (RGB) for debugging
                rgb = step_data['observation']['rgb']
                plt.imsave(os.path.join(episode_goal_dir, f"rgb_step_{i:03d}.png"), rgb)

        return goal_traj
    
    
    def __getattr__(self, attr):
        if attr in ["obj_ids", "render_camera", "add_object"]: return getattr(self._env, attr)
        else: raise AttributeError(f"'EnvWrapper' object has no attribute '{attr}'")
