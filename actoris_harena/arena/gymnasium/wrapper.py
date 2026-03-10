import os
import numpy as np
import cv2
import gymnasium as gym

from actoris_harena.arena.arena import Arena
from ..loggers.standard_logger import StandardLogger
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)


class GymnasiumArena(Arena):

    def __init__(self, config):
        os.environ["MUJOCO_GL"] = "egl"
        super().__init__(config)

        self._domain = config.domain
        self._max_env_step = 20000
        self._action_repeat = 1
        self.logger = StandardLogger()
        
        if config.domain == 'pushT':
            from .envs.pushT import PushTImageEnv
            self._env = PushTImageEnv(**config)
            self._max_env_step = 1000
        else:
            self._env = gym.make(config.domain, render_mode=config.render_mode)

        print('[Gymansium] action_space', self._env.action_space)

        self.set_disp(bool(config.disp))

        self.eval_params = [{'eid': i, 'save_video': True} for i in range(10)]
        self.eval_params.extend([{'eid': i, 'save_video': False} for i in range(10, 30)])
        self.val_params = [{'eid': i, 'save_video': False} for i in range(3)]

        ### first 100 seeds for evaluation, next 100 for validation, and the rest for training
        self._num_seeds = 1000
        self._eval_seeds = np.arange(30)
        self._val_seeds = np.arange(30, 40)
        self._train_seeds = np.arange(100, self._num_seeds)

    def reset(self, episode_config=None):
        self._sim_step = 0  # Reset internal timer
        self._total_reward = 0
        
        # get seed and set save_frame flag
        self._save_frame = False
        if episode_config is not None:
            seed = episode_config['eid']
            self._save_frame = episode_config.get('save_video', False)
            if self.mode == 'val':
                seed = self._val_seeds[seed]
            elif self.mode == 'eval':
                seed = self._eval_seeds[seed]
            elif self.mode == 'train':
                seed = self._train_seeds[seed]
            else:
                raise ValueError('mode must be either train, eval, or val')
        else:
            if self.mode == 'val':
                seed = self._val_seeds[np.random.randint(0, len(self._val_seeds))]
            elif self.mode == 'eval':
                seed = self._eval_seeds[np.random.randint(0, len(self._eval_seeds))]
            elif self.mode == 'train':
                seed = self._train_seeds[np.random.randint(0, len(self._train_seeds))]
        
        if self._save_frame:
            self._frames = []
        
        obs, info_ = self._env.reset(seed=int(seed))
        
        # ADDED: Store the latest obs and info for the success() method
        self._last_obs = obs
        self._last_info = info_

        if self.disp:
            self._display()
        
        self._sim_step = 0
        info = {
            'action_space': self._env.action_space, 
            'observation': obs,
            'evaluation': self.evaluate(),
            'done': False,
            'arena_id': self.id,
            'sim_steps': 0,
            'success': False
        }

        # Safely handle different observation structures
        if self._domain == 'pushT':
            info['observation']['rgb'] = obs['image']
            info['observation']['vector_state'] = obs['agent_pos']
        else:
            info['observation']['rgb'] = self._env.render()
            
        return info

    def success(self):
        """Determines if the current state is a success."""
        # 1. Gymnasium Robotics standard: check 'is_success' in the info dict
        if hasattr(self, '_last_info') and 'is_success' in self._last_info:
            return bool(self._last_info['is_success'])
        
        # 2. Fallback: Manually calculate distance for goal-based environments
        if hasattr(self, '_last_obs') and isinstance(self._last_obs, dict):
            if 'achieved_goal' in self._last_obs and 'desired_goal' in self._last_obs:
                distance = np.linalg.norm(self._last_obs['achieved_goal'] - self._last_obs['desired_goal'])
                return distance < 0.05  # Standard threshold for Fetch tasks
                
        return False
    
    def get_action_horizon(self):
        return self._max_env_step
    
    def step(self, action):
        if isinstance(action, dict):
            action = action['default']
            
        reward = 0
        info = {}

        for _ in range(self._action_repeat):
            obs, r, term, trunc, info_ = self._env.step(action)
            
            if self.disp:
                self._display()

            if self._save_frame:
                self._frames.append(self._env.render())

            reward += r
            self._sim_step += 1
            
            done = (term or trunc or self._sim_step >= self._max_env_step)

            if done:
                break
                
        # ADDED: Store the latest obs and info so success() can read them
        self._last_obs = obs
        self._last_info = info_

        info = {
            'done': done,
            'reward': reward,
            'evaluation': self.evaluate(),
            'action_space': self._env.action_space, 
            'observation': obs,
            'arena_id': self.id,
            'sim_steps': self._action_repeat,
            'success': self.success()  # Now this will work correctly
        }

        # Safely handle different observation structures
        if self._domain == 'pushT':
            info['observation']['rgb'] = obs['image']
            info['observation']['vector_state'] = obs['agent_pos']
        else:
            info['observation']['rgb'] = self._env.render()

        self._total_reward += reward
        return info

    def get_name(self, episode_config=None):
        return 'Open AI Gym:' + self._domain
    
    def get_goal(self):
        return {}

    def evaluate(self):
        res = {
            'total_reward': self._total_reward,
        }
        return res

    def get_frames(self):
        return self._frames
    
    def clear_frames(self):
        self._frames = []

    def get_eval_configs(self):
        return self.eval_params
    
    def get_val_configs(self):
        return self.val_params

    def get_max_interactive_steps(self):
        return self._max_env_step

    def get_no_op(self):
        return self.no_op
        
    def set_disp(self, flag):
        super().set_disp(flag)
        if self.disp:
            cv2.startWindowThread()
            cv2.namedWindow("simulation", cv2.WINDOW_NORMAL)

    def sample_random_action(self):
        return self.get_action_space().sample()
    
    def get_action_space(self):
        return self._env.action_space
    
    def _display(self):
        # FIXED: Render mode is already defined in gym.make
        pixels = self._env.render()
        tmp_pixels = pixels.copy()
        # Convert RGB to BGR for OpenCV
        pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2] = \
              tmp_pixels[:, :, 2], tmp_pixels[:, :, 1], tmp_pixels[:, :, 0]
        cv2.imshow('simulation', pixels)
        cv2.waitKey(1)

    def set_task(self, task):
        self._env.set_task(task)
    
    def compare(self, result_1, result_2):
        """
        result_1 and result_2 are the validation results from two different 'policies'. 
        They are in the form of a list of information dictionaries for each episode.
        If result_1 is better than result_2 return 1, worse return -1, if similar return 0.
        """
        
        def get_scalar(val):
            """Helper to extract the final scalar if the value is a list (trajectory)."""
            if isinstance(val, list):
                # Return the last element (final state) if list is not empty
                return val[-1] if len(val) > 0 else 0
            return val

        def get_stats(results):
            processed_successes = []
            processed_rewards = []
            successful_steps = []

            for r in results:
                # --- Handle 'success' ---
                raw_success = r.get('success', 0)
                # Use helper to handle cases where raw_success is [0, 0, 1, 1...]
                final_success = float(get_scalar(raw_success))
                processed_successes.append(final_success)

                # --- Handle 'reward' ---
                raw_reward = r.get('total_reward', r.get('reward', 0.0))
                final_reward = float(get_scalar(raw_reward))
                processed_rewards.append(final_reward)

                # --- Handle 'steps' (only for successful episodes) ---
                if final_success > 0.5: # Treat as True
                    # Check for explicit length/steps keys
                    if 'length' in r:
                        steps = r['length']
                    elif 'steps' in r:
                        steps = r['steps']
                    # Fallback: if 'success' was a list, the list length = num steps
                    elif isinstance(raw_success, list):
                        steps = len(raw_success)
                    else:
                        steps = 0 
                    
                    successful_steps.append(float(get_scalar(steps)))
            
            mean_success = np.mean(processed_successes) if processed_successes else 0.0
            std_reward = np.std(processed_rewards) if processed_rewards else 0.0
            mean_steps = np.mean(successful_steps) if successful_steps else float('inf')
            
            return mean_success, std_reward, mean_steps

        # Get stats for both
        mean_s1, std_r1, steps_s1 = get_stats(result_1)
        mean_s2, std_r2, steps_s2 = get_stats(result_2)
        
        # Thresholds
        SUCCESS_EPS = 1e-4
        STD_EPS = 1e-4
        STEP_EPS = 0.5

        # 1. Compare Success Rate (Higher is better)
        if mean_s1 > mean_s2 + SUCCESS_EPS:
            return 1
        elif mean_s2 > mean_s1 + SUCCESS_EPS:
            return -1
        
        # 2. Compare Reward Standard Deviation (Lower is better)
        if std_r1 < std_r2 - STD_EPS:
            return 1
        elif std_r2 < std_r1 - STD_EPS:
            return -1

        # 3. Compare Average Steps to Success (Lower is better)
        if steps_s1 < steps_s2 - STEP_EPS:
            return 1
        elif steps_s2 < steps_s1 - STEP_EPS:
            return -1

        return 0