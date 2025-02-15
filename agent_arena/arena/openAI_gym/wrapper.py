import numpy as np
import cv2
import gym

from agent_arena.arena.arena import Arena
from ...utilities.logger.standard_logger import StandardLogger

# https://github.com/Kaixhin/PlaNet/blob/master/env.py

class OpenAIGymArena(Arena):

    def __init__(self, domain, **kwargs):

        super().__init__()

        self._domain = domain
        self._max_env_step = 20000
        self._action_repeat = 1
        self.logger = StandardLogger()

        
        if domain == 'pushT':
            from .envs.pushT import PushTImageEnv
            self._env =  PushTImageEnv(**kwargs)
            self._max_env_step = 1000
        else:
            self._env = gym.make(domain)

        if kwargs['disp'] == 'True':
            self.set_disp(True)
        else:
            self.set_disp(False)


        self.eval_params = [{'eid': i, 'save_video': True} for i in range(10)]
        self.eval_params.extend([{'eid': i, 'save_video': False} for i in range(10, 30)])
        self.val_params = [{'eid': i, 'save_video': False} for i in range(3)]

        ### first 100 seeds for evaluation, next 100 for validation, and the rest for training
        self._num_seeds = 1000
        self._eval_seeds = np.arange(50)
        self._val_seeds = np.arange(50, 100)
        self._train_seeds = np.arange(100, self._num_seeds)

    
    def reset(self, episode_config=None):
        self._env_step = 0  # Reset internal timer
        self._total_reward = 0
        
        # get seed and set save_frame flag
        self._save_frame = False
        if episode_config != None:
            seed = episode_config['eid']
            self._save_frame = episode_config['save_video'] if 'save_video' in episode_config else False
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
        
        #print('save_frame', self._save_frame)
        
        # self._env.seed(seed)
        obs, info = self._env.reset(seed=int(seed))
        if self.disp:
            self._display()

        info = {'action_space': self._env.action_space, 
                'observation': obs,
                'done': False}
        if self._domain == 'pushT':
            info['observation']['rgb'] = obs['image']
            info['observation']['vector_state'] = obs['agent_pos']
        info['arena_id'] = self.id
        return info
    
    def set_eval(self):
        self.mode = 'eval'
    
    def success(self):
        return False
    
    def get_action_horizon(self):
        return self._max_env_step
    
    def step(self, action):
        action = action['default']
        reward = 0
        info = {}
        #print(self._env_step )

        for _ in range(self._action_repeat):
            obs, r, term, info_ = self._env.step(action)
            if self.disp:
                self._display()
            

            if self._save_frame:
                self._frames.append(self._env.render(mode='rgb_array'))

            reward += r
            
            self._env_step += 1  # Increment internal timer
            done = (term or self._env_step == self._max_env_step)

                
            if done:
                break

        #print('reward', reward)

        info = {
            'done': done,
            'reward': reward,
            'action_space': self._env.action_space, 
            'observation': obs}
        if self._domain == 'pushT':
            info['observation']['rgb'] =  obs['image']
            info['observation']['vector_state'] = obs['agent_pos']
        info['arena_id'] = self.id
        self._total_reward += reward

        #print('info', info)
        
        return info
    
    def success(self):
        return False


    def get_name(self, episode_config=None):
        return  'Open AI Gym:' + self._domain
    
    def get_goal(self):
        return {}

    def evaluate(self):

        res = {
            'mdp_return': self._total_reward,
        }

        return res

    def get_frames(self):
        #print('len frames', len(self._frames))
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
        action = self.get_action_space().sample()
        return action
    
    def get_action_space(self):
        return self._env.action_space
    

    def _display(self):
        pixels = self._env.render(mode='rgb_array')
        tmp_pixels = pixels.copy()
        pixels[:, :, 0], pixels[:, :, 1],  pixels[:, :, 2] =\
              tmp_pixels[:, :, 2], tmp_pixels[:, :, 1], tmp_pixels[:, :, 0]
        cv2.imshow('simulation', pixels)
        cv2.waitKey(1)

    def set_task(self, task):
        self._env.set_task(task)