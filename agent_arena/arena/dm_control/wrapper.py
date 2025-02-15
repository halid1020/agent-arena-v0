import numpy as np
import cv2
from gym.spaces import Box
from dm_control import suite
from dm_control.suite.wrappers import pixels

from agent_arena.arena.arena import Arena

# https://github.com/Kaixhin/PlaNet/blob/master/env.py

class DM_ControlSuiteArena(Arena):

    def __init__(self, domain, task, **kwargs):

        self._domain = domain
        self._task = task

        
        self._num_variation = kwargs['num_variations'] if 'num_variations' in kwargs else 1000
        self._train_portion = kwargs['train_portion'] if 'train_portion' in kwargs else 0.9
        self._pixel_observation = kwargs['pixel_observation']
        if self._pixel_observation:
            self._img_dim = kwargs['img_dim'] if 'img_dim' in kwargs else (128, 128)

        self._max_env_step = 1000
        if self._domain == 'walker':
            self._action_repeat = 2
        elif self._domain == 'cartpole' and self._task in ['balance', 'swingup']:
            self._action_repeat = 8
            
        elif self._domain == 'reacher' and self._task == 'easy':
            self._action_repeat = 4
        elif self._domain == 'finger' and self._task == 'spin':
            self._action_repeat = 2
        elif self._domain == 'cheetah' and self._task == 'run':
            self._action_repeat = 4
        elif self._domain == 'ball_in_cup' and self._task == 'catch':
            self._action_repeat = 6
        else:
            raise NotImplementedError
        
        print('action repeat', self._action_repeat)
        print('maximum interactive steps', self._max_env_step)

        variations = np.arange(0, self._num_variation)
        self._train_seeds = variations[:int(self._train_portion*self._num_variation)]
        self._eval_seeds = variations[int(self._train_portion*self._num_variation):] 

        if kwargs['gui'] == 'True':

            self.set_gui(True)
        else:
            self.set_gui(False)

        

        self.set_train()
        self._init_env(0)
        print('action space', self.action_space)
        self.no_op = np.zeros(*self.action_space.shape)

        self.eval_params = [{'eid': i, 'save_video': True} for i in range(10)]
        self.eval_params.extend([{'eid': i, 'save_video': False} for i in range(10, 30)])
    
    def get_name(self, episode_config=None):
        return  self._domain + "-" + self._task

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

    def get_max_interactive_steps(self):
        return self._max_env_step

    def get_no_op(self):
        return self.no_op
        

    def set_eval(self):
        self.training = False
    
    def set_train(self):
        self.training = True

    def set_gui(self, flag):
        self._headless = (not flag)

        if not self._headless:
            cv2.startWindowThread()
            cv2.namedWindow("simulation", cv2.WINDOW_NORMAL)

    def sample_random_action(self):
        action = np.random.uniform(self._spec.minimum, self._spec.maximum, self._spec.shape)
        return action
    
    def get_action_space(self):
        return self.action_space


    def reset(self, episode_config=None):
        self._env_step = 0  # Reset internal timer
        self._total_reward = 0

        if episode_config == None:
            self._init_env(np.random.choice(self._train_seeds if self.training else self._eval_seeds))
            self._save_frame = False
        else:
            eid = episode_config['eid']
            self._save_frame = episode_config['save_video'] if 'save_video' in episode_config else False
            self._init_env(self._train_seeds[eid] if self.training else self._eval_seeds[eid])

        if self._save_frame:
            self._frames = []
        
        timestep = self._env.reset()

        if not self._headless:
            self._display()

        return self._process_timestep(timestep)

    def step(self, action):
        
        reward = 0
        
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            if not self._headless:
                self._display()
            

            info = self._process_timestep(time_step)
            if self._save_frame:
                self._frames.append(info['observation']['rgb'])

            reward += info['reward']
            
            self._env_step += 1  # Increment internal timer
            done = (info['done'] or self._env_step == self._max_env_step)

                
            if done:
                break
        #print('reward', reward)
        info['reward'] = reward
        info['done'] = done

        self._total_reward += reward
        
        return info
    
    def _init_env(self, seed):
        self._env = suite.load(self._domain, self._task, task_kwargs={'random': seed})
        if self._pixel_observation:
             self._env = pixels.Wrapper(
                 self._env, 
                 render_kwargs={
                     'camera_id':0, 
                     'height': self._img_dim[0], 
                     'width': self._img_dim[1]})
             
        self._spec = self._env.action_spec()
        self.action_space = Box(self._spec.minimum, self._spec.maximum, dtype=np.float32)
        

    def _process_timestep(self, timestep):
        obs = {}
        for k, v in timestep.observation.items():
            obs[k] = v
        
        if self._pixel_observation:
            obs['rgb'] = timestep.observation['pixels'].copy()

        info = {}
        info['reward'] = timestep.reward
        info['observation'] = obs
        info['done'] = timestep.last()
        return info
    
    

    def _display(self):
        pixels = self._env.physics.render(height=200, width=200, camera_id=0)
        tmp_pixels = pixels.copy()
        pixels[:, :, 0], pixels[:, :, 1],  pixels[:, :, 2] =\
              tmp_pixels[:, :, 2], tmp_pixels[:, :, 1], tmp_pixels[:, :, 0]
        cv2.imshow('simulation', pixels)
        cv2.waitKey(1)