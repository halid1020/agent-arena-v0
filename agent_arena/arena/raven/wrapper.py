import os
import numpy as np

from agent_arena.arena.arena import Arena

from .environments.environment import Environment
from . import tasks
from .utils.video_recorder import VideoRecorder
from agent_arena import StandardLogger

ENV_ASSETS_DIR = os.environ["RAVENS_ASSETS_DIR"]

class RavenEnvWrapper(Arena):

    def __init__(self, task, disp=False):
        super().__init__()
        self._env = Environment(
            ENV_ASSETS_DIR,
            disp=disp,
            shared_memory=False,
            hz=240)
        
        self._control_step_info = {
            'frame': []
        }
        self.disp = disp
        
        self._task = tasks.names[task]()
        self._vid_rec = VideoRecorder(
            save_dir='.',
            episode_idx=None,
            record_mp4=True,
            display=self.disp,
            verbose=False)
        self._task.primitive._set_video_recorder(self._vid_rec)
        self.set_train()
        self.eval_params = [{'eid': i, 'save_video': False} for i in range(30)]
        self.val_params = [{'eid': 30+i, 'save_video': False} for i in range(3)]
        for i in range(10):
            self.eval_params[i]['save_video'] = True
        
        self.logger = StandardLogger()

        #self.action_horizon = 30
    
    def get_name(self):
        return "Raven"

    # TODO: make this function requried for the interface
    def get_action_horizon(self):
        return self._task.max_steps

    def get_mode(self):
        if self.training:
            return "train"
        else:
            return "eval"

    def get_action_space(self):
        return self._env.action_space

    def set_eval(self):
        self._task._set_mode('test')
        self.training = False

    def set_train(self):
        self._task._set_mode('train')
        self.training = True

    def get_action_space(self):
        return self._env.action_space
    
    def sample_random_action(self):
        ## sample random action from action space
        return self._env.action_space.sample()

    def get_no_op(self):
        return {
            'pose0': (self.position_bounds.high, 
                      np.array([0., 0., 0., 1.], dtype=np.float32)),
            'pose1': (self.position_bounds.high,
                      np.array([0., 0., 0., 1.], dtype=np.float32))
        }

    def reset(self, episode_config=None):

        self._step = 0
        self._total_reward = 0

        if 'save_video' not in episode_config.keys():
            episode_config['save_video'] = False

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

        #self.episode_id = episode_config['eid'] if episode_config else None
        config_id = self.episode_id*2 + (0 if self.training else 1)
        

       
        
        np.random.seed(config_id)
        self._env.seed(config_id)
        self._env.set_task(self._task)
        obs = self._env.reset()
        info = {}
        info['observation'] = {
            'color': obs['color'],
            'depth': obs['depth'],
            'rgb': obs['color'][0],
        }
        info['done'] = False
        info['arena'] = self
        info['arena_id'] = self.id
        
        
        return info
    
    def get_episode_id(self):
        return self.episode_id
    
    def get_step(self):
        return self._step
    
    def get_goal(self):
        return {}

    def step(self, action):

        obs, reward, done, other_info = self._env.step(action)
        self._step  += 1

        self._total_reward += reward

        info = {}
        

        info['observation'] = {
            'color': obs['color'],
            'depth': obs['depth'],
            'rgb': obs['color'][0]
        }

        info['done'] = ((self._step >= self._task.max_steps) or (self.success()))
        info['reward'] = reward
        info['others'] = other_info
        info['arena'] = self
        info['arena_id'] = self.id

        if reward >= 0.99:
            info['success'] = True
        else:
            info['success'] = False

        return info
    
    def evaluate(self):

        res = {
            'total_reward': self._total_reward,
        }

        return res
    
    def get_eval_configs(self):
        return self.eval_params
    
    def get_val_configs(self):
        return self.val_params

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

    
   
    
    # # TODO
    # def set_save_control_step_info(self, flg):
    #     self._save_control_step_info = flg

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

    def success(self):
        return self._total_reward > 1-1e-6

    

    ###
    ### Arena specific functions
    ###

    def __getattr__(self, attr):
        if attr in ["obj_ids", "render_camera", "add_object"]:
            return getattr(self._env, attr)
        else:
            raise AttributeError(f"'EnvWrapper' object has no attribute '{attr}'")

    # def render_camera(self, config):
    #     return self._env.render_camera(config)