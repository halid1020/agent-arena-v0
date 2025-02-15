import os
import numpy as np
import gym

from agent_arena.arena.arena import Arena
from arena.goal_condition_interface import GoalConditionInterface

from arena.deformable_raven.src.environment import Environment
from arena.deformable_raven.src import tasks
from arena.deformable_raven.src import cameras

from agent.transporter.utils.video_recorder import VideoRecorder

# ENV_ASSETS_DIR = os.environ["DEFORMABLE_RAVEN_ASSETS_DIR"]
MAX_ORDER = 3

goal_tasks = ['insertion-goal', 'cable-shape-notarget', 'cable-line-notarget',
            'cloth-flat-notarget', 'bag-color-goal']

class DeformableRavenWrapper(Arena, GoalConditionInterface):

    def __init__(self, task, gui=False):

        self.set_gui(gui)
        
        self.position_bounds = gym.spaces.Box(
            low=np.array([0.25, -0.5, 0.], dtype=np.float32),
            high=np.array([0.75, 0.5, 0.28], dtype=np.float32),
            shape=(3,),
            dtype=np.float32)

        self.action_space = gym.spaces.Dict({
            'pose0':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
            'pose1':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)))
        })
        self.camera_config = cameras.RealSenseD415.CONFIG

        self._frames = []
        
        self.task_name = task

        self._task = tasks.names[task]()

        self._vid_rec = VideoRecorder(
            save_dir='.',
            episode_idx=None,
            record_mp4=True,
            display=self.gui,
            verbose=False)
        
        self._task.set_video_recorder(self._vid_rec)
        self.logger_name = 'standard_logger'


        

        self.eval_params = [{'eid': i, 'save_video': False} for i in range(30)]
        for i in range(10):
            self.eval_params[i]['save_video'] = True

        if task in goal_tasks:
            for i in range(30):
                self.eval_params[i]['save_goal'] = True

        self.set_train()

    def get_name(self):
        return "DeformableRaven"
        
    def get_mode(self):
        if self.training:
            return "train"
        else:
            return "eval"
    
    def get_eval_configs(self):
        return self.eval_params
        
    def get_frames(self):
        return np.stack(self._vid_rec.frames)
    
    def clear_frames(self):
        self._vid_rec.__enter__()

    def get_action_space(self):
        return self.action_space
    
    def sample_random_action(self):
        ## sample random action from action space
        return self.action_space.sample()

    def get_no_op(self):
        return {
            'pose0': (self.position_bounds.high, 
                      np.array([0., 0., 0., 1.], dtype=np.float32)),
            'pose1': (self.position_bounds.high,
                      np.array([0., 0., 0., 1.], dtype=np.float32))
        }

    def set_eval(self):
        self._task.mode = 'test'
        self.training = False

    def set_train(self):
        self._task.mode = 'train'
        self.training = True
    
    def set_gui(self, flg):
        self._env = Environment(
            # ENV_ASSETS_DIR,
            disp=flg,
            camera_configs=cameras.RealSenseD415.CONFIG,
            # shared_memory=True,
            hz=480)
        self.gui = flg
    
    # TODO: implement this. If there is no goal saved, run with oracle and save the last observation.
    def get_goal(self):
        if (self.task_name in goal_tasks) and self.goal == None:
            expert_policy = self._task.oracle(self._env)
            info = self.info
            while not info['done']:
                action = expert_policy.act(self.info, None)
                info = self.step(action)
            self.goal = info['observation']

        return self.goal

    def reset(self, episode_config=None):

        self._step = 0
        self._total_reward = 0

        if episode_config == None:
            self.episode_id = np.random.randint(0, 1000)
            save_video = False
        else:
            self.episode_id = episode_config['eid']
            save_video = episode_config['save_video']

        

        config_id = self.episode_id*2 + (0 if self.training else 1)
        
        seed = 10**MAX_ORDER + config_id

       
        
        np.random.seed(seed)
        self._vid_rec.record_mp4 = False
        obs = self._env.reset(self._task)
        
        

        info = {}
        info['observation'] = {
            'color': obs['color'],
            'depth': obs['depth'],
            'rgb': obs['color'][0]
        }
        info['done'] = False
        self.info = info
        self.goal = None

        # Generate Goals
        print('Generating the goal ...')
        info['goal'] = self.get_goal()
        print('Finished generating the goal.')

        np.random.seed(seed)
        obs = self._env.reset(self._task)
        if save_video:
            self._vid_rec.record_mp4 = True
            self.clear_frames()

        info['action_space'] = self.get_action_space()
        
        return info
    
    def get_episode_id(self):
        return self.episode_id
    
    def get_step(self):
        return self._step

    def step(self, action):

        #print('\n\naction to step', action)
        #print('action keys', action.keys())

        pose0 = action['pose0']
        pose1 = action['pose1']

        act = {
            'pose0': pose0, 
            'pose1': pose1,
            'primitive': 'pick_place',
            #'camera_config': self.camera_config
        }
       
        if self.task_name == 'sweeping':
            act['primitive'] = 'sweep'
        elif self.task_name == 'pushing':
            act['primitive'] = 'push'



        obs, reward, done, _ = self._env.step(act)
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

        info['action_space'] = self.get_action_space()
        info['success'] = self.success()

        return info
    
    def evaluate(self):

        res = {
            'total_reward': self._total_reward,
            'success': self.success(),
        }

        return res
    
    def get_eval_params(self):
        return self.eval_params
    

    def success(self):
        return self._task.done()
    

    ###
    ### Arena specific functions
    ###

    def __getattr__(self, attr):
        if attr in ["render"]:
            return getattr(self._env, attr)
        else:
            raise AttributeError(f"'EnvWrapper' object has no attribute '{attr}'")

    # def render_camera(self, config):
    #     return self._env.render_camera(config)