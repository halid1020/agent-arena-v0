import numpy as np
from gym.spaces import Box

from .pixel_pick_and_fling import PixelPickAndFling
from .pixel_pick_and_place import PixelPickAndPlace
from gym.spaces import Dict, Discrete, Box

class HybridActionPrimitive():

    def __init__(self, 
        #env,
        pick_lower_bound=[-1, -1],
        pick_upper_bound=[1, 1],
        place_lower_bound=[-1, -1],
        place_upper_bound=[1, 1],
        action_horizon=20,

        **kwargs):
        
        ### Environment has to be WorldPickAndFlingWrapper
        self.np_pnp = PixelPickAndPlace(**kwargs)
        self.np_pnf = PixelPickAndFling(**kwargs)
        #self.env = env

        
        self.num_pickers = 2
        space_low = np.concatenate([pick_lower_bound, place_lower_bound]*self.num_pickers)\
            .reshape(self.num_pickers, -1).astype(np.float32)
        space_high = np.concatenate([pick_upper_bound, place_upper_bound]*self.num_pickers)\
            .reshape(self.num_pickers, -1).astype(np.float32)
        self.action_horizon = action_horizon
        self.action_step = 0
    
    def get_no_op(self):
        return self.no_op
        
    def sample_random_action(self):
        return self.action_space.sample()

    def get_action_space(self):
        ## combine the action space of np_pnp and np_pnf
        pnp_action_space =  self.np_pnp.get_action_space()
        
        action_space = Dict({
            'norm-pixel-pick-and-place': pnp_action_space,
        })
        ## TODO: add the componenet of np_pnf
        return action_space
        
    
    def get_action_horizon(self):
        return self.action_horizon
    
    def reset(self, env):
        self.np_pnf.reset(env)
        info = env.get_info()
        self.action_step = 0
        return info
    
    ## It accpet action has shape (num_picker, 2, 3), where num_picker can be 1 or 2
    def step(self, env, action):
        self.action_step += 1
        swap = action['swap'] if 'swap' in action else False
        if 'norm-pixel-pick-and-fling' in action:
            action = action['norm-pixel-pick-and-fling']
            info = self.np_pnf.step(env, action)
        elif 'norm-pixel-pick-and-place' in action:
            action = action['norm-pixel-pick-and-place']
            action['swap'] = swap
            info = self.np_pnp.step(env, action)
        elif 'no-op' in action and action['no-op']:
            info = env.get_info()
        else:
            raise ValueError('Action not recognized')

        # print('action_step', self.action_step)
        # print('action_horizon', self.action_horizon)
        
        info['done'] = self.action_step >= self.action_horizon
        return info