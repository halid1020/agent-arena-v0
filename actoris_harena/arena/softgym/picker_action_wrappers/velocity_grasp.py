import numpy as np
from gym.spaces import Box
import matplotlib.pyplot as plt



class VelocityGrasp():

    def __init__(self, 
                 env,
                 action_repeat=10,
                 max_interactive_step=1000):
        
        ### Environment has to be WorldPickAndFlingWrapper
        self.env = env
        self.action_repeat = action_repeat
        self.max_interactive_step = max_interactive_step
    
    def reset(self, episode_config=None):
        self.interactive_step = 0
        return self.env.reset(episode_config)
    
    def step(self, action):
        
        #self.env.reset_control_step_info()
        for _ in range(self.action_repeat):
            info = self.env.step(action)
            self.interactive_step += 1
            if self.interactive_step >= self.max_interactive_step:
                info['done'] = True
                break

        return info
    

    def __getattr__(self, name):
        method = getattr(self.env, name)
        if callable(method):
            # If the attribute is a method, return a bound method
            return method.__get__(self.env, self.env.__class__)
        else:
            # If it's not a method, return the attribute itself
            return method

    def __setattr__(self, name, value):
        if name == 'env':
            super().__setattr__(name, value)
        else:
            setattr(self.env, name, value)

    def __delattr__(self, name):
        delattr(self.env, name)