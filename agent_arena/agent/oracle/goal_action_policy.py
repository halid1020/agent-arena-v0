import numpy as np
import gym

from agent.oracle.base_policy import BasePolicy

class GoalActionPolicy(BasePolicy):

    def __init__(self, **kwargs):
        super().__init__()
        #self.action_space = kwargs['action_space'] # gym.spaces.Dict
        #print('Example action', self.act(None, None))
        self.step = 0

    def get_phase(self):
        return "default"

    def success(self, info):
        return self.step >= len(info['goal']['action'])

    def act(self, info=None, environment=None):
        #print('goal action policy act')
        actions = info['goal']['action']
        no_op = info['no_op']
        print('info largest_particle_distance {}'.format(info['largest_particle_distance']))
        if self.success(info):
            return no_op
        
        action = actions[self.step]
        self.step += 1
        print('goal action', action)
        
        return action
    
    def get_state(self):
        return {}

    def get_action_type(self):
        return 'goal_action'
    
    def get_name(self):
        return 'goal_action'
    
    def reset(self):
        self.step = 0
    
    def init(self, info):
        pass

    def update(self, info, action):
        pass