from abc import ABC, abstractmethod
import numpy as np

from agent_arena.agent.agent import Agent

class BasePolicy(Agent):
    
    def __init__(self):
        
        # self.no_op =  np.asarray(kwargs['no_op']) if 'no_op' in kwargs else np.zeros(self.action_dim)
        self.is_success = False
        self.action_types = []
    
    @abstractmethod
    def get_action_type(self):
        raise NotImplementedError
    
    @abstractmethod
    def success(self, arena):
        return self.is_success
    
    def get_action_types(self):
        return self.action_types

    def _reset(self):
        self.is_success = False

    def init(self, information):
        pass
    
    def update(self, information, action):
        pass


class MaxActionPolicy(BasePolicy):

    def __init__(self, **kwargs):
        super().__init__()
        self.action_space = kwargs['action_space'] # gym.spaces.Dict
        #print('Example action', self.act(None, None))

    def act(self, state=None, environment=None):
        return self.action_space.high
    

    


