import numpy as np
from agent.base_policy import BasePolicy
from agent_arena.agent.agent import Agent

### The dynamic model must provide unroll_action, init_state and update_state functions
### Cost function should be choose.

class NoOp(Agent):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'no-op'

    def act(self, information):
        return information['no_op']
    
    def get_phase(self):
        return 'no-op'
    
    def get_state(self):
        return {}
    
    def init(self, information):
        pass

    def update(self, information, action):
        pass

    def reset(self):
        pass
    
    # def get_action_type(self):
    #     return 'no-op'