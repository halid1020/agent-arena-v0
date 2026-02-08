import numpy as np

from agent.policies.base_policies import BasePolicy


### The dynamic model must provide unroll_action, init_state and update_state functions
### Cost function should be choose.

class SuccessNoOp(BasePolicy):

    def __init__(self, base_policy, **kwargs):
        super().__init__(**kwargs)
        
        self.base_policy = base_policy
        self.no_op = kwargs['no_op']
    
    def act(self, state, env=None):
        action_space = env.get_action_space()
        if env.success():
            
            action = np.asarray(self.no_op)
            return np.clip(
                    action.astype(float).reshape(*action_space.shape), 
                    action_space.low,
                    action_space.high).reshape(*action_space.shape)
        
        return self.base_policy.act(state, env).reshape(*action_space.shape)

    def init(self, state):
        return self.base_policy.init_state(state)
    
    def update(self, state, action):
        return self.base_policy.update_state(state, action)
    