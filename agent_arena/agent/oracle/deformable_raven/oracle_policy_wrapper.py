import os

from arena.deformable_raven.src import tasks 


class OraclePolicyWrapper():

    def __init__(self, task, env):
        self._task = env._task
        self._policy = self._task .oracle(env)
        
    def act(self, state, env):
        #print('state', state['observation'].keys())
        if 'color' is not state:
            state['color'] = state['observation']['color']
        if 'depth' is not state:
            state['depth'] = state['observation']['depth']
        action =  self._policy.act(state, None)
        return action
    
    def success(self, state):
        return self._task.done()
    
    def get_name(self):
        return 'deformable_raven_oracle'
    
    def reset(self):
        pass

    def init(self, state):
        pass

    def update_state(self, state, action):
        pass