import os
from agent_arena import Agent
import numpy as np

ENV_ASSETS_DIR = os.environ["RAVENS_ASSETS_DIR"]

class RavenOraclePolicyAdapter(Agent):

    def __init__(self, config):
        super().__init__(config)
        self._policy = None
        
    def single_act(self, info, update=False):
        
        arena_id = info['arena_id']
        if 'color' is not info:
            info['color'] = info['observation']['color']
        if 'depth' is not info:
            info['depth'] = info['observation']['depth']
        action =  self.internal_states[arena_id]['policy'].act(info, None)
        
        # return 14 values, [0-2] pose0 coord, [3-6] pose0 orient
        # [7-9] pose1 coord, [10-13] pose1 orient, 
        ret_action = np.concatenate([
            action['pose0'][0], action['pose0'][1], 
            action['pose1'][0], action['pose1'][1]])

        return ret_action
        
    def init(self, infos):

        for info in infos:
            arena = info['arena']
            arena_id = info['arena_id']
            self.internal_states[arena_id]['policy'] = arena._task.oracle(arena)

    def update(self, infos, actions):
        pass

    def get_state(self):
        return {}