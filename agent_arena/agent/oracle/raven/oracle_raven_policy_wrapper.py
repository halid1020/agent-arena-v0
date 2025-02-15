import os

from agent_arena import Agent


ENV_ASSETS_DIR = os.environ["RAVENS_ASSETS_DIR"]

class OracleRavenPolicyWrapper(Agent):

    def __init__(self, config):
        super().__init__(config)
        self._policy = None
        self.task = config.task
        
    def act(self, infos):
        #print('state', state['observation'].keys())
        actions = []
        for info in infos:
            arena_id = info['arena_id']
            if 'color' is not info:
                info['color'] = info['observation']['color']
            if 'depth' is not info:
                info['depth'] = info['observation']['depth']
            action =  self.internal_states[arena_id]['policy'].act(info, None)
            actions.append(action)
        return actions

    def init(self, infos):

        for info in infos:
            arena = info['arena']
            arena_id = info['arena_id']
            self.internal_states[arena_id]['policy'] = arena._task.oracle(arena)

    def update(self, infos, actions):
        pass

    def get_state(self):
        return {}