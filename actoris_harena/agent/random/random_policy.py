from actoris_harena import Agent

class RandomPolicy(Agent):

    def __init__(self, config):
        super().__init__(config)

    def act(self, info_list, update=False):
        actions = []
        for info in info_list:
            if 'action_space' in info:
                action_space = info['action_space']
                actions.append(action_space.sample())
            else:
                raise ValueError('action_space not found in info')
        
        return actions
    
    def single_act(self, info, update=False):
        action_space = info['action_space']
        return action_space.sample()
    
    def get_name(self):
        return 'random'