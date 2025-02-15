import numpy as np

from agent_arena.agent.agent import Agent

class MixPolicy(Agent):
    def __init__(self, mix_policies, policy_weigths, action_dim, random_seed, **kwargs):
        #super().__init__()
        self.action_dim = action_dim
        self.random_sampler = np.random.RandomState(random_seed)
        self.policy_weights = policy_weigths
        self.policies = mix_policies
        self.policy_weights = self.policy_weights / np.sum(self.policy_weights)
        self.policy_idx = -1
    
    def act(self, state=None):
        self.policy_idx = self.random_sampler.choice(len(self.policies), p=self.policy_weights)
        return self.policies[self.policy_idx].act(state).reshape(*self.action_dim)
    
    def _reset(self):
        self.policy_idx = -1
        for policy in self.policies:
            policy._reset()

    def get_action_type(self):
        return self.policies[self.policy_idx].get_action_type()
    
    def get_name(self):
        return 'Mix Policy'

    def get_state(self):
        return {}
    
    def init(self, state):
        pass

    def update(self, state, action):
        pass

    def success(self, arena):
        return False