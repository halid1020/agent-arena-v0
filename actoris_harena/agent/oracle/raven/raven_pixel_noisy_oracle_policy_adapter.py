import os
import numpy as np
import pybullet as p
from actoris_harena import Agent

from .raven_pixel_oracle_policy_adapter import RavenPixelOraclePolicyAdapter

ENV_ASSETS_DIR = os.environ.get("RAVENS_ASSETS_DIR", "")

class RavenPixelNoisyOraclePolicyAdapter(RavenPixelOraclePolicyAdapter):
    """
    Noisy Oracle Policy that outputs normalized pixel-based actions (5-dim).
    Has a 50% probability of acting perfectly; otherwise injects Gaussian noise.
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "raven-pixel-noisy-oracle"
        self.add_noise_prob = config.get('add_noise_prob', 0.50)
        self.noise_scale = config.get('noise_sacle', 0.1)  # Adjust this standard deviation to make the error more or less severe

    def single_act(self, info, update=False):
        # 1. Get the pristine expert action from the parent class
        expert_action = super().single_act(info, update=update)
        
        # Ensure we don't add noise to a no-op/done action (all zeros)
        if np.all(expert_action == 0.0):
            return expert_action

        # 2. Roll the dice against the 50% success probability
        if np.random.rand() < self.add_noise_prob:
            # Disable noise: return the flawless expert action
            return expert_action
        
        # 3. Generate Gaussian noise for all 5 dimensions
        noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=expert_action.shape)
        noisy_action = expert_action + noise
        
        # 4. Enforce strict [-1.0, 1.0] boundaries
        clipped_action = np.clip(noisy_action, -1.0, 1.0)
        
        return clipped_action