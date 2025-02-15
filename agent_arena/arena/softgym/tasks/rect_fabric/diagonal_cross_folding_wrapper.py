import numpy as np

import os
import numpy as np
import matplotlib.pyplot as plt

from .folding_wrapper \
    import FoldingWrapper
import agent_arena as ag_ar
from agent_arena.utilities.constants.rect_fabric import *

class DiagonalCrossFoldingWrapper(FoldingWrapper):
    def __init__(self, env, canonical=False, 
                 domain='mono-square-fabric', initial='crumple', 
                action='pixel-pick-and-place(1)'):
        super().__init__(env, canonical)
        self.env = env
        self.domain = domain
        self.initial = initial
        self.task_name = 'diagonal-cross-folding'
        if 'real2sim' in domain:
            self.oracle_policy = ag_ar.build_agent(
                'oracle-rect-fabric|action:{},task:diagonal-cross-folding,strategy:real2sim-expert'.format(action)
            )
        else:
            self.oracle_policy = ag_ar.build_agent(
                    'oracle-rect-fabric|action:{},task:diagonal-cross-folding,strategy:expert'.format(action)
                )
        self.action = action

    def reset(self, episode_config=None):
        info_ = self.env.reset(episode_config)
        episode_config = self.env.get_episode_config()
        H, W = self.env.get_cloth_size()
        num_particles = H*W

        particle_grid_idx = np.array(list(range(num_particles))).reshape(H, W)
        ## Only allow square fabric
        assert H == W, "Only allow square fabric"


        self.fold_groups = []
        X, Y = particle_grid_idx.shape[0], particle_grid_idx.shape[1]
        upper_triangle_ids = np.triu_indices(X)
        group_a = np.concatenate([
            particle_grid_idx[upper_triangle_ids].flatten(),
            np.flip(particle_grid_idx, axis=0)[upper_triangle_ids].flatten()])
        
        group_b = np.concatenate([
            particle_grid_idx.T[upper_triangle_ids].flatten(),
            np.flip(particle_grid_idx, axis=0).T[upper_triangle_ids].flatten()])
        
        self.fold_groups.append((group_a, group_b))
        particle_grid_idx = np.rot90(particle_grid_idx)

        # state = self.get_state()
        # state['particle_pos'] = self.get_flattened_pos()
        # self.set_state(state)
        # self.wait_until_stable()

        ### Load goal observation
        self.goals = self.load_goals(self.env.get_episode_id(), self.env.get_mode())
        
        
        info_ = self.env.reset(episode_config)

        return self._process_info(info_)
    
    def success(self):
        is_success = self._largest_particle_distance() < DIAGNOL_CROSS_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            is_success = is_success and self._get_canonical_IoU() >= FOLDING_IoU_THRESHOLD
        
        return is_success
    