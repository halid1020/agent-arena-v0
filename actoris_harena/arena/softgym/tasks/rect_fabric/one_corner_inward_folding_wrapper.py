import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging

from arena.softgym.task_wrappers.rect_fabric.folding_wrapper \
    import FoldingWrapper

from arena.softgym.task_wrappers.rect_fabric.flattening_wrapper \
    import FlatteningWrapper
from utilities.constants.rect_fabric import *

import api as ag_ar

class OneCornerInwardFoldingWrapper(FoldingWrapper):
    def __init__(self, env, canonical=False, 
                 domain='mono-square-fabric', initial='crumple', 
                 action='pixel-pick-and-place(1)'):
        super().__init__(env, canonical)
        self.env = env
        self.domain = domain
        self.initial = initial
        self.task_name = 'one-corner-inward-folding'
        self.oracle_policy = ag_ar.build_agent(
                'oracle-rect-fabric|action:{},task:one-corner-inward-folding,strategy:expert'.format(action),
                self.env
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

        for _ in range(4):
            x_split = X // 2
            upper_triangle_ids = np.triu_indices(x_split)
            particles = particle_grid_idx[:x_split, :x_split].copy()
            group_a = particles[upper_triangle_ids].flatten().copy()
            group_b = np.flip(np.flip(particles, axis=0), axis=1).T[upper_triangle_ids].flatten().copy()

            self.fold_groups.append((group_a, group_b))
            particle_grid_idx = np.rot90(particle_grid_idx)
        
        ### Load goal observation
        self.load_goals(self.env.get_episode_id(), self.env.get_mode())
        
        
        info_ = self.env.reset(episode_config)

        return self._process_info(info_)

    def success(self):
        is_success = self._largest_particle_distance() < ONE_CORNER_INWARD_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            is_success = is_success and self._get_canonical_IoU() >= 0.7
        
        return is_success
