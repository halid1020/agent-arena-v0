import os
import numpy as np
import matplotlib.pyplot as plt
from dotmap import DotMap

from .folding \
    import FoldingTask
import agent_arena as ag_ar
from agent_arena.utilities.constants.rect_fabric import ALL_CORNER_INWARD_FOLDING_SUCCESS_THRESHOLD
from scipy.optimize import minimize

class AllCornerInwardFolding(FoldingTask):
    def __init__(self, canonical=False, 
                 domain='mono-square-fabric', initial='crumple', action='pixel-pick-and-place(1)'):
        super().__init__(canonical)
        self.domain = domain
        self.initial = initial
        self.task_name = 'all-corner-inward-folding'
        config = DotMap()
        config.oracle = True
        if 'real2sim' in domain:
            self.oracle_policy = ag_ar.build_agent(
                'oracle-rect-fabric|action:{},task:all-corner-inward-folding,strategy:real2sim-expert'.format(action),
                config
            )
        else:    
            self.oracle_policy = ag_ar.build_agent(
                'oracle-rect-fabric|action:{},task:all-corner-inward-folding,strategy:oracle-towel'.format(action),
                config
            )
        self.action = action

    def reset(self, arena):
        info = super().info(arena)
        episode_config = arena.get_episode_config()
        H, W = arena.get_cloth_size()
        num_particles = H*W

        particle_grid_idx = np.array(list(range(num_particles))).reshape(H, W)
        ## Only allow square fabric
        assert H == W, "Only allow square fabric"

        self.fold_groups = []
        X, Y = particle_grid_idx.shape[0], particle_grid_idx.shape[1]

        x_split = X // 2
        upper_triangle_ids = np.triu_indices(x_split)
        
        group_a = np.concatenate([
            particle_grid_idx[:x_split, :x_split][upper_triangle_ids].flatten(), 
            particle_grid_idx[X-x_split:, X-x_split:][upper_triangle_ids].flatten(),
            particle_grid_idx[:x_split, X-x_split:][upper_triangle_ids].flatten(),
            particle_grid_idx[X-x_split:, :x_split][upper_triangle_ids].flatten()])
        
        group_b = np.concatenate([
            np.flip(np.flip(particle_grid_idx[:x_split, :x_split], axis=0), axis=1).T[upper_triangle_ids].flatten(),  
            np.flip(np.flip(particle_grid_idx[X-x_split:, X-x_split:], axis=0), axis=1).T[upper_triangle_ids].flatten(),
            particle_grid_idx[:x_split, X-x_split:].T[upper_triangle_ids].flatten(),
            particle_grid_idx[X-x_split:, :x_split].T[upper_triangle_ids].flatten()])

        self.fold_groups.append((group_a, group_b))

        ### Load goal observation
        self.load_goals(self.env.get_episode_id(), self.env.get_mode())
        
        

        return info
    
    def success(self, arena):
        is_success = self._largest_particle_distance() < ALL_CORNER_INWARD_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            is_success = is_success and self._get_canonical_IoU() >= 0.7
        
        return is_success