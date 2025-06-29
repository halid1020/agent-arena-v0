import numpy as np

from .folding \
    import FoldingWrapper
import agent_arena as ag_ar
from agent_arena.utilities.constants.rect_fabric import *

class RectangularFoldingWrapper(FoldingWrapper):
    """
        Judge if the longer side of the fabric is folded in half.
    """

    def __init__(self, env, canonical=False, 
                 domain='mono-square-fabric', 
                 initial='crumple',
                 action='pixel-pick-and-place(1)'):
        super().__init__(env, canonical)
        self.domain = domain
        self.initial = initial
        self.task_name = 'rectangular-folding'
        if 'real2sim' in domain:
            self.oracle_policy = ag_ar.build_agent(
                'oracle-rect-fabric|action:{},task:rectangular-folding,strategy:real2sim-expert'.format(action)
            )
        else:
            self.oracle_policy = ag_ar.build_agent(
                    'oracle-rect-fabric|action:{},task:rectangular-folding,strategy:expert'.format(action)
                )
        self.action = action

    def reset(self, episode_config=None):
        info_ = self.env.reset(episode_config)
        episode_config = self.env.get_episode_config()

        H, W = self.env.get_cloth_size()
        num_particles = H*W
        
        particle_grid_idx = np.array(list(range(num_particles))).reshape(H, W)#.T  # Reversed index here

        if H < W:
            particle_grid_idx = particle_grid_idx.T
            H, W = W, H

        self.fold_groups = []

        if H == W:
            for _ in range(2):
                X = particle_grid_idx.shape[0]
                x_split = X // 2
                group_a = particle_grid_idx[:x_split-5, :].flatten()
                group_b = np.flip(particle_grid_idx[X-x_split+4:-1, :], axis=0).flatten()
                self.fold_groups.append((group_a, group_b))
                particle_grid_idx = np.rot90(particle_grid_idx)
        else:
            X = particle_grid_idx.shape[0]
            x_split = X // 2
            group_a = particle_grid_idx[:x_split-5, :].flatten()
            group_b = np.flip(particle_grid_idx[X-x_split+4:-1, :], axis=0).flatten()
            self.fold_groups.append((group_a, group_b))
            particle_grid_idx = np.rot90(particle_grid_idx)

        # ## TODO: debug this.
        self.load_goals(self.env.get_episode_id(), self.env.get_mode())

        info_ = self.env.reset(episode_config)

        return self._process_info(info_)
    
    def success(self):
        is_success = self._largest_particle_distance() < RECTANGLUAR_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            is_success = is_success and self._get_canonical_IoU() >= FOLDING_IoU_THRESHOLD
        
        return is_success