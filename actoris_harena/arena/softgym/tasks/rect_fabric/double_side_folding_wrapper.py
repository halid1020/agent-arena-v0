import os
import numpy as np
import matplotlib.pyplot as plt
import logging


from arena.softgym.task_wrappers.rect_fabric.folding_wrapper \
    import FoldingWrapper
import api as ag_ar
from utilities.constants.rect_fabric import *

class DoubleSideFoldingWrapper(FoldingWrapper):
    def __init__(self, env, canonical=False, 
                 domain='mono-square-fabric', 
                 initial='crumple',
                 action='pixel-pick-and-place(1)'):
        super().__init__(env, canonical)
        self.env = env
        self.domain = domain
        self.initial = initial
        self.task_name = 'double-side-folding'
        self.oracle_policy = ag_ar.build_agent(
                'oracle-rect-fabric|action:{},task:double-side-folding,strategy:expert'.format(action)
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
                x_split = X // 4
                group_a = np.concatenate([
                    particle_grid_idx[:x_split-3].flatten(), 
                    particle_grid_idx[X-x_split+3:].flatten()])
                group_b = np.concatenate([
                    np.flip(particle_grid_idx[x_split+2:x_split*2-1], axis=0).flatten(),
                    np.flip(particle_grid_idx[X-2*x_split+1:X-x_split-2], axis=0).flatten()])
                
                    
                self.fold_groups.append((group_a, group_b))
                particle_grid_idx = np.rot90(particle_grid_idx)
        else:
            logging.info('[softgym, double side folding wrapper], H != W, set folding gourps')
            X = particle_grid_idx.shape[0]
            #print('particle_grid_idx.shape {}'.format(particle_grid_idx.shape))
            x_split = X // 4
            group_a = np.concatenate([
                    particle_grid_idx[:x_split].flatten(), 
                    particle_grid_idx[X-x_split:].flatten()])
            group_b = np.concatenate([
                    np.flip(particle_grid_idx[x_split:x_split*2], axis=0).flatten(),
                    np.flip(particle_grid_idx[X-2*x_split:X-x_split], axis=0).flatten()])
            
                
            self.fold_groups.append((group_a, group_b))

        if self.initial == 'crumple':
            pass
        else:
            self.load_goals(self.env.get_episode_id(), self.env.get_mode())

        info_ = self.env.reset(episode_config)

        return self._process_info(info_)
    
    def success(self):
        is_success = self._largest_particle_distance() < DOUBLE_SIDE_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            is_success = is_success and self._get_canonical_IoU() >= FOLDING_IoU_THRESHOLD
        
        return is_success
    