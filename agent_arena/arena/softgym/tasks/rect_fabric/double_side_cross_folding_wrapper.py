import numpy as np

from environments.softgym.task_wrappers.folding_wrapper import FoldingWrapper

class DoubleSideCrossFoldingWrapper(FoldingWrapper):
    def __init__(self, env):
        self.env = env

    def reset(self, episode_config={'eid': None}):
        info = self.env.reset(episode_config)

        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        self.fold_groups = []
        for _ in range(2):
            X, Y = particle_grid_idx.shape
            x_split, y_split = X // 4, Y // 2
            group_a = np.concatenate([particle_grid_idx[:x_split].flatten(), particle_grid_idx[X-x_split:].flatten()])
            group_b = np.concatenate([
                np.flip(particle_grid_idx[x_split:2*x_split], axis=0).flatten(),
                np.flip(particle_grid_idx[X-x_split:X], axis=0).flatten()])
            
            group_a = np.concatenate([group_a, particle_grid_idx[:, :y_split].flatten()])
            group_b = np.concatenate([group_b, np.flip(particle_grid_idx[:, Y-y_split:], axis=1).flatten()])
            
                
            self.fold_groups.append((group_a, group_b))
            particle_grid_idx = np.rot90(particle_grid_idx)

        return info
    