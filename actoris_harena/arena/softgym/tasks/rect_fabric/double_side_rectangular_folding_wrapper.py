import numpy as np

from arena.softgym.task_wrappers.rect_fabric.folding_wrapper \
    import FoldingWrapper

class DoubleSideRectangularFoldingWrapper(FoldingWrapper):
    def __init__(self, env):
        self.env = env

    def reset(self, episode_config={'eid': None}):
        info = self.env.reset(episode_config)

        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        self.fold_groups = []
        for _ in range(2):
            X = particle_grid_idx.shape[0]
            x_split, x_split_2 = X // 4, X // 2
            group_a = np.concatenate([particle_grid_idx[:x_split].flatten(), particle_grid_idx[X-x_split:].flatten()])
            group_b = np.concatenate([
                np.flip(particle_grid_idx[x_split:2*x_split], axis=0).flatten(),
                np.flip(particle_grid_idx[X-x_split:X], axis=0).flatten()])
            
            group_a = np.concatenate([group_a, particle_grid_idx[:x_split_2].flatten()])
            group_b = np.concatenate([group_b, np.flip(particle_grid_idx[X-x_split_2:], axis=0).flatten()])
            
                
            self.fold_groups.append((group_a, group_b))
            particle_grid_idx = np.rot90(particle_grid_idx)


        return info
    