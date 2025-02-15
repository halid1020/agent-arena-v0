import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import json


from ..task import TaskWrapper
from ..metrics_utils import *
from .towel_flattening \
    import FlatteningWrapper
from scipy.optimize import minimize


def rotation_matrix_z(theta):
    """Returns the rotation matrix for a rotation around the z-axis by angle theta."""
    theta = np.asscalar(theta) if isinstance(theta, np.ndarray) else theta
    return np.asarray([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


def objective_function(cur_particles, goal_particles, theta):
    """Objective function to minimize: sum of squared distances after rotation."""
    R = rotation_matrix_z(theta)
    #print('R', R)
    rotated_particles = np.dot(cur_particles, R.T)  # Apply rotation
    # print('rotated particles', rotated_particles.shape, rotated_particles[0])
    # print('goal particles', goal_particles.shape, goal_particles[0])


    distances = np.linalg.norm(rotated_particles - goal_particles, axis=1)

    # find the one-to-one correspondence between the particles that makes the mean distance minimum

   

    return np.sum(distances**2)


class FoldingWrapper(TaskWrapper):
    def __init__(self, env, canonical=False):
        self.env = env
        self.canonical = canonical
        self.goals = None

    

    def step(self, action):
        #print('folding wraper action', action)
        return self._process_info(self.env.step(action))
    
    def reset(self, episode_config=None):
        info = self.env.reset(episode_config)
        return self._process_info(info)
    
    def success(self):
        # print('ldp', self._largest_particle_distance())
        # print('nc', self.get_normalised_coverage())
        is_success = self._largest_particle_distance() < 0.017
        if self.canonical:
            is_success = is_success and self._get_canonical_IoU() >= 0.7
        
        return is_success
    
    
    
    def evaluate(self,
            metrics = [
                'wrinkle_pixel_ratio', 
                'mean_particle_distance', 
                'largest_particle_distance', 
                'largest_corner_distance',
                'mean_edge_distance',
                'canonical_IoU',
                'success',
                'goal_mean_particle_distance',
                'largest_edge_distance']
        ):
        print('evaluate')
        ### go over metrics and compute them
        results = {}
        for metric in metrics:
            if metric == 'wrinkle_pixel_ratio':
                results[metric] = get_wrinkle_pixel_ratio(
                    self.render(mode='rgb'), 
                    self.get_cloth_mask(resolution=(128, 128)))
            elif metric == 'mean_particle_distance':
                results[metric] = self._mean_particle_distance()
            elif metric == 'largest_particle_distance':
                results[metric] = self._largest_particle_distance()
            elif metric == 'largest_corner_distance':
                results[metric] = self._largest_corner_distance()
            elif metric == 'canonical_IoU':
                if self.goals is not None:
                    results[metric] = self._get_canonical_IoU()
            elif metric == 'mean_edge_distance':
                results[metric] = self._mean_edge_distance()
            elif metric == 'goal_mean_particle_distance':
                results[metric] = self._goal_mean_particle_distance()
            elif metric == 'largest_edge_distance':
                results[metric] = self._largest_edge_distance()
            elif metric == 'success':
                results[metric] = self.success()
            else:
                raise NotImplementedError


        return results
    
    def _get_distance(self, positions, group_a, group_b):
        if positions is None:
            position = self._env.get_particle_positions()
        cols = [0, 2]
        pos_group_a = position[np.ix_(group_a, cols)]
        pos_group_b = position[np.ix_(group_b, cols)]
        distance = np.linalg.norm(pos_group_a-pos_group_b, axis=1)
        #print('distance', distance.shape)
        return distance
    
    def _mean_edge_distance(self, particles=None):
        
        distances = []
        edge_ids = self.get_edge_ids()
        edge_distance = []
        for group_a, group_b in self.fold_groups:
            group_ab = np.concatenate([group_a, group_b])
            group_ba = np.concatenate([group_b, group_a])
            distances = self._get_distance(particles, group_ab, group_ba)
            edge_distance.append(
                np.mean([distances[i] \
                         for i, p in enumerate(group_ab) if p in edge_ids]))
        
        return np.min(edge_distance)

    def _largest_edge_distance(self, particles=None):

        distances = []
        edge_ids = self.get_edge_ids()
        edge_distance = []
        for group_a, group_b in self.fold_groups:
            group_ab = np.concatenate([group_a, group_b])
            group_ba = np.concatenate([group_b, group_a])
            distances = self._get_distance(particles, group_ab, group_ba)
            edge_distance.append(
                np.max([distances[i] \
                        for i, p in enumerate(group_ab) if p in edge_ids]))
        
        return np.min(edge_distance)

    def _get_goal_particle_distances(self):
        """
        Return the distance between the goal particles and the current particles in meters.
        """
        # Particles are in the form of N*(x, z, y)
        cur_particles = self.env.get_particle_positions()
        best_particles = cur_particles
        goal_particles = self.goals[-1]['particle']

        if self.initial == 'centre-flatten':
            # Rearrange particles to N*(x, y, z)
            cur_particles = cur_particles[:, [0, 2, 1]]
            goal_particles = goal_particles[:, [0, 2, 1]]
            H, W = self.env.get_cloth_size()
            num_particles = H*W
            particle_grid_idx = np.array(list(range(num_particles))).reshape(H, W)
            min_value = np.inf
            for _ in range(4):
                initial_theta = 0
                partciels = cur_particles[particle_grid_idx.flatten()]
                result = minimize(lambda x: objective_function(partciels, goal_particles, x), initial_theta, bounds=[(-np.pi, np.pi)])
                optimal_theta = result.x[0]
                optimal_rotation_matrix = rotation_matrix_z(optimal_theta)

                partciels = np.dot(partciels, optimal_rotation_matrix.T)

                tmp_value = np.mean(np.linalg.norm(partciels - goal_particles, axis=1))
                if min_value > tmp_value:
                    best_particles = partciels
                    min_value = tmp_value

                particle_grid_idx = np.rot90(particle_grid_idx)


        # Get the particle-wise distance
        return np.linalg.norm(best_particles - goal_particles, axis=1)

    def _goal_mean_particle_distance(self):
        """
            Compare the goal particles and the current particles, and find the largest distance.
        """
        if self.goals is None:
            return np.nan

        value = np.mean(self._get_goal_particle_distances())
        print('goal mean', value)
        return value
    
    def _largest_corner_distance(self, particles=None):
        distances = []
        corner_distance = []
        cornder_ids = self.get_corner_ids()
        for group_a, group_b in self.fold_groups:
            group_ab = np.concatenate([group_a, group_b])
            group_ba = np.concatenate([group_b, group_a])
            distances = self._get_distance(particles, group_ab, group_ba)
            corner_distance.append(
                np.max([distances[i] \
                        for i, p in enumerate(group_ab) if p in cornder_ids]))
            #print(corner_distance)

        return np.min(corner_distance)


    def _mean_particle_distance(self, particles=None):

        distances = []
        for group_a, group_b in self.fold_groups:
            distances.append(np.mean(self._get_distance(particles, group_a, group_b)))
        value = np.min(distances)
        print('MPD', value)
        return value


    def _largest_particle_distance(self, particles=None):
        distances = []
        for group_a, group_b in self.fold_groups:
            distances.append(np.max(self._get_distance(particles, group_a, group_b)))
        
        return np.min(distances)
    

    def _get_canonical_IoU(arena):
        mask = arena.get_cloth_mask(resolution=(128, 128)).reshape(128, 128)
        depth = arena.get_goals()[-1]['depth']
        # resizes depth to 128x128
        depth = cv2.resize(depth, (128, 128))
        canonical_mask = (depth < 1.49).reshape(128,  128)

        IoUs = []

        for i in range(4):
            mask = np.rot90(mask, i)
            intersection = np.sum(np.logical_and(mask, canonical_mask))
            union = np.sum(np.logical_or(mask, canonical_mask))
            IoUs.append(intersection/union)

        return np.max(IoUs)
    

    ### Return List of dictotionary.
    def get_goals(self):
        return self.goals
    
    def get_goal(self):
        return self.goals[-1].copy()
    
    # def _generate_goals(self):
    #     logging.info('[softgym, folding_wrapper, load gaols] generate goal')
    #     info = self.env.set_to_flatten()
    #     info = self._process_info(info)
    #     #print('info largest_particle_distance {}'.format(info['largest_particle_distance']))
        

    #     #print('generating goal')
        
    #     episode_config = self.env.get_episode_config()
    #     #print('episode config generating goals', episode_config)
    #     self.oracle_policy.reset()
    #     self.oracle_policy.init(info)
    #     #actions = []
    #     while not self.oracle_policy.success(info):
    #         #print('reset to generate')
    #         self.env.reset(episode_config)
    #         actions = []
    #         #print('step 0')
    #         info = self.env.set_to_flatten()
    #         info = self._process_info(info)
    #         logging.info('[softgym, folding_wrapper, load gaols] info keys {}'\
    #                         .format(info.keys()))
    #         self.oracle_policy.reset()
    #         self.oracle_policy.init(info)
    #         #print('info largest_particle_distance {}'.format(info['largest_particle_distance']))
    #         while not self.oracle_policy.success(info) and not info['done']:
    #             #print('generate step')
    #             #print('info largest_particle_distance {}'.format(info['largest_particle_distance']))
    #             info['arena'] = self
    #             action = self.oracle_policy.act(info)
    #             actions.append(action)
    #             #print('action shape', action.shape)
    #             info = self.step(action)
    #             self.oracle_policy.update(info, action)
            
    #     #print('finish generating goals')
    #     self.goal = info['observation']
    #     self.goal['action'] = np.array(actions)
    #     self.goal['particle'] = self.get_particle_positions()
        
    

    # def _save_goal(self):

    #     eid = self.env.get_episode_id()
    #     mode = self.env.get_mode()
    #     if not os.path.exists(self._get_goal_path(eid, mode)):
    #         os.makedirs(self._get_goal_path(eid, mode))
    #     plt.imsave(self._get_goal_path(eid, mode) + '/rgb.png', self.goal['rgb'])
    #     np.save(self._get_goal_path(eid, mode) + '/depth.npy', self.goal['depth'])
    #     np.save(self._get_goal_path(eid, mode) + '/mask.npy', self.goal['mask'])

    def load_goals(self, eid, mode):
        
        if self._goal_cached(eid, mode):
            logging.info('[softgym, folding_wrapper, load gaols] load goal from cache')
            self._load_goal(eid, mode)
        else:
            self._generate_goals()
        
        self._save_goal()
        
        return [self.env.get_flatten_observation(), self.goal]

    def _save_goal(self):
        """
            Save goals, rgb, depth, and the action and the particle.
        """
        eid = self.env.get_episode_id()
        mode = self.env.get_mode()
       
        
        if not os.path.exists(self._get_goal_path(eid, mode)):
            os.makedirs(self._get_goal_path(eid, mode))

        for i in range(len(self.goals)):
            goal_path = self._get_goal_path(eid, mode) + f'/step_{i}'
            if not os.path.exists(goal_path):
                os.makedirs(goal_path)
            plt.imsave(goal_path + '/rgb.png', self.goals[i]['rgb'])
            np.save(goal_path + '/depth.npy', self.goals[i]['depth'])
            np.save(goal_path + '/mask.npy', self.goals[i]['mask'])
            np.save(goal_path + '/particle.npy', self.goals[i]['particle'])

            # Convert NumPy arrays in the action dictionary to lists
            action_serializable = {}
            for key, value in self.goals[i]['action'].items():
                if isinstance(value, np.ndarray):
                    action_serializable[key] = value.tolist()
                else:
                    action_serializable[key] = value
            
            # Save action as JSON
            with open(goal_path + '/action.json', 'w') as f:
                json.dump(action_serializable, f)

    def _load_goal(self, eid, mode):
        self.goals = []

        # count the number of steps
        goal_path = self._get_goal_path(eid, mode)
        step_count = 0
        while os.path.exists(goal_path + f'/step_{step_count}'):
            step_count += 1
        
        print('step count', step_count)
        
        for i in range(step_count):
            goal_path = self._get_goal_path(eid, mode) + f'/step_{i}'
            goal = {}
            goal['rgb'] = plt.imread(goal_path + '/rgb.png')
            goal['depth'] = np.load(goal_path + '/depth.npy')
            goal['mask'] = np.load(goal_path + '/mask.npy')
            
            # Load action from JSON and convert lists back to NumPy arrays if necessary
            with open(goal_path + '/action.json', 'r') as f:
                action_data = json.load(f)
            
            goal['action'] = {}
            for key, value in action_data.items():
                if isinstance(value, list):
                    goal['action'][key] = np.array(value)
                else:
                    goal['action'][key] = value
            
            goal['particle'] = np.load(goal_path + '/particle.npy')
            self.goals.append(goal)
        
        self.goal = self.goals[-1]
    
    def _generate_goals(self):
        episode_config = self.env.get_episode_config()
        

        
        info = self.env.reset(episode_config)
        info = self._process_info(info)
        goals = []
        self.oracle_policy.reset()
        self.oracle_policy.init(info)

            
        while info['done'] == False and info['success'] == False and self.oracle_policy.terminate() == False:
            action = self.oracle_policy.act(info)
            #print('generate goal action', action)
           
            #print('action shape', action.shape)
            info = self.step(action)
            info = self._process_info(info)
            self.oracle_policy.update(info, action)
            
            particles = self.get_particle_positions()
            
            goal = info['observation'].copy()
            goal['particle'] = particles
            goal['action'] = action

            goals.append(goal)
            print('append goals')
        
        print('len of goals', len(goals))
        self.goals = goals
        self.goal = self.goals[-1]
            

    def _process_info(self, info):
        
        if self.goals is not None:
            info['goal'] = self.get_goal()
            info['canonical_IoU'] = self._get_canonical_IoU()
            info['goals'] = self.get_goals()

        info['success'] = self.success()
        info['cloth_size'] = self.env.get_cloth_size()
        info['largest_particle_distance'] = self._largest_particle_distance()
        info['mean_particle_distance'] = self._mean_particle_distance()
        info['task'] = self.task_name
            
        return info
    
    

    # def _load_goal(self, eid, mode):
    #     goal = {}
    #     goal['rgb'] = plt.imread(self._get_goal_path(eid, mode) + '/rgb.png')[:, :, :3]
    #     goal['depth'] = np.load(self._get_goal_path(eid, mode) + '/depth.npy')
    #     goal['mask'] = np.load(self._get_goal_path(eid, mode) + '/mask.npy')
    #     return goal
    
    
    def _goal_cached(self, eid, mode):
        return os.path.exists(self._get_goal_path(eid, mode))
    
    def get_next_goal(self):
        if FlatteningWrapper._get_canonical_IoU(self) >= 0.8 \
            and self.env.get_normalised_coverage() >= 0.999:
            
            return self.goals[-1].copy()
        
        return self.goals[0].copy()
    
    def _get_goal_path(self, eid, mode):

        """
            Ovewrite from the TaskWrapper class of SoftGym, as all domain we follow the following structure
            for saving the goals.
        """
        
        return '{}/data/rect_fabric/goals/{}/{}/initial_{}/{}/{}_eid_{}'\
            .format(os.environ['AGENT_ARENA_PATH'], self.task_name, self.domain, 
                    self.initial, self.action, mode, eid)