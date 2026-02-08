import sys
import cv2
import numpy as np
import random 
# from scipy.spatial import distance_matrix
from scipy.spatial import ConvexHull #, convex_hull_plot_2d

from agent.policies.base_policies import *

from agent.policies.base_policies import BasePolicy, RandomPolicy
from agent.utilities.utils import *

class GarmentPickAndFlingExpertPolicy(RandomPolicy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.boundary_threshold = kwargs['boundary_threshold'] if 'boundary_threshold' in kwargs else 0.8
        self.camera_height = 1.5
        self.camera_to_world = #0.4135
        self.search_range = 0
        self.search_interval = 0.01
        self.verbose = True
        self.revealing_keypoint = None
    
    def reset(self):
        self.success = False
        self.revealing_keypoint = None

    
    ### (1) Match the key points toward their corresponding target points
    def act(self, state, environment, noise=False):
        if environment.is_flattened():
            self.success = True
            if self.verbose:
                print('case no-op')

            self.action_type = 'no-op'
            return np.clip(
                    self.no_op.astype(float).reshape(*self.action_dim), 
                    self.action_lower_bound, 
                    self.action_upper_bound)
        
        action = np.ones(self.action_dim)

        key_positions = environment.get_key_positions()
        N = key_positions.shape[0]       
        key_positions_2d = key_positions[:, [0, 2]]
        key_visible_positions, key_projected_positions = environment.get_visibility(
                key_positions, 
                cameras=["default"])
        
 
        canonical_target_key_positions = environment.get_flatten_key_positions()
        cananical_target_key_positions_2d = canonical_target_key_positions[:, [0, 2]]*1.1
        cananical_target_key_projected_position = cananical_target_key_positions_2d.copy()/(self.camera_height*self.camera_to_world)

        pairs = []

        for i in range(N):
            p = cananical_target_key_projected_position[i].copy()
            p[0] = -p[0]
            for j in range(N):

                if np.linalg.norm(p-cananical_target_key_projected_position[j]) < 0.1:
                    pairs.append((i, j))

        self.num_picker = environment.get_num_picker()
        
        ### sort the pairs by their y a-xis
        pairs = sorted(pairs, key=lambda x: cananical_target_key_projected_position[x[0]][1])

        if self.num_picker == 2:
            pid = 0
            for i in range(len(pairs)):
                if key_visible_positions[0][pairs[i][0]] and \
                   key_visible_positions[0][pairs[i][1]] and \
                   np.linalg.norm(key_positions_2d[pairs[i][0]] - key_positions_2d[pairs[i][1]]) > 0.1: 
                    pid = i
                    break
            action[0] = key_projected_positions[0][pairs[pid][0]].copy()
            action[1] = key_projected_positions[0][pairs[pid][1]].copy()


        return action.reshape(*self.action_dim)