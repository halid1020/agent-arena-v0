import logging

from agent.oracle.rect_fabric.pick_and_place_folding_policies \
    import RectFabricMultiStepFoldingExpertPolicy
from utilities.constants.rect_fabric import *
import numpy as np

class RectFabricCrossFoldingExpertPolicy(RectFabricMultiStepFoldingExpertPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_types.append('cross-folding')
        if self.folding_noise:
            self.action_types.append('noisy-cross-folding')
        self.next_step_threshold = 0.2
    
    def init(self, info):
        # self.action_space = info['action_space']
        # self.no_op = info['no_op']
        H, W = info['cloth_size']
        if H > W:
            self.folding_pick_order = np.asarray([
                [[0, 0]], [[0, 1]], #[[0, 0.5]], [[0, 0]], [[0, 1]],
                [[0.5, 0]], #[[0, 0]], [[1, 0]],
                #[[0.5, 0]], [[0, 0]], [[1, 0]]
            ])
            self.folding_place_order = np.asarray([
                [[1, 0]], [[1, 1]], #[[1, 0.5]], [[1, 0]], [[1, 1]],
                #[[0.5, 0.9]], [[1, 1]], [[1, 1]],
                [[0.5, 1]], #[[1, 1]], [[1, 1]]
            
            ])
            self.over_ratios = [0.03]*3
        else:
            self.folding_pick_order = np.asarray([
                [[0, 0]], [[1, 0]], #[[0.5, 0]], [[0, 0]], [[1, 0]],
                [[0, 0.5]], #[[0, 0]], [[0, 1]],
                #[[0, 0.5]], [[0, 0]], [[0, 1]]
            ])
            self.folding_place_order = np.asarray([
                [[0, 1]], [[1, 1]], #[[0.5, 1]], [[0, 1]], [[1, 1]],
                #[[0.9, 0.5]], [[1, 1]], [[1, 1]],
                [[1, 0.5]], #[[1, 1]], [[1, 1]]
            ])
            self.over_ratios = [0.03]*3

    def success(self, info=None):
        if info is None:
            info = self.last_info

        flg = (self.fold_steps != len(self.folding_pick_order))
        print('LPD', info['largest_particle_distance'])
        flg  = flg and info['largest_particle_distance'] < CROSS_FOLDING_SUCCESS_THRESHOLD 
        if self.canonical:
            flg = flg and info['canonical_IoU'] >= FOLDING_IoU_THRESHOLD
        return flg
    

    
class RectFabricTwoPickerCrossFoldingExpertPolicy(RectFabricCrossFoldingExpertPolicy):
    

    def init(self, info):
        # self.action_space = info['action_space']
        # self.no_op = info['no_op']
        H, W = info['cloth_size']
        if H > W:
            print('here 2')
            self.folding_pick_order = [
                [[0, 0], [0, 1]],
                [[0, 0.5], [-1, -1]],
                [[0, 0], [0, 1]],

                [[0.5, 0], [0, 0]],
                [[-1, -1], [1, 0]],
                [[0.5, 0], [0, 0]],
                [[-1, -1], [1, 0]],

            ] #steps*num_picker*2
            self.folding_place_order = [
                [[1, 0], [1, 1]],
                [[1, 0.5], [-1, -1]],
                [[1, 0], [1, 1]],

                [[0.5, 1], [1, 1]],
                [[-1, -1], [1, 1]],
                [[0.5, 1], [1, 1]],
                [[-1, -1], [1, 1]],
            ]
        else:
            #print('here')
            self.folding_pick_order = [
                [[0, 0], [1, 0]],
                [[0.5, 0], [-1, -1]],
                [[0, 0], [1, 0]],
                [[0, 0.5], [0, 0]],
                [[-1, -1], [0, 1]],
                [[0, 0.5], [0, 0]],
                [[-1, -1], [0, 1]],
            ]
            
            self.folding_place_order = [
                [[0, 1], [1, 1]],
                [[0.5, 1], [-1, -1]],
                [[0, 0], [1, 1]],
                [[0.9, 0.5], [1, 1]],
                [[-1, -1], [1, 1]],
                [[1, 0.5], [1, 1]],
                [[-1, -1], [1, 1]],
            ]
        
        self.over_ratios = [
            0, -0.1, 0,
            0.03, 0.03, 0.03, 0.03
        ]