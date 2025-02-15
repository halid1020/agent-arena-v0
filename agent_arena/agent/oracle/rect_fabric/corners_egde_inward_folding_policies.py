import logging
import numpy as np

from .rectangular_folding_policies \
    import RectFabricMultiStepFoldingExpertPolicy
import agent_arena as ag_ar

from agent_arena.utilities.constants.rect_fabric import *


class RectFabricCornersEdgeFoldingExpertPolicy(RectFabricMultiStepFoldingExpertPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_types.append('corners-edge-inward-folding')
        if self.folding_noise:
            self.action_types.append('noisy-corners-edge-inward-folding')
        self.next_step_threshold = 0.2

    def init(self, info):
        H, W = info['cloth_size']
        self.folding_pick_order = [
                [[0, 1]], [[1, 1]], 

                [[0, 0]], [[1, 0]], 
                
                # [[0.4, 0]], 
                # [[0, 0]], [[1, 0]],
                
                
            ]
            
        self.folding_place_order = [
            [[0.43, 0.57]], [[0.57, 0.57]],

            [[0, 0.44]], [[1, 0.44]], 
            
            # [[0.4, 0.5]], 
            # [[0, 0.5]], [[1, 0.5]],
            
            
        ]

        self.over_ratios = [0, 0, 0, 0, 0.04, 0.04, 0.04]

        # Shorten folding distance if W/(H/2) < 1
        small = min(H, W)
        large = max(H, W)
        if small/(large/2.0) < 1.5:
            self.folding_pick_order = [
                [[0, 0]], [[1, 0]], [[0.4, 0]], 
                [[0, 0]], [[1, 0]], [[0.4, 0]], 
                [[0, 0]], [[1, 0]],

                [[0, 1]], [[1, 1]], 
            ]
                
            self.folding_place_order = [
                [[0, 0.3]], [[1, 0.3]], [[0.4, 0.3]], 
                [[0, 0.45]], [[1, 0.45]], [[0.4, 0.5]],  
                [[0, 0.5]], [[1, 0.5]],

                [[0.48, 0.52]], [[0.52, 0.52]],
            ]
            self.over_ratios = [0, 0, 0, 0, 0, 0.04, 0.04, 0.04, 0, 0]

        self.folding_pick_order = np.asarray(self.folding_pick_order)
        self.folding_place_order = np.asarray(self.folding_place_order)

        if H > W:
            ## flip the pick and place order
            #print('shape', self.folding_pick_order.shape)

            self.folding_pick_order = self.folding_pick_order[:, :, [1, 0]]
            self.folding_place_order = self.folding_place_order[:, :, [1, 0]]
    
    def success(self, info=None):
        if info is None:
            info = self.last_info
        logging.debug('[oracle, side folding] largest_particle_distance {}'.format(info['largest_particle_distance']))
        #print('info largest_particle_distance {}'.format(info['largest_particle_distance']))

        #flg = (self.fold_steps >= len(self.folding_pick_order)-1)
        flg  = info['largest_particle_distance'] < CORNERS_EDGE_INWARD_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            flg = flg and info['canonical_IoU'] >= 0.7
        return flg