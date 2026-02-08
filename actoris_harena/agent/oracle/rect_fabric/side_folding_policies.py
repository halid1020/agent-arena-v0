import logging
import numpy as np

from agent.oracle.rect_fabric.rectangular_folding_policies \
    import RectFabricMultiStepFoldingExpertPolicy
import api as ag_ar

from utilities.constants.rect_fabric import *


class RectFabricSideFoldingExpertPolicy(RectFabricMultiStepFoldingExpertPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_types.append('side-folding')
        if self.folding_noise:
            self.action_types.append('noisy-side-folding')
        print('HELLLO!!!!!')
    
    def init(self, info):
        print('HELLLO')
        H, W = info['cloth_size']
        self.folding_pick_order = [
                [[0, 0]], [[1, 0]], [[0.4, 0]], [[0, 0]], [[1, 0]]
            ]
            
        self.folding_place_order = [
            [[0, 0.45]], [[1, 0.45]], [[0.4, 0.5]], [[0, 0.5]], [[1, 0.5]]
        ]

        self.over_ratios = [0, 0, 0.04, 0.04, 0.04]

       
        
        
            

        # Shorten folding distance if W/(H/2) < 1
        small = min(H, W)
        large = max(H, W)
        print('value', small/(large/2.0) )
        if small/(large/2.0) < 1.5:
            self.folding_pick_order = [
                [[0, 0]], [[1, 0]], [[0.4, 0]], [[0, 0]], [[1, 0]], [[0.4, 0]], [[0, 0]], [[1, 0]]
            ]
                
            self.folding_place_order = [
                [[0, 0.3]], [[1, 0.3]], [[0.4, 0.3]], [[0, 0.45]], [[1, 0.45]], [[0.4, 0.5]],  [[0, 0.5]], [[1, 0.5]]
            ]
            self.over_ratios = [0, 0, 0, 0, 0, 0.04, 0.04, 0.04]

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
        flg  = info['largest_particle_distance'] < SIDE_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            flg = flg and info['canonical_IoU'] >= 0.7
        return flg
    
class RectFabricTwoPickerSideFoldingExpertPolicy(RectFabricSideFoldingExpertPolicy):

    def init(self, info):
        print('ALLLO')
        H, W = info['cloth_size']
        drag_ratio = H*W/100000
        #print('hello')

        length = max(H, W)
        width = min(H, W)

        ### when it is a big wide fabric, we need to fold step by step
        # if width > 80:
        span = 2400//width
        span_ratio = 1.0*span/length
        #print('span', span)
        num_steps = length//(span*2)
        self.folding_pick_order = []
        self.folding_place_order = []
        self.over_ratios = []
        #print('num_steps', num_steps)
        tp = 0.47
        for i in range(num_steps):
            target_pos = min(tp, 1.0*(i+1)*span_ratio)
            target_pos_1 = min(tp, 1.0*(i+1.5)*span_ratio)
            
            self.folding_pick_order.append([[0, 0], [1, 0]])
            self.folding_place_order.append([[0, target_pos], [1, target_pos]])
            self.over_ratios.append(0)
        
            self.folding_pick_order.append([[0.5, 0], [-1, -1]])
            self.folding_place_order.append([[0.5, (target_pos_1+ target_pos)/2], [-1, -1]])
            self.over_ratios.append(0)
        
        self.folding_pick_order.append([[0, 0], [1, 0]])
        self.folding_place_order.append([[0, tp], [1, tp]])
        self.over_ratios.append(0)

        self.folding_pick_order.append([[0.5, 0], [1, 0]])
        self.folding_place_order.append([[0.5, 0.49], [1, 0.49]])
        self.over_ratios.append(drag_ratio)

        self.folding_pick_order.append([[0, 0], [0.5, 0]])
        self.folding_place_order.append([[0, 0.49], [0.5, 0.49]])
        self.over_ratios.append(drag_ratio)


        self.folding_pick_order = np.asarray(self.folding_pick_order)
        
        self.folding_place_order = np.asarray(self.folding_place_order)
    

        if H > W:
            # folding dimenstion 1*3*2
            # swap the two values in the last dimension
            self.folding_pick_order = self.folding_pick_order[:, :, [1, 0]]
            self.folding_place_order = self.folding_place_order[:, :, [1, 0]]