import logging
import numpy as np
from math import ceil
from .pick_and_place_folding_policies \
    import RectFabricMultiStepFoldingExpertPolicy
import agent_arena as ag_ar
from agent_arena.utilities.constants.rect_fabric import *

class RectFabricRectangularFoldingExpertPolicy(RectFabricMultiStepFoldingExpertPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_types.append('rectangular-folding')
        if self.folding_noise:
            self.action_types.append('noisy-rectangular-folding')
        
        self.real2sim = False if 'real2sim' not in kwargs else kwargs['real2sim']
        #print('sim2real folding', self.real2sim)
    
    def init(self, info):
        # self.action_space = info['action_space']
        # self.no_op = info['no_op']
        #print('hello!!!!!')
        H, W = info['cloth_size']
        length = max(H, W)
        width = min(H, W)

        if self.real2sim:

            self.folding_pick_order = [
                [[0, 0]], [[1, 0]], [[0, 0]]
            ]
            self.folding_place_order = [
                [[0.4, 1.0]], [[1, 0.95]],  [[0, 1.0]]
            ]
            self.over_ratios = [0.1, 0, 0]
            self.next_step_thresholds = [0.3, 0.1, 0.1]
            
        else:
            self.next_step_threshold = 0.08
            phase_steps = ceil(length/width * 2)
            span_per_phase = 1.0/phase_steps
            self.folding_pick_order = []
            self.folding_place_order = []
            self.over_ratios = []
            for i in range(phase_steps):
                target_pos = min(0.9, 1.0*(i+1)*span_per_phase)
                tt_pos = min(1.0, 1.0*(i+1.25)*span_per_phase)
                self.folding_pick_order.extend([[[0, 0]], [[1, 0]], [[0.4, 0]]])
                self.folding_place_order.extend([[[0, target_pos]], [[1, target_pos]], [[0.4, tt_pos]]])
                self.over_ratios.extend([0, 0, 0])
            
            self.folding_pick_order.extend([[[0, 0]], [[1, 0]]])
            self.folding_place_order.extend([[[0, 1]], [[1, 1]]])
            self.over_ratios.extend([0.04, 0.04])

            self.folding_pick_order.extend([[[0, 0]], [[1, 0]]])
            self.folding_place_order.extend([[[0, 1]], [[1, 1]]])
            self.over_ratios.extend([0.04, 0.04])

        self.folding_pick_order = np.asarray(self.folding_pick_order)
        self.folding_place_order = np.asarray(self.folding_place_order)
        #print('shape', self.folding_pick_order)


        if H > W:
            ## flip the pick and place order
            #print('shape', self.folding_pick_order.shape)

            self.folding_pick_order = self.folding_pick_order[:, :, [1, 0]]
            self.folding_place_order = self.folding_place_order[:, :, [1, 0]]
        
        

    def success(self, info=None):
        if info is None:
            info = self.last_info
        logging.debug('[oracle, rectangular folding] largest_particle_distance {}'.format(info['largest_particle_distance']))
        #print('rectangular folding, info largest_particle_distance {}'.format(info['largest_particle_distance']))
        #flg = (self.fold_steps >= len(self.folding_pick_order)-1)
        #print('largest_particle_distance', info['largest_particle_distance'])
        flg  = info['largest_particle_distance'] < RECTANGLUAR_FOLDING_SUCCESS_THRESHOLD 
        if self.canonical:
            flg = flg and info['canonical_IoU'] >= FOLDING_IoU_THRESHOLD
        return flg
    

    
class RectFabricTwoPickerRectangularFoldingExpertPolicy(RectFabricRectangularFoldingExpertPolicy):
    

    def init(self, info):
        H, W = info['cloth_size']

        length = max(H, W)
        width = min(H, W)

        ### when it is a big wide fabric, we need to fold step by step
        # if width > 80:
        span = 2400//width
        span_ratio = 1.0*span/length
        #print('span', span)
        num_steps = length//span
        self.folding_pick_order = []
        self.folding_place_order = []
        self.over_ratios = []
        #print('num_steps', num_steps)
        for i in range(num_steps):
            target_pos = min(1.0, 1.0*(i+1)*span_ratio)
            target_pos_1 = min(1.0, 1.0*(i+1.5)*span_ratio)
            
            self.folding_pick_order.append([[0, 0], [1, 0]])
            self.folding_place_order.append([[0, target_pos], [1, target_pos]])
            self.over_ratios.append(0)
        
            self.folding_pick_order.append([[0.5, 0], [-1, -1]])
            self.folding_place_order.append([[0.5, (target_pos_1+ target_pos)/2], [-1, -1]])
            self.over_ratios.append(0)
        
        self.folding_pick_order.append([[0, 0], [1, 0]])
        self.folding_place_order.append([[0, 1], [1, 1]])
        self.over_ratios.append(0)

        self.folding_pick_order.append([[0.5, 0], [1, 0]])
        self.folding_place_order.append([[0.5, 1], [1, 1]])
        self.over_ratios.append(0.5)

        self.folding_pick_order.append([[0, 0], [0.5, 0]])
        self.folding_place_order.append([[0, 1], [0.5, 1]])
        self.over_ratios.append(0.5)


        self.folding_pick_order = np.asarray(self.folding_pick_order)
        
        self.folding_place_order = np.asarray(self.folding_place_order)
    

        if H > W:
            # folding dimenstion 1*3*2
            # swap the two values in the last dimension
            self.folding_pick_order = self.folding_pick_order[:, :, [1, 0]]
            self.folding_place_order = self.folding_place_order[:, :, [1, 0]]