import logging
import numpy as np

from agent.oracle.rect_fabric.pick_and_place_folding_policies \
    import RectFabricMultiStepFoldingExpertPolicy
from utilities.constants.rect_fabric import *

class RectFabricDiagonalFoldingExpertPolicy(RectFabricMultiStepFoldingExpertPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_types.append('diagonal-folding')
        if self.folding_noise:
            self.action_types.append('noisy-diagonal-folding')

        self.folding_pick_order = np.asarray([[[0, 0]]]) # step*num_picker*2
        self.folding_place_order = np.asarray([[[1, 1]]])
        self.over_ratios = [0.06]
    

    def success(self, info=None):
        if info is None:
            info = self.last_info
        logging.debug('[oracle, cross folding] largest_particle_distance {}'.format(info['largest_particle_distance']))
        #print('diagonal foldig, info largest_particle_distance {}'.format(info['largest_particle_distance']))
        #flg = (self.fold_steps != len(self.folding_pick_order))
        flg  = info['largest_particle_distance'] < DIAGONAL_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            flg = flg and info['canonical_IoU'] >= FOLDING_IoU_THRESHOLD
        return flg