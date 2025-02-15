import logging
import numpy as np

from .pick_and_place_folding_policies \
    import RectFabricMultiStepFoldingExpertPolicy
from agent_arena.utilities.constants.rect_fabric import *

class RectFabricDiagonalCrossFoldingExpertPolicy(RectFabricMultiStepFoldingExpertPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_types.append('diagonal-cross-folding')
        if self.folding_noise:
            self.action_types.append('noisy-diagonal-cross-folding')

        self.real2sim = False if 'real2sim' not in kwargs else kwargs['real2sim']

        if self.real2sim:
            self.folding_pick_order = np.asarray([[[0, 0]], [[0, 1]]])
            self.folding_place_order =  np.asarray([[[0.95, 0.95]], [[0.95, 0]]])
            self.over_ratios = [0, 0]
            self.next_step_thresholds = [0.1]*2
        else:
            self.folding_pick_order = np.asarray([[[0, 0]], [[1, 1]]])
            self.folding_place_order =  np.asarray([[[0.98, 0.98]], [[0, 0.98]]])
            self.over_ratios = [0, 0]
            self.next_step_thresholds = [0.05]*2
    

    def success(self, info=None):
        if info is None:
            info = self.last_info
        logging.debug('[oracle, diagonal cross folding] largest_particle_distance {}'.format(info['largest_particle_distance']))
        #print('diagonal foldig, info largest_particle_distance {}'.format(info['largest_particle_distance']))
        flg = (self.fold_steps != len(self.folding_pick_order))
        flg  = flg and info['largest_particle_distance'] < DIAGNOL_CROSS_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            flg = flg and info['canonical_IoU'] >= FOLDING_IoU_THRESHOLD
        return flg