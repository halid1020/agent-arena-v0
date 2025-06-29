import logging
import numpy as np
from .pick_and_place_folding_policies \
    import RectFabricMultiStepFoldingExpertPolicy

from agent_arena.utilities.constants.rect_fabric import ALL_CORNER_INWARD_FOLDING_SUCCESS_THRESHOLD

class RectFabricAllCornerInwardFoldingExpertPolicy(RectFabricMultiStepFoldingExpertPolicy):
    def __init__(self, config):
        super().__init__(config)
        self.action_types.append('all-corner-inward-folding')
        if self.folding_noise:
            self.action_types.append('noisy-all-corner-inward-folding')
        kwargs= config.toDict()
        self.real2sim = False if 'real2sim' not in kwargs else kwargs['real2sim']

        self.folding_pick_order = np.asarray([[[0, 0]], [[1, 1]], [[0, 1]], [[1, 0]]])
        
        if self.real2sim:
            self.folding_place_order = np.asarray([[[0.43, 0.43]], [[0.57, 0.57]], [[0.43, 0.57]], [[0.57, 0.43]]])
            self.next_step_thresholds = [0.2] * 4
        else:
            self.folding_place_order = np.asarray([[[0.48, 0.8]], [[0.52, 0.52]], [[0.48, 0.52]], [[0.52, 0.48]]])
            self.next_step_thresholds = [0.08] * 4
        
        self.over_ratios = [0, 0, 0, 0]
        

    def success(self, info=None):
        if info is None:
            info = self.last_info
        flg = (self.fold_steps != len(self.folding_pick_order))
        flg  = flg and info['largest_particle_distance'] < ALL_CORNER_INWARD_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            flg = flg and info['canonical_IoU'] >= 0.7
        return flg