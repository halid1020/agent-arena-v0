import logging
import numpy as np

from agent.oracle.rect_fabric.pick_and_place_folding_policies \
    import RectFabricMultiStepFoldingExpertPolicy
from utilities.constants.rect_fabric import *

class RectFabricDoubleCornerInwardFoldingExpertPolicy(RectFabricMultiStepFoldingExpertPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_types.append('double-corner-inward-folding')
        if self.folding_noise:
            self.action_types.append('noisy-one-corner-inward-folding')

        self.folding_pick_order = np.asarray([[[0, 0]], [[1, 1]]]) # step*num_picker*2
        self.folding_place_order = np.asarray([[[0.48, 0.48]], [[0.52, 0.52]]])
        self.over_ratios = [0, 0]
    

    def success(self, info=None):
        if info is None:
            info = self.last_info
        flg = (self.fold_steps != len(self.folding_pick_order))
        flg  = flg and info['largest_particle_distance'] < DOUBLE_CORNER_INWARD_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            flg = flg and info['canonical_IoU'] >= FOLDING_IoU_THRESHOLD
        return flg