from agent.oracle.rect_fabric.pick_and_place_folding_policies import *

class CanonicaliseRandomFoldingCanonicalisePolicy(RectFabricMultiStepFoldingExpertPolicy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def act(self, state, arena):
        if super().finsihed():
            self.reset()
        return super().act(state, arena)