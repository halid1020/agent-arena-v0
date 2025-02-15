from ..cem import *
from utilities.utils import *

class RectFabricPickPlaceReadjustMPC(MPC_CEM):
    def __init__(self, model, **kwargs):

        super().__init__(model, **kwargs)
        self.readjust_pick = kwargs['readjust_pick']
        self.readjust_pick_threshold =  kwargs['readjust_pick_threshold']
        self.flatten_threshold =  kwargs['flatten_threshold']
        self.conservative_place =  kwargs['conservative_place']
        
        #self.no_op =  kwargs['no_op']

    
    def act(self, state, env=None):
        action =  super().act(state, env=env)

        if self.readjust_pick and env.get_normalised_coverage() < self.flatten_threshold:
            cloth_mask = env.get_cloth_mask()
            #print('cloth_mask', cloth_mask.shape, cloth_mask.dtype)
            new_pick = find_closest_pickpoint(cloth_mask, action[:2].copy())
            if np.linalg.norm(new_pick - action[:2]) < self.readjust_pick_threshold:
                action[:2] = new_pick.copy()

        action[2:] = action[:2] + \
            self.conservative_place*(action[2:] - action[:2])

        if env.get_normalised_coverage() >= self.flatten_threshold:
            action = np.asarray(self.no_op).reshape(4)

        return action.clip(self.action_lower_bound, self.action_upper_bound)