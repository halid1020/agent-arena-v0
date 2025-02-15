from ..cem import MPC_CEM
from agent_arena.agent.utilities.utils import *
import numpy as np
from gym.spaces import Box

class RectFabricPickPlaceClothMaskMPC(MPC_CEM):
    def __init__(self, config):

        super().__init__(config)
        #self.flatten_threshold =  kwargs['flatten_threshold']
        self.cloth_mask = config.cloth_mask
        self.swap_action = config.swap_action if 'swap_action' in config else True
        #self.no_op = kwargs['no_op']
        if self.cloth_mask == 'from_model':
            self.cloth_mask_threshold = config.cloth_mask_threshold

        self.name = 'Rectangular-fabric Cloth-mask MPC'

        # logging.info('[cloth-mask-mpc, init] action space {}'.format(self.action_space))
        #print('action shape', self.action_space)

    def get_name(self):
        return self.name + " on " + self.model.get_name()
    
    def get_phase(self):
        return 'flattening'

    def act(self, states, update=False):

        acts = []
        costs = []
        for state in states:
            # TODO: this is a hack to get the action space
            action_space = Box(low=-1, high=1, shape=(1, 4), dtype=np.float32)
            num_elites = int(0.1 * self.candidates)
            plan_hor = self.planning_horizon

            mean = np.tile(np.zeros([1, 4]).flatten(), [plan_hor])
            std = np.tile(np.ones([1, 4]).flatten(), [plan_hor])

            if self.cloth_mask == 'from_env':
                cloth_mask = state['observation']['mask']
            elif self.cloth_mask == 'from_model':
                cloth_mask = self.model.reconstruct_observation(self.model.cur_state)
                cloth_mask = cloth_mask.reshape(*cloth_mask.shape[-2:])
                cloth_mask = cloth_mask > self.cloth_mask_threshold

            
            for i in range(self.iterations):
                popsize = self.candidates
                samples = np.stack([np.random.normal(mean, std) for _ in range(popsize)]).reshape(popsize, plan_hor, -1)
                
                H, W = cloth_mask.shape[:2]

                first_pick_actions = ((samples[:, 0, :2] + 1) * (H / 2)).astype(int)
                first_pick_actions = first_pick_actions.astype(int).clip(0, H-1).reshape(self.candidates, -1)

                if self.swap_action:
                    # print('swapping actions')
                    valid_indices = cloth_mask[first_pick_actions[:, 1], first_pick_actions[:, 0]] == 1
                else:
                    # print('not swapping actions')
                    valid_indices = cloth_mask[first_pick_actions[:, 0], first_pick_actions[:, 1]] == 1
                samples = samples[valid_indices]
                popsize = samples.shape[0]

                if self.clip:
                    samples = np.clip(samples, action_space.low[:1], action_space.high[:1])
                    
                costs, _ = self._predict_and_eval(samples, state, 
                        goal=(state['goals'] if self.goal_condition else None))
                elites = samples[np.argsort(costs)][:num_elites]
                new_mean = np.mean(elites, axis=0)
                new_std = np.std(elites, axis=0)
                mean, std = new_mean, new_std

            ret_act = np.clip(mean.reshape(plan_hor, *(1, 4))[0], action_space.low[:1], action_space.high[:1])[0]
            
            self.internal_states[state['arena_id']] ={}
            acts.append(ret_act)

        return acts