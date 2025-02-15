import os

from ..cem import *
from agent_arena.agent.utilities.utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class RectFabricPickPlaceClothContourMPCStepGoals(MPC_CEM):
    def __init__(self, config):

        super().__init__(config)
        #self.flatten_threshold =  kwargs['flatten_threshold']
        self.cloth_mask =config.cloth_mask
        #self.no_op = kwargs['no_op']
        if self.cloth_mask == 'from_model':
            self.cloth_mask_threshold = config.cloth_mask_threshold

        self.max_candidates = config.max_candidates

        self.name = 'Rectangular-fabric Cloth-contour MPC with Multiple-step Goals'
    
    def get_name(self):
        return self.name + " on " + self.model.get_name()


    def act(self,  state, env=None):
        #print('env.get_no_op()', env.get_no_op())

        if env.success():
            action = np.asarray(env.get_no_op())\
                .clip(self.action_space.low, self.action_space.high).reshape(4)
            return action
        
        num_elites = int(0.1 * self.max_candidates)
        plan_hor = self.planning_horizon

        mean = np.tile(np.zeros(self.action_space.shape).flatten(), [plan_hor])
        std = np.tile(np.ones(self.action_space.shape).flatten(), [plan_hor])

        ## create online folder if not exist in one line
        os.makedirs(os.path.join(self.model.config.save_dir, 'online'), exist_ok=True)

        mpimg.imsave(
            os.path.join(
                self.model.config.save_dir, 
                'online', 
                'rgb.png'),
            state['observation']['rgb']
        )

        cloth_edge_mask = env.get_cloth_edge_mask()
        mpimg.imsave(
            os.path.join(
                self.model.config.save_dir, 
                'online', 
                'edge.png'),
            
            (np.stack([cloth_edge_mask, cloth_edge_mask, cloth_edge_mask], axis=-1)*255).astype(np.uint8)
        )
        
        cloth_mask = env.get_cloth_mask()
        mpimg.imsave(
            os.path.join(
                self.model.config.save_dir, 
                'online', 
                'mask.png'),
            (np.stack([cloth_mask, cloth_mask, cloth_mask], axis=-1)*255).astype(np.uint8)
        )

        cloth_countor = np.zeros_like(cloth_mask)
        for i in range(1, cloth_mask.shape[0]-1):
            for j in range(1, cloth_mask.shape[1]-1):
                if cloth_mask[i, j] == 1:
                    if np.sum(cloth_mask[i-1:i+2, j-1:j+2]) == 9:
                        cloth_countor[i, j] = 0
                    else:
                        cloth_countor[i, j] = 1

        mpimg.imsave(
            os.path.join(
                self.model.config.save_dir, 
                'online', 
                'border.png'), 
            # extend the countor to have 3 channels
            (np.stack([cloth_countor, cloth_countor, cloth_countor], axis=-1)*255).astype(np.uint8)
        )
        
        
        
        cloth_countor = (cloth_countor + cloth_edge_mask).clip(0, 1)

        mpimg.imsave(
            os.path.join(
                self.model.config.save_dir, 
                'online', 
                'countor.png'), 
            # extend the countor to have 3 channels
            (np.stack([cloth_countor, cloth_countor, cloth_countor], axis=-1)*255).astype(np.uint8)
        )
        
        

        goal = env.get_next_goal()if self.goal_condition else None
        if goal is not None:
            mpimg.imsave(
                os.path.join(
                    self.model.config.save_dir, 
                    'online', 
                    'goal_rgb.png'), goal['rgb'])
            
            

        for i in range(self.iterations):
            popsize = self.candidates
            samples = np.stack([np.random.normal(mean, std) for _ in range(popsize)]).reshape(popsize, plan_hor, -1)

            
            H, W = cloth_countor.shape

            first_pick_actions = ((samples[:, 0, :2] + 1) * (H / 2)).astype(int)
            first_pick_actions = first_pick_actions.astype(int).clip(0, H-1).reshape(self.candidates, -1)
            
            valid_indices = cloth_countor[first_pick_actions[:, 1], first_pick_actions[:, 0]] == 1
            samples = samples[valid_indices]

            ## shuffle the samples interms of axis 0, and get first max_candidates
            np.random.shuffle(samples)
            samples = samples[:self.max_candidates]

            popsize = samples.shape[0]

            if self.clip:
                samples = np.clip(samples, self.action_space.low, self.action_space.high)

            costs, _ = self._predict_and_eval(samples, goal=goal)
            elites = samples[np.argsort(costs)][:num_elites]
            new_mean = np.mean(elites, axis=0)
            new_std = np.std(elites, axis=0)
            mean, std = new_mean, new_std
        return np.clip(mean.reshape(plan_hor, *self.action_space.shape)[0], self.action_space.low, self.action_space.high)[0]