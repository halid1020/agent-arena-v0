import numpy as np

from ...agent import Agent

### The dynamic model must provide unroll_action, init_state and update_state functions
### Cost function should be choose.


class MPC_CEM(Agent):

    def __init__(self, config):
        super().__init__(config)
        
        self.model = config.model
        self.action_space = config.action_space
        self.initialise_cost_fn(config.cost_fn)
        
        self.candidates = config.candidates
        self.planning_horizon = config.planning_horizon
        self.iterations = config.iterations
        self.clip = config.clip
        self.goal_condition = config.goal_condition
        self.verbose = config.verbose

    def initialise_cost_fn(self, cost_fn):
        if cost_fn == 'from_model':
            self.cost_fn = self.model.cost_fn
        else:
            raise NotImplementedError
    
    def reset(self, arena_ids):
        pass

    # def get_state(self):
    #     return {}
    
    def act(self, state, env=None):

        num_elites, popsize = int(0.1*self.candidates), self.candidates
        plan_hor = self.planning_horizon
        
        mean = np.tile(np.zeros(self.action_space.shape).flatten(), [plan_hor])
        std = np.tile(np.ones(self.action_space.shape).flatten(), [plan_hor])
        

        for i in range(self.iterations):

            samples = np.stack([np.random.normal(mean, std) for _ in range(popsize)])

            if self.clip:
                #print('non here clip')
                # samples = samples.clip(self.action_lower_bound, self.action_upper_bound)
                samples = samples\
                    .reshape(self.planning_horizon*popsize, *self.action_space.shape)\
                    .clip(self.action_space.low, self.action_space.high)\
                    .reshape(popsize, -1)
                    
            costs, _ = self._predict_and_eval(samples, state, goal=(env.get_goal() if self.goal_condition else None), )
            #print("CEM Iteration: ", i, "Cost (mean, std): ", np.mean(costs), ",", np.std(costs))
            elites = samples[np.argsort(costs)][:num_elites]
            new_mean = np.mean(elites, axis=0)
            new_std = np.std(elites, axis=0)
            mean, std = new_mean, new_std


        return mean.reshape(plan_hor, *self.action_space.shape)[0]\
              .clip(self.action_space.low, self.action_space.high)
    
    def _predict_and_eval(self, actions, state, goal=None):
 
        actions = actions.reshape(-1, self.planning_horizon, 4) #TODO: fix this
        #print('actions shape', actions.shape)
        pred_trajs = self.model.unroll_action_from_cur_state(actions, state)
        costs = self.cost_fn(pred_trajs, goal)
        
        return np.array(costs), pred_trajs
    
    # def init(self, state):
    #     pass
    
    # def update(self, state, action):
    #     pass