from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.utils.data import DataLoader

from agent_arena.agent.agent import TrainableAgent
from agent.behaviour_cloning.dataset import BC_Dataset
import api as ag_ar

class BehaviourCloning(TrainableAgent):

    def __init__(self, config):
        self.dataset = BC_Dataset()
        self.collect_interval = config.collect_interval
        self.collect_episodes = config.collect_episodes
        self.update_steps = config.update_steps
        self.save_interval = config.save_interval
        
        self.batch_size = config.batch_size
        self.config = config
    
    @abstractmethod
    def process_state(self, state):
            
        """
            This method process the state for the dataset.
        """
        
        raise NotImplementedError
    
    def train(self, arena):
        
        metrics = self.load()
        self.demo_policy = ag_ar.build_agent(self.config.demo_policy, arena)
        start_step = metrics['update_step'][-1] \
            if 'update_step' in metrics.keys() else 0
        
        for u in tqdm(range(start_step, self.update_steps+1), desc='Training'):

            if u % self.collect_interval == 0:
                self.extend_dataset(
                    self.collect_trajectory(arena, self.collect_episodes),
                    arena
                )
            
            self.update_actor()

            if u % self.save_interval == 0:
                self.save()

    def update_actor(self):
        raise NotImplementedError
    
    def extend_dataset(self, trajectories, arena):
        
        """
            This method extends the dataset with the trajectory with the new action
            generated by the policy.
        """

        for traj in trajectories:
            for state, _ in traj:

                self.demo_policy.reset()
                self.demo_policy.init(state)
                demo_action = self.demo_policy.act(state, arena)

                self.dataset.append(
                    self.process_state(state), 
                    demo_action
                )
        self.train_dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            prefetch_factor=2,
            shuffle=True)

    def collect_trajectory(self, arena, num_episodes):

        """
            This method collects `num_episodes` trajectories from the arena 
            using the agent policy.
        """
        
        trajectories = []
        
        for _ in tqdm(range(num_episodes), desc='Collecting Trajectories'):
            state = arena.reset()
            self.reset()
            trajectory = []
            self.init_state(state)
            done = False
            while not done:
                action = self.act(state, arena)
                trajectory.append((state, action))
                state = arena.step(action)
                self.update_state(state, action)
                done = state['done']
            trajectories.append(trajectory)
        
        return trajectories