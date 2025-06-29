import os

import torch
import numpy as np
from tqdm import tqdm

from agent_arena import TrainableAgent
from agent_arena.utilities.networks.utils import *


class REINFORCE(TrainableAgent):
    
        def __init__(self, config):
            super().__init__(config)
            self.name = "REINFORCE"
            
            if self.config.action_mode == "discrete":
                self.policy_network = torch.nn.Sequential(
                    torch.nn.Linear(self.config.state_size, self.config.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.config.hidden_size, self.config.action_size),
                    torch.nn.Softmax(dim=0)
                ).to(self.config.device)
            else:
                raise NotImplementedError("Only discrete action space is supported")
            
            self.optimiser =  OPTIMISERS[config.optimiser.name](
                self.policy_network.parameters(), 
                **config.optimiser.params)
            
            self.update_step = 0
            self.save()
            
        
        def reset(self):
            pass

        def init(self, information):
            pass

        def update(self, information, action):
            pass
    
            
        def act(self, information):
            act_prob = self.policy_network(
                torch.from_numpy(information['state']).float().to(self.config.device)
            )
            action = np.random.choice(np.array([0,1]), p=act_prob.detach().cpu().numpy())
            return action
        
        def train(self, update_steps, arena):
            self.set_train()
            
            assert arena is not None, "arena is not provided"

            start_update_step = self.load() + 1
            end_update_step = min(start_update_step + update_steps, self.config.total_update_steps)

            for u in tqdm(range(start_update_step, end_update_step)):
                
                trajs = self._collect_trajectories(arena) # N * H
                rewards = [[r for _, _, r in trj] for trj in trajs]
                states = [[s for s, _, _ in trj] for trj in trajs]
                actions = [[a for _, a, _ in trj] for trj in trajs]
                
                
                
                # make rewards into a tensor, pad with 0 if the length is less than the intended horizon
                rewards_ts = torch.tensor(
                    rewards
                        # [r + [0] * (self.config.collect_episode_horizon - len(r)) for r in rewards]
                    ).to(self.config.device)
                #print('rewards_ts', rewards_ts)

                # create dones_ts that has the same shape as rewards_ts
                dones_ts = torch.tensor(
                        [[0] * len(r) + [1] * (self.config.collect_episode_horizon - len(r)) for r in rewards]
                    ).to(self.config.device).float()

                # calculate reward-to-gos with discount factor self.config.gamma
                batch_Gvals =[]
                for i in range(len(rewards[0])):
                    new_Gval=0
                    power=0
                    for j in range(i,len(rewards[0])):
                        new_Gval=new_Gval+ \
                                (self.config.gamma**power)*rewards[0][j]
                        power+=1
                    batch_Gvals.append(new_Gval)

                expected_returns_batch=torch.FloatTensor(batch_Gvals).to(self.config.device)
                #print('expected_returns_batch', expected_returns_batch)
                expected_returns_batch /= expected_returns_batch.max()   
                
                states_ts = torch.tensor(
                        states
                        # [s + [s[-1]] * (self.config.collect_episode_horizon - len(s)) for s in states]
                    ).to(self.config.device).float()

                actions_ts = torch.tensor(
                        actions
                        # [a + [a[-1]] * (self.config.collect_episode_horizon - len(a)) for a in actions]
                    ).to(self.config.device).float()
                
                states_ts = states_ts.reshape(-1, self.config.state_size)
                actions_ts = actions_ts.reshape(-1, 1)
                done_ts = dones_ts.reshape(-1, 1)
                #print('done_ts', done_ts.shape)

                pred_probs = self.policy_network(states_ts.float())

                if self.config.action_mode == "discrete":
                    action_probs = pred_probs.gather(dim=1, index=actions_ts.long().view(-1, 1)).squeeze()
          
                
                loss = -torch.sum(torch.log(action_probs) * expected_returns_batch)
                #print('loss', loss)
                
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if u % 50 == 0:
                    print('Update step {}\t Average Score: {:.2f}'
                            .format(u, np.sum(rewards)))
            
            self.update_step = end_update_step - 1
            self.save()

        def set_train(self):
            self.policy_network.train()
        
        def set_eval(self):
            self.policy_network.eval()

        def save(self, path=None):
            if path is None:
                path = self.config.save_dir + '/checkpoints'
            
            # make directory regardless of the existence
            os.makedirs(path, exist_ok=True)

            path = path + '/model_{}.pth'.format(self.update_step)
            torch.save(
                {
                    'policy_network': self.policy_network.state_dict(),
                    'optimiser': self.optimiser.state_dict(),
                }
                , path)
            return True
        
        def load(self, path=None):
            if path is None:
                path = self.config.save_dir + '/checkpoints'
           
            if os.path.exists(path):
                # get the latest checkpoint
                checkpoints = os.listdir(path)
                checkpoints = [int(cp.split('_')[-1].split('.')[0]) for cp in checkpoints]
                checkpoints.sort()
                checkpoint = checkpoints[-1]
                checkpoint_dict = torch.load(path + '/model_{}.pth'.format(checkpoint))
                self.policy_network.load_state_dict(checkpoint_dict['policy_network'])
                self.optimiser.load_state_dict(checkpoint_dict['optimiser'])
                return checkpoint
            
            return -1
        
        def load_checkpoint(self, checkpoint):
            path = self.config.save_dir + '/checkpoints'
            if os.path.exists(path + '/model_{}.pth'.format(checkpoint)):
                checkpoint_dict = torch.load(path + '/model_{}.pth'.format(checkpoint))
                self.policy_network.load_state_dict(checkpoint_dict['policy_network'])
                self.optimiser.load_state_dict(checkpoint_dict['optimiser'])
                return True
            return False

        def _collect_trajectories(self, arena):
            arena.set_train()
            trajs = []
            for _ in range(self.config.num_collect_episodes):
                information = arena.reset()
                trj = []               
                for _ in range(self.config.collect_episode_horizon):
                    action = self.act(information)
                    new_information = arena.step(action)
                    trj.append((information['state'], action, new_information['reward']))
                    information = new_information
                    if information['done']:
                        break
                trajs.append(trj)
            return trajs