import os

import math
import numpy as np
import random

import cv2
import imutils
from torch.utils.data import Dataset
import h5py
import shelve

from agent_arena.agent.utilities.utils import preprocess_observation

class ClothHDF5DatasetV2(Dataset):
    def __init__(self, 
        data_dir='',  
        
        eval_action_horizon=10, ## The horizon of the action that is sampled in evaluation
        episode_len=10, ## The actual length of the episode in dataset 
        action_horizon=10, ## The horizon of the action that can be sampled, which should be smaller than episode_len
        sequence_len=10,  ## The length of the sequence that is actually sampled

        return_depth=False, 
        return_rgb=True, 
        return_control=False, 
        action_repeat=1,
        cross_traj=False,
        num_episodes=100, # The number of episodes available in the dataset
        eval_episodes=100, # The number of episodes used for evaluation in the dataset
        img_dim=(64, 64), # The dimension of the image
        mode='train', 
        transform=lambda x: x,
        ):
        
        if not os.path.exists(data_dir):
            print('there is not such data directory: {}'.format(data_dir))
            exit(1)


        self.data_dir = data_dir
        
        self.action_horizon = action_horizon
        self.eval_action_horizon = eval_action_horizon
        self.episode_len = episode_len
        self.sequence_len = sequence_len

        self.return_depth = return_depth
        self.return_rgb = return_rgb
        self.return_control = return_control
        self.action_repeat = action_repeat
        

        self.transform = transform
        self.mode = mode
        self.cross_traj = cross_traj
        
        
        

        ## Open hdf5 file and get data length
        with h5py.File(self.data_dir, 'r') as f:
        
            self.N = min(len(f['action']), num_episodes)
            print('Number of available episodes in dataset: {}'.format(self.N))

            # ### If Pick and Place
            # if not return_control:
            #     min_last_step_action_horizons = 1 + min(f['action'][:, -1])
            #     if self.action_horizon > min_last_step_action_horizons:
            #         print('sampling action horizon {} is too large, suggest setting to: {}'\
            #             .format(self.action_horizon, min_last_step_action_horizons-1))
            #         exit(1)
        
        ## Set Train and Test
        self.random_state = np.random.RandomState(0)
        self.shuffeled_ids = np.arange(self.N) 
        self.random_state.shuffle(self.shuffeled_ids)
        

        if mode == 'train':
            self._N_start = eval_episodes
            self._N_end = self.N-1
        else:
            self._N_start = 0
            self._N_end = eval_episodes-1
        
        self._N = self._N_end - self._N_start


    def __len__(self):
        if self.cross_traj:
            return self._N * self.action_horizon - self.sequence_len + 1
        else:
            return self._N *(self.action_horizon - self.sequence_len + 1)
    
    def _load(self, episode_id):
        episode_id = self.shuffeled_ids[self._N_start + episode_id]
        res_data = {}
        with h5py.File(self.data_dir, 'r') as f:

            if not self.return_control:
                #_, idx = np.unique(f['pick_and_place_action_step'][episode_id], return_index=True)
                idx = range(self.action_horizon+1) #idx[:self.action_horizon+1]

                res_data = {
                    'action': f['action'][episode_id][idx[:self.action_horizon]],
                    'reward': f['reward'][episode_id][idx[:self.action_horizon]],
                    'rgb': f['rgb'][episode_id][idx[:self.action_horizon+1]].transpose(0, 3, 1, 2)
                }
                
            else:
                res_data = {
                    'action': f['control_signal'][episode_id][:self.action_horizon],
                    'reward': f['reward'][episode_id][:self.action_horizon],
                    'rgb': f['rgb'][episode_id][:self.action_horizon+1].transpose(0, 3, 1, 2)
                }


        return res_data
    
    def get_episode(self, episode_id, transform=False, train=False):
        data = self._load(episode_id)
        
        ## TODO: add depth
        data = {
            'observation': data['rgb'],
            'action': data['action'].reshape(len(data['action']), -1),
            'reward': data['reward'].reshape(data['reward'].shape[0])
        }

        if transform:
            return self.transform(data, train=train)
        
        return data
    
    def __getitem__(self, idx):
        sequence_len = self.sequence_len 

        observations = []
        actions = []
        rewards = []
        dones = []

        if self.cross_traj:
            episode_id = idx//self.action_horizon
            step_start_id = idx % self.action_horizon
        else:
            episode_id = idx//(self.action_horizon-sequence_len+1)
            step_start_id = idx % (self.action_horizon-sequence_len+1)

        t = 0
        while sequence_len > 0:

            if (not self.cross_traj) and t > 0:
                print('not suppose to cross trajctoery')
                exit(0)

            step_end_id = min(self.action_horizon, step_start_id + sequence_len)
            add_value = 0 if self.cross_traj else 1

            data = self._load(episode_id)

            #TODO: add depth
            raw_observations = data['rgb'][step_start_id: step_end_id+add_value].astype('float')
            observations.append(raw_observations)

            action = data['action'][step_start_id: step_end_id].reshape(step_end_id-step_start_id, -1)
            normal_actions = action
            actions.append(normal_actions)

            rewards.append(data['reward'][step_start_id: step_end_id].reshape(step_end_id-step_start_id))
            
            done = np.zeros_like(rewards[-1])
            done[-1] = 1
            dones.append(done)
            
            sequence_len -= (step_end_id - step_start_id)
            step_start_id = 0
            episode_id += 1
            t += 1
        
        data = {
            'observation': np.concatenate(observations),
            'action': np.concatenate(actions),
            'reward': np.concatenate(rewards),
            'terminal': np.concatenate(dones),
        }

        data = self.transform(data, train=True)

        return data