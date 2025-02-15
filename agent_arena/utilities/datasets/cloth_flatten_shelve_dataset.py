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

### TODO: refactor the constructor
class ClothFlattenShelveDataset(Dataset):
    def __init__(self, 
        data_dir='softgym_cloth_pick_and_place', 

        eval_action_horizon=50,
        action_horizon=50,
        episode_len=50,
        sequence_len=50,

        return_depth=False, 
        return_rgb=True, 
        mode='train',
        transform=lambda x: x,
        **kwargs):

        num_episodes = kwargs['num_episodes']
        self.img_dim = kwargs['img_dim']
        
        if not os.path.exists(data_dir):
            print('there is not such data directory: {}'.format(data_dir))
        
        N = len(os.listdir(data_dir))//3

        if mode == 'train':
            self._N_start = 100
            self._N_end = min(self._N_start + num_episodes, N) - 1
        else:
            self._N_start = 0
            self._N_end = 99
        
        self._N = self._N_end - self._N_start

        self._episode_len = episode_len
        self._sequence_len = sequence_len
        self._directory = data_dir
        #self.configs = config
        self.action_horizon=action_horizon
        self.return_rgb = return_rgb
        self.return_depth = return_depth
        self.eval_action_horizon = eval_action_horizon
        self.cross_traj = kwargs['cross_traj']
        self.transform = transform
    
    def __len__(self):
        if self.cross_traj:
            return self._N * self.action_horizon - self._sequence_len + 1
        else:
            return self._N *(self.action_horizon - self._sequence_len + 1)

    
    def _load(self, episode_id):
        
        episode_id = self._N_start + episode_id
        data_file_name = os.path.join(self._directory, 'episode_{}.db'.format(episode_id))
        try:
            data = shelve.open(data_file_name)
        except:
            print('episode_id {} fail'.format(episode_id))
        return data
    
    def get_episode(self, episode_id, transform=False, train=False):
        data = self._load(episode_id)
        
        obs = []
        if self.return_rgb:
            rgb_img =  data['observations']
            
            np.stack(
                    [cv2.resize(img, self.img_dim) for img in  data['observations'].transpose(0, 2, 3, 1)]
                ).astype('float').transpose(0, 3, 1, 2)

            obs.append(rgb_img) # preprocess_observation(rgb_imgs, self.configs.bit_depth, noise=self.configs.obs_noise))
        if self.return_depth:
            depth_imgs = np.stack([cv2.resize(d, self.img_dim) for d in data['depth_images']])
            obs.append(depth_imgs.reshape(depth_imgs.shape[0], *self.img_dim, -1).transpose(0, 3, 1, 2))
        obs = np.concatenate(obs, axis=1)

        data = {
            'observation': obs,
            'action': data['actions'].reshape(-1, 4),
            'reward': data['rewards']
        }

        if transform:
            return self.transform(data, train=train)
        
        return data
    
    def __getitem__(self, idx):
        sequence_len = self._sequence_len

        observations = []
        actions = []
        rewards = []
        dones = []

        if self.cross_traj:
            episode_id = idx//self.action_horizon
            step_start_id = idx % self.action_horizon
        else:
            episode_id = idx//(self.action_horizon - sequence_len+1)
            step_start_id = idx % (self.action_horizon-sequence_len+1)

        t = 0
        while sequence_len > 0:
            
            if (not self.cross_traj) and t == 1:
                print('not suppose to cross trajctoery')
                exit(0)
            
            step_end_id = min(self.action_horizon, step_start_id + sequence_len)
            add_value = 0 if self.cross_traj and step_end_id < self.action_horizon else 1

            data = self._load(episode_id)

            obs = []
            if self.return_rgb:
                rgb_imgs = np.stack(
                    [cv2.resize(img, self.img_dim) for img in  data['observations'][step_start_id: step_end_id+add_value].transpose(0, 2, 3, 1)]
                ).astype('float').transpose(0, 3, 1, 2)

                obs.append(rgb_imgs) # preprocess_observation(rgb_imgs, self.configs.bit_depth, noise=self.configs.obs_noise))
            if self.return_depth:
                depth_imgs = np.stack([cv2.resize(d, self.img_dim) for d in data['depth_images'][step_start_id: step_end_id+add_value]])
                obs.append(np.expand_dims(depth_imgs, axis=3).transpose(0, 3, 1, 2))
            # S*C*H*W
            obs = np.concatenate(obs, axis=1)

            observations.append(obs)

            action = data['actions'][step_start_id: step_end_id]
            normal_actions = action
            actions.append(normal_actions)

            rewards.append(data['rewards'][step_start_id: step_end_id])
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

        #print('data observation shape', data['observation'].shape)
        return data