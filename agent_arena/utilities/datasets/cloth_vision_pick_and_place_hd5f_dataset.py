import os

import numpy as np
from torch.utils.data import Dataset
import h5py
import json

import shelve

from agent_arena.agent.utilities.utils import preprocess_observation


### TODO: fliter all the sucessful episodes on the given task

class ClothVisionPickAndPlaceHDF5Dataset(Dataset):
    def __init__(self, 
        data_dir='',  
        
        episode_len=20, ## The actual length of the episode in dataset 
        action_horizon=20, ## The horizon of the action that can be sampled, which should be smaller than episode_len
        sequence_len=50,  ## The length of the sequence that is actually sampled
        eval_action_horizon=20,

        return_depth=False, 
        return_rgb=True, 
        return_mask=True,
        cross_traj=False,
        swap_action=False,
        num_episodes=100, # The number of episodes available in the dataset
        raw_img_dim=(128, 128), # The dimension of the raw image
        mode='train', 
        # transform=lambda x: x,
        random_seed=0,
        reward_mode='normalised_coverage',
        return_pick_and_place_action_z=False,
        penalise_action_threshold=None,
        extreme_action_penalty=None,
        misgrasping_penalty=None,
        unflatten_penalty=None,
        flatten_bonus=None,
        misgrasping_threshold=None,
        unflatten_threshold=None,
        flattening_threshold=None,
        **kwargs
    ):
        self.data_dir = os.path.join(
            os.environ['AGENT_ARENA_PATH'],
            '..',
            'data',
            data_dir)

        if not os.path.exists(self.data_dir ):
            print('there is not such data directory: {}'.format(self.data_dir ))
            exit(1)

        
        self.action_horizon = action_horizon
        self.episode_len = episode_len
        self.sequence_len = sequence_len
        self.eval_action_horizon = eval_action_horizon

        self.return_depth = return_depth
        self.return_rgb = return_rgb
        self.return_pick_and_place_action_z = return_pick_and_place_action_z
        self.return_mask = return_mask
        self.swap_action = swap_action
        
        self.raw_img_dim = raw_img_dim

        self.reward_mode = reward_mode
        if self.reward_mode == 'hoque_ddpg':
            self.penalise_action_threshold = penalise_action_threshold
            self.extreme_action_penalty = extreme_action_penalty
            self.misgrasping_penalty = misgrasping_penalty
            self.unflatten_penalty = unflatten_penalty
            self.flatten_bonus = flatten_bonus
            self.misgrasping_threshold = misgrasping_threshold
            self.unflatten_threshold = unflatten_threshold
            self.flattening_threshold = flattening_threshold
        

        # self.transform = transform
        self.mode = mode
        self.cross_traj = cross_traj
        

        ## Open hdf5 files and get data length
        train_N = 0
        train_files_and_valid_episodes = []
        eval_N = 0
        eval_files_and_valid_episodes = []
        files = os.listdir(self.data_dir)
        for data_file in files:
            if not data_file.endswith('.hdf5'):
                continue
            with h5py.File(os.path.join(self.data_dir, data_file), 'r') as f:
                if ('filter_success' in kwargs):
                    episodes_ids = []



                    with open(os.path.join(self.data_dir, '{}.json'\
                                           .format(data_file.split(':')[0])), 'r') as fff:
                        policy_map = json.load(fff)['policies']
                        
                        for filter_policy in kwargs['filter_policies']:
                            if filter_policy in policy_map.keys():
                                policy_id = policy_map[filter_policy]
                                eids = np.where((f['success'][:]==True) & (f['policy'][:] == policy_id))[0].tolist()
                                if not kwargs['filter_success']:
                                    eids.extend(np.where((f['success'][:]==False) & (f['policy'][:] == policy_id))[0].tolist())
                                episodes_ids.extend(eids)
                    

                else:
                    episodes_ids = np.arange(len(f['action']))
                
                if 'eval' in data_file:

                    eval_N += len(episodes_ids)
                    eval_files_and_valid_episodes.append((data_file, episodes_ids))
                else:
                    train_N +=  len(episodes_ids)
                    train_files_and_valid_episodes.append((data_file, episodes_ids))

        train_files_and_valid_episodes = sorted(train_files_and_valid_episodes, key=lambda x: x[0])
        eval_files_and_valid_episodes = sorted(eval_files_and_valid_episodes, key=lambda x: x[0])

        
        ## Set Train and Test
        if mode == 'train':
            self.random_state = np.random.RandomState(random_seed)
            self.shuffeled_ids = np.arange(train_N) 
            self.random_state.shuffle(self.shuffeled_ids)
            self.files_and_valid_episodes = train_files_and_valid_episodes
            self._N_start = 0
            self._N_end = min(train_N, num_episodes) - 1
        
        else:
            self.files_and_valid_episodes = eval_files_and_valid_episodes
            self.shuffeled_ids = np.arange(eval_N) 
            self._N_start = 0
            self._N_end = eval_N - 1
        
        self._N = self._N_end - self._N_start + 1

        print('Number of episodes in the dataset:', self._N)


    def __len__(self):
        if self.cross_traj:
            return self._N * self.action_horizon - self.sequence_len + 1
        else:
            #print('len', self._N *(self.action_horizon - self.sequence_len  + 1))
            return self._N *(self.action_horizon - self.sequence_len  + 1)
    
    def _load(self, episode_id):
        episode_id = self.shuffeled_ids[self._N_start + episode_id]

        ### find the file that contains the episode
        file_id = 0
        while episode_id >= len(self.files_and_valid_episodes[file_id][1]):
            episode_id -= len(self.files_and_valid_episodes[file_id][1])
            file_id += 1

        res_data = {}
        file_name =  self.files_and_valid_episodes[file_id][0]
        with h5py.File(os.path.join(self.data_dir, file_name), 'r') as f:
            episode_id = self.files_and_valid_episodes[file_id][1][episode_id]
            #f['mask'][episode_id] = (f['mask'][episode_id] > 0.9)
             #idx[:self.action_horizon+1]
            res_data = {
                'action': f['action'][episode_id],
                'normalised_coverage': f['normalised_coverage'][episode_id],
                'rgb': f['rgb'][episode_id], #.transpose(0, 3, 1, 2),
                'depth': f['depth'][episode_id], #.transpose(0, 3, 1, 2),
                'mask': f['mask'][episode_id], #.transpose(0, 3, 1, 2),
                #'success': f['success'][episode_id],
                #'policy':  f['policy'][episode_id]
            }

            if 'success' in f.keys():
                res_data['success'] = f['success'][episode_id]
            
            if 'policy' in f.keys():
                res_data['policy'] = f['policy'][episode_id]

        return res_data
    
    def preprocess(self, eps_data):
        data = {}

        # Action
        data['action'] = eps_data['action']
        if not self.return_pick_and_place_action_z:
            data['action'] = data['action'][:self.action_horizon]\
                .reshape(self.action_horizon, 2, -1)[:, :, :2]\
                .reshape(self.action_horizon, -1)
        
        # Observation

        if self.return_rgb:
            data['rgb'] = eps_data['rgb'].transpose(0, 3, 1, 2)
        
        if self.return_depth:
            data['depth'] = eps_data['depth'].transpose(0, 3, 1, 2)

        # Mask
        if self.return_mask:
            data['mask'] = eps_data['mask'].transpose(0, 3, 1, 2)
        #     ### put mask if depth between 1.4 and 1.51
        #     data['mask'] = (data['depth'] > 1.4 ) & (data['depth'] < 1.499)

            ### put the unmask part as 1.5
            #data['depth'][~data['mask']] = 1.5

        
        # Reward
        if self.reward_mode == "hoque_ddpg":
            data['reward'] = (eps_data['normalised_coverage'][1:] - eps_data['normalised_coverage'][:-1]).flatten()
            
            ### Misgrasping penalty
            indices = np.where((np.abs(data['reward']) < 1e-4)
                               & (eps_data['normalised_coverage'][1:].flatten() < self.misgrasping_threshold))
            data['reward'][indices[0]] = self.misgrasping_penalty


            ### Extreme Action Penalty
            #print('self.penalise_action_threshold: ', self.penalise_action_threshold)
            indices = np.where(np.max(np.abs(data['action']), axis=1) > self.penalise_action_threshold)
            data['reward'][indices[0]] = self.extreme_action_penalty

            ### Unflatten Penality
            indices = np.where(
                (eps_data['normalised_coverage'][:-1].flatten() > self.unflatten_threshold) & 
                (eps_data['normalised_coverage'][1:].flatten() < self.unflatten_threshold))
            data['reward'][indices[0]] = self.unflatten_penalty


            ### Bonus for success
            indices = np.where(eps_data['normalised_coverage'][1:].flatten() > self.flattening_threshold)
            data['reward'][indices[0]] = self.flatten_bonus

            ## Find the idx of the entries where it is smaller than 1e-4 then set them to -0.05


        elif self.reward_mode == 'normalised_coverage':    
            data['reward'] = eps_data['normalised_coverage'][1:].flatten()
        else:
            raise NotImplementedError
        
        return data
    
    def get_trajectory(self, episode_id, train=False):
        eps_data = self._load(episode_id)
        
        data = self.preprocess(eps_data)

        # if transform:
        #     return self.transform(data, train=train)
        
        return data
    
    def __getitem__(self, idx):

        sequence_len = self.sequence_len 
        # C = 3 if self.return_rgb else 0
        # if self.return_depth:
        #     C += 1
        
        if self.return_rgb:
            rgb_imgs = np.zeros((sequence_len+1, 3, *self.raw_img_dim), dtype=np.float32)
        
        if self.return_depth:
            depth_imgs = np.zeros((sequence_len+1, 1, *self.raw_img_dim), dtype=np.float32)
        
        actions = np.zeros((sequence_len,  4), dtype=np.float32)
        rewards = np.zeros((sequence_len), dtype=np.float32)
        dones = np.zeros((sequence_len), dtype=np.float32)
        if self.return_mask:
            masks = np.zeros((sequence_len+1, 1, *self.raw_img_dim), dtype=bool)


        if self.cross_traj:
            episode_id = idx//self.action_horizon
            step_start_id = idx % self.action_horizon
        else:
            episode_id = idx//(self.action_horizon-sequence_len+1)
            step_start_id = idx % (self.action_horizon-sequence_len+1)

        t = 0
        fill_start_step = 0
        fill_end_step = 0
        while sequence_len > 0:

            if (not self.cross_traj) and t > 0:
                print('not suppose to cross trajctoery')
                exit(0)

            step_end_id = min(self.action_horizon, step_start_id + sequence_len)
            add_value = 0 if self.cross_traj and step_end_id < self.action_horizon else 1

            data = self._load(episode_id)
            data = self.preprocess(data)

            fill_end_step = fill_start_step + (step_end_id - step_start_id)
            
            if self.return_rgb:
                rgb_imgs[fill_start_step: fill_end_step+add_value] = \
                    data['rgb'][step_start_id: step_end_id+add_value].astype('float')
            
            if self.return_depth:
                depth_imgs[fill_start_step: fill_end_step+add_value] = \
                    data['depth'][step_start_id: step_end_id+add_value].astype('float')
                
            ## add mask if needed
            if self.return_mask:
                masks[fill_start_step: fill_end_step+add_value] \
                    = data['mask'][step_start_id: step_end_id+add_value]

            action = data['action'][step_start_id: step_end_id].reshape(step_end_id-step_start_id, -1)
            normal_actions = action
            actions[fill_start_step: fill_end_step] = normal_actions

            rewards[fill_start_step: fill_end_step] \
                = data['reward'][step_start_id: step_end_id].reshape(step_end_id-step_start_id)
            
            done = np.zeros_like(data['reward'][step_start_id: step_end_id])
            done[-1] = 1
            dones[fill_start_step: fill_end_step] = done
            
            sequence_len -= (step_end_id - step_start_id)
            step_start_id = 0
            episode_id += 1
            t += 1
            fill_start_step = fill_end_step

        if self.swap_action:
            actions = actions[:, [1, 0, 3, 2]]
        
        data = {
            'action': actions,
            'reward': rewards,
            'terminal': dones
        }

        if self.return_mask:
            data['mask'] = masks
        
        if self.return_rgb:
            data['rgb'] = rgb_imgs

        if self.return_depth:
            data['depth'] = depth_imgs

        # data = self.transform(data, train=True)

        return data