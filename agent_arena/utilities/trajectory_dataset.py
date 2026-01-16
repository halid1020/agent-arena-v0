import torch
from torch.utils.data import Dataset
import zarr
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
import shutil

class TrajectoryDataset(Dataset):
    
    def __init__(self, data_path: str, seq_length: Optional[int] = None, cross_trajectory: bool = False, 
                 io_mode: str = 'r', obs_config: Dict[str, Tuple] = None, act_config: Dict[str, Tuple] = None, 
                 goal_config: Dict[str, Tuple] = None, save_goal=False,
                 whole_trajectory: bool = False, sample_mode='all', 
                 sample_terminal=True, split_ratios=[0.1, 0.1, 0.8],
                 num_trj=None, data_dir: Optional[str]=None,
                 transform=None, cache_in_memory: bool = False): # <--- Added flag here
        """
        Args:

            cache_in_memory (bool): If True, loads the entire dataset into RAM (numpy arrays) 
                                        to speed up training (avoids disk I/O per batch).

            zarr_path (str): Path to the Zarr directory.

            seq_length (Optional[int]): The fixed sequence length to sample. Set to None if whole_trajectory is True.

            cross_trajectory (bool): If True, allow sampling across trajectory boundaries. Ignored if whole_trajectory is True.

            mode (str): 'r' for read-only, 'a' for read/write append mode, 'w' for write mode (create new or overwrite).

            obs_shapes (Dict[str, Tuple]): Dictionary of observation shapes for each observation type (required for 'w' mode).

            action_shapes (Dict[str, Tuple]): Dictionary of action shapes for each action type (required for 'w' mode).

            whole_trajectory (bool): If True, sample whole trajectories instead of fixed-length sequences.

            splot_raitos (list of floats): Eval, val, and train.

        """

        self.whole_trajectory = whole_trajectory
        self.sample_mode = sample_mode
        self.sample_terminal = sample_terminal
        self.split_ratios = split_ratios
        self.save_goal = save_goal
        self.transform = transform
        self.num_trj = num_trj
        self.cache_in_memory = cache_in_memory # <--- Store flag
        
        if whole_trajectory:
            self.seq_length = None
            self.cross_trajectory = False
        else:
            self.seq_length = seq_length
            self.cross_trajectory = cross_trajectory
        
        if io_mode not in ['r', 'a', 'w']:
            raise ValueError("Mode must be 'r', 'a', or 'w'.")
        
        self.mode = io_mode
        
        if data_dir is None:
            # Fallback for data_path resolution
            if 'AGENT_ARENA_PATH' in os.environ:
                 base_path = os.path.join(os.environ['AGENT_ARENA_PATH'], '..', 'data')
            else:
                 base_path = './data' # default fallback
            
            data_path = os.path.join(base_path, data_path)
        else:
            data_path = os.path.join(data_dir, data_path)
        print('[agent-arena, TrajectoryDatset] data_path', data_path)

        if io_mode == 'w':
            if obs_config is None or act_config is None:
                raise ValueError("obs_shapes and action_shapes must be provided when using write mode.")
            if os.path.exists(data_path):
                shutil.rmtree(data_path)
        
        # Open the Zarr store
        if io_mode == 'r':
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"The file {data_path} does not exist.")
            if data_path.endswith('.zip'):
                self.store = zarr.ZipStore(data_path)
            else:
                self.store = zarr.DirectoryStore(data_path)
        else:
            self.store = zarr.DirectoryStore(data_path)

        self.root = zarr.open(self.store, mode=io_mode) # Fixed: pass store to zarr.open

        # Initialize or load arrays (Zarr handles)
        if io_mode in ['a', 'w']:
            self.observation = self.root.require_group('observation')
            self.action = self.root.require_group('action')
            
            for obs_type, obs_info in obs_config.items():
                shape = tuple(obs_info['shape'])
                if obs_type not in self.observation:
                    self.observation.create_dataset(obs_type, shape=(0,) + shape, dtype=np.float32, chunks=(1000,) + shape, elastic=True)
            for action_type, act_info in act_config.items():
                shape = tuple(act_info['shape'])
                if action_type not in self.action:
                    self.action.create_dataset(action_type, shape=(0,) + shape, dtype=np.float32, chunks=(1000,) + shape, elastic=True)
            if save_goal:
                self.goal = self.root.require_group('goal')
                for goal_type, goal_info in goal_config.items():
                    shape = tuple(goal_info['shape'])
                    if goal_type not in self.goal:
                        self.goal.create_dataset(goal_type, shape=(0,) + shape, dtype=np.float32, chunks=(1000,) + shape, elastic=True)
            
            if 'trajectory_lengths' not in self.root:
                self.trajectory_lengths = self.root.require_dataset('trajectory_lengths', shape=(0,), dtype=np.int64, chunks=(1000,), elastic=True)
            else:
                self.trajectory_lengths = self.root['trajectory_lengths']
        else:
            self.observation = self.root['observation']
            self.action = self.root['action']
            self.trajectory_lengths = self.root['trajectory_lengths'][:]
            if save_goal and 'goal' in self.root:
                self.goal = self.root['goal']
            else:
                self.goal = None

        self.obs_config = obs_config
        self.act_config = act_config
        
        # In read mode, infer config from Zarr if not provided
        if self.obs_config is None and io_mode == 'r':
             # Simple inference, might need adjustment based on your config structure
             self.obs_types = list(self.observation.keys())
             # Warning: this assumes standard shape/key structure
        else:
            self.obs_types = list(self.obs_config.keys())
            self.obs_shapes = {k: v['shape'] for k, v in self.obs_config.items()}
            self.obs_output_types = [v['output_key'] for k, v in self.obs_config.items()]

        self.act_shapes = {k: v['shape'] for k, v in act_config.items()}
        self.action_types = list(act_config.keys())
        self.action_output_types = [v['output_key'] for k, v in act_config.items()]

        if save_goal:
            self.goal_config = goal_config
            self.goal_types = list(self.goal_config.keys())
            self.goal_output_types = [v['output_key'] for k, v in self.goal_config.items()]
        
        self.update_dataset_info()
        
        # --- CACHING LOGIC START ---
        # By default, sources point to Zarr groups
        self.obs_source = self.observation
        self.act_source = self.action
        if save_goal:
            self.goal_source = self.goal

        if self.cache_in_memory:
            if self.mode != 'r':
                print("[agent-arena, TrajectoryDatset]  Warning: cache_in_memory is strictly recommended for 'r' mode. " 
                      "[agent-arena, TrajectoryDatset]  Writing new data will not update the cache automatically.")
            
            print("[agent-arena, TrajectoryDatset] Caching dataset into memory... (This may take a moment)")
            
            # Cache Observations
            self.obs_source = {}
            for k in self.obs_types:
                # [:] forces load from zarr to numpy RAM
                self.obs_source[k] = self.observation[k][:] 
            
            # Cache Actions
            self.act_source = {}
            for k in self.action_types:
                self.act_source[k] = self.action[k][:]
                
            # Cache Goals
            if self.save_goal and self.goal is not None:
                self.goal_source = {}
                for k in self.goal_types:
                    self.goal_source[k] = self.goal[k][:]
                    
            print(f"Finished caching. Loaded {self.total_timesteps} timesteps.")
        # --- CACHING LOGIC END ---

        print('[agent-arena, TrajectoryDatset]  total trj', self.total_trj)
        print('[agent-arena, TrajectoryDatset]  num_samples', self.num_samples)

    def update_dataset_info(self):
        """Update dataset information after adding new data."""
        if self.num_trj is None:
            self.traj_lengths = self.trajectory_lengths[:]
        else:
            self.traj_lengths = self.trajectory_lengths[:self.num_trj]
        self.total_trj = len(self.traj_lengths)
        self.traj_starts = np.concatenate(([0], np.cumsum(self.traj_lengths)[:-1])) if len(self.traj_lengths) > 0 else np.array([])
        self.total_timesteps = np.sum(self.traj_lengths)
        
        # self.terminals is always created in memory (numpy), so no extra caching needed
        self.terminals = np.zeros((self.total_timesteps, 1))
        for i in range(len(self.traj_lengths)):
            self.terminals[self.traj_starts[i] + self.traj_lengths[i] - 1] = 1

        if self.whole_trajectory:
            self.all_samples = len(self.traj_lengths)
        elif self.cross_trajectory:
            self.all_samples = max(0, self.total_timesteps - self.seq_length)          
        else:
            self.valid_ranges = [
                (start, start + length - (self.seq_length+1))
                for start, length in zip(self.traj_starts, self.traj_lengths)
                if length >= self.seq_length + 1
            ]
            self.flat_ranges = [
                (traj_idx, start_idx)
                for traj_idx, (traj_start, traj_end) in enumerate(self.valid_ranges)
                for start_idx in range(traj_start, traj_end + 1)
            ]
        
            self.all_samples = len(self.flat_ranges)

        if self.sample_mode == 'all':
            self.num_samples = self.all_samples
            self.start_sample = 0
            self.end_sample = self.num_samples
        elif self.sample_mode == 'eval':
            self.num_samples = int(self.split_ratios[0] * self.all_samples)
            self.start_sample = 0
            self.end_sample = self.num_samples
        elif self.sample_mode == 'val':
            self.num_samples = int(self.split_ratios[1] * self.all_samples)
            self.start_sample = int(self.split_ratios[0] * self.all_samples)
            self.end_sample = self.start_sample + self.num_samples
        elif self.sample_mode == 'train':
            self.num_samples = self.all_samples - int(np.sum(self.split_ratios[:-1]) * self.all_samples)
            self.start_sample = int(np.sum(self.split_ratios[:-1]) * self.all_samples)
            self.end_sample = self.all_samples

    def add_transition(self, observation: Dict[str, np.ndarray], action: Dict[str, np.ndarray], done: bool):
        # NOTE: This writes to ZARR (disk). If cache_in_memory is True, the cache is NOT updated here.
        if self.mode not in ['a', 'w']:
            raise ValueError("[agent-arena, TrajectoryDatset]  Dataset not opened in append or write mode.")

        for obs_type, obs_data in observation.items():
            self.observation[obs_type].append(obs_data[np.newaxis])

        for action_type, action_data in action.items():
            self.action[action_type].append(action_data[np.newaxis])

        if done:
            if len(self.trajectory_lengths) == 0:
                new_length = len(self.action[list(self.action.keys())[0]])
            else:
                new_length = len(self.action[list(self.action.keys())[0]]) - np.sum(self.traj_lengths)
            self.trajectory_lengths.append(np.array([new_length]))

        self.update_dataset_info()

    def add_trajectory(self, observations: Dict[str, np.ndarray], actions: Dict[str, np.ndarray], goals=None):
        if self.mode not in ['a', 'w']:
            raise ValueError("[agent-arena, TrajectoryDatset]  Dataset not opened in append or write mode.")

        for obs_type, obs_data in observations.items():
            if obs_type not in self.obs_config.keys():
                continue
            
            if len(obs_data) != len(list(actions.values())[0]) + 1:
                raise ValueError(f"[agent-arena, TrajectoryDatset]  Number of {obs_type} observations should be one more than the number of actions.")
            
            if isinstance(obs_data, list):
                obs_data = np.array(obs_data)
            obs_data_ = obs_data.reshape(-1, *self.obs_config[obs_type]['shape'])
            self.observation[obs_type].append(obs_data_)

        for action_type, action_data in actions.items():
            action_data = np.concatenate([action_data, np.zeros_like(action_data[:1])])
            self.action[action_type].append(action_data)
        
        if self.save_goal:
            for goal_type, goal_data in goals.items():
                if goal_type not in self.goal_config.keys():
                    continue
                if len(goal_data) != 1:
                    raise ValueError(f"Number of {goal_type} goals should be one.")
                if isinstance(goal_data, list):
                    goal_data = np.array(goal_data)
                goal_data_ = goal_data.reshape(-1, *self.goal_config[goal_type]['shape'])
                self.goal[goal_type].append(goal_data_)

        self.trajectory_lengths.append(np.array([len(list(observations.values())[0])]))
        self.update_dataset_info()

    def get_trajectory(self, idx: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        start_idx = self.traj_starts[idx]
        end_idx = start_idx + self.traj_lengths[idx] - 1
        
        # Use self.obs_source / self.act_source (can be Zarr or Dict of Numpy)
        obs = {
            obs_output_type: self.obs_source[obs_type][start_idx:end_idx+1]\
                .reshape(-1, *self.obs_config[obs_type]['shape']) \
                for obs_type, obs_output_type in zip(self.obs_types, self.obs_output_types)
        }
        if self.sample_terminal:
            obs['terminal'] = self.terminals[start_idx:end_idx+1].reshape(-1, 1)

        actions = {
            act_output_type: self.act_source[action_type][start_idx:end_idx]\
                .reshape(-1, *self.act_config[action_type]['shape']) \
                for action_type, act_output_type in zip(self.action_types, self.action_output_types)      
        }

        ret = {
            'observation': obs,
            'action': actions,
            'action_steps': end_idx - start_idx
        }

        if self.save_goal:
            goals = {
                goal_output_type: self.goal_source[goal_type][idx].reshape(*self.goal_config[goal_type]['shape']) \
                    for goal_type, goal_output_type in zip(self.goal_types, self.goal_output_types)
            }
            ret['goal'] = goals
        return ret
    
    def set_transform(self, transform):
        self.transform = transform
    
    def num_trajectories(self) -> int:
        return len(self.traj_lengths)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        idx = idx + self.start_sample
        if idx >= self.end_sample:
            raise IndexError(f"[agent-arena, TrajectoryDatset]  Index {idx-self.start_sample} out of range for sampling of length {self.num_samples}.")

        if self.whole_trajectory:
            start_idx = self.traj_starts[idx]
            end_idx = start_idx + self.traj_lengths[idx] - 1
        elif self.cross_trajectory:
            start_idx = idx
            end_idx = start_idx + self.seq_length
        else:
            traj_idx, start_idx = self.flat_ranges[idx]
            end_idx = start_idx + self.seq_length

        # Use abstract sources here (self.obs_source, self.act_source)
        obs = {
            obs_output_type: self.obs_source[obs_type][start_idx:end_idx+1]\
                .reshape(-1, *self.obs_shapes[obs_type])
            for obs_type, obs_output_type in zip(self.obs_types, self.obs_output_types)
        }

        if self.sample_terminal:
            obs['terminal'] = self.terminals[start_idx:end_idx+1].reshape(-1, 1)

        actions = {
            act_output_type: self.act_source[action_type][start_idx:end_idx].reshape(-1, *self.act_shapes[action_type])
            for action_type, act_output_type in zip(self.action_types, self.action_output_types)      
        }

        ret = {
            'observation': obs,
            'action': actions
        }

        if self.save_goal:
            goals = {
                goal_output_type: self.goal_source[goal_type][traj_idx].reshape(-1, *self.goal_shapes[goal_type])
                for goal_type, goal_output_type in zip(self.goal_config.keys(), self.goal_output_types)
            }
            ret['goal'] = goals
        
        if self.transform is not None:
            ret_ = self.transform(ret)
            for key in ret['observation'].keys():
                if key not in ret_:
                    ret_[key] = ret['observation'][key]
            return ret_

        return ret

    # ... (rest of methods like get_trajectory_lengths, etc. remain unchanged)
    def get_trajectory_lengths(self) -> List[int]:
        return self.traj_lengths.tolist()

    def get_total_timesteps(self) -> int:
        return self.total_timesteps

    def is_cross_trajectory(self) -> bool:
        return self.cross_trajectory

    def is_whole_trajectory(self) -> bool:
        return self.whole_trajectory

    def get_observation_types(self) -> List[str]:
        return self.obs_types

    def get_action_types(self) -> List[str]:
        return self.action_types