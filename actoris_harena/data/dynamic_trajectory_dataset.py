"""
Dynamic Trajectory Dataset
==========================
PyTorch Dataset for dynamically synchronizing and loading multi-modal trajectory data.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import bisect
import os
import json

## TODO: If the current state is at time step t, I want to have functinoality to sample each data type between 
# an interval. For example, [-3, 2] for camera means that it should give the camera view for t-3, t-2, t-1, t, t+1, t+2. 
# if the signal is not given for a certain position, for now please just give 0 filled values.

class DynamicTrajectoryDataset(Dataset):
    """
    Loads asynchronous streams, synchronizes them to a target frequency, 
    and decodes streams dynamically based on naming conventions.
    """
    def __init__(self, dataset_dir: str, target_freq_hz: float, action_horizon: int = 4, selected_streams: list = None):
        self.dataset_dir = dataset_dir
        self.target_freq_hz = target_freq_hz
        self.action_horizon = action_horizon
        
        self.metadata_path = os.path.join(self.dataset_dir, "metadata.json")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"No metadata found in {self.dataset_dir}")
            
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.streams_to_load = selected_streams or self.metadata["streams"]
        self.episodes = [] 
        self.cumulative_lengths = []
        
        total_valid_steps = 0
        for rec_id, rec_info in self.metadata["recordings"].items():
            file_path = os.path.join(self.dataset_dir, rec_info["file"])
            episode_data = self._load_and_sync_episode(file_path)
            
            if episode_data is not None:
                ep_length = len(episode_data['master_timestamps'])
                valid_steps = max(0, ep_length - self.action_horizon)
                if valid_steps > 0:
                    self.episodes.append(episode_data)
                    total_valid_steps += valid_steps
                    self.cumulative_lengths.append(total_valid_steps)

    def _load_and_sync_episode(self, file_path: str):
        data = np.load(file_path, allow_pickle=True)
        start_times, end_times = [], []
        
        for stream in self.streams_to_load:
            rel_ts_key = f"{stream}_rel_ts"
            if rel_ts_key in data and len(data[rel_ts_key]) > 0:
                start_times.append(data[rel_ts_key][0])
                end_times.append(data[rel_ts_key][-1])
                
        if not start_times: return None
            
        sync_start, sync_end = max(start_times), min(end_times)
        master_ts = np.arange(sync_start, sync_end, 1.0 / self.target_freq_hz)
        synced_episode = {'master_timestamps': master_ts}
        
        for stream in self.streams_to_load:
            ts, vals = data[f"{stream}_rel_ts"], data[f"{stream}_values"]
            synced_episode[stream] = [self._get_nearest(ts, vals, q_ts) for q_ts in master_ts]
            
        return synced_episode

    def _get_nearest(self, timestamps, values, query_ts):
        idx = bisect.bisect_left(timestamps, query_ts)
        if idx == 0: return values[0]
        if idx == len(timestamps): return values[-1]
        if (query_ts - timestamps[idx - 1]) <= (timestamps[idx] - query_ts):
            return values[idx - 1]
        return values[idx]

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        ep_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        local_idx = idx if ep_idx == 0 else idx - self.cumulative_lengths[ep_idx - 1]
        episode = self.episodes[ep_idx]
        
        sample = {'timestamp': episode['master_timestamps'][local_idx]}
        
        # Dynamically process streams based on naming conventions to decouple hardcoded logic
        for stream in self.streams_to_load:
            if 'action' in stream.lower():
                chunk = episode[stream][local_idx : local_idx + self.action_horizon]
                sample[stream] = torch.tensor(chunk, dtype=torch.float32)
            elif 'camera' in stream.lower() or 'rgb' in stream.lower():
                raw_bytes = episode[stream][local_idx]
                img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                sample[stream] = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 127.5 - 1.0
            else:
                # Catch-all for joint states, tactile sensors, etc.
                sample[stream] = torch.tensor(episode[stream][local_idx], dtype=torch.float32)
                
        return sample