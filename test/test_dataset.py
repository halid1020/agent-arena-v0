# TODO: make this test work

import numpy as np
from agent_arena.utilities.trajectory_dataset \
    import TrajectoryDataset

# Define the shapes of your observation and action types
obs_shapes = {
    'rgb': {'shape': (64, 64, 3), 'output_key': 'rgb'},
    'depth': {'shape': (64, 64, 1), 'output_key': 'depth'},
    'particles': {'shape': (100, 3), 'output_key': 'particles'},
}

action_shapes = {
    'velocity': {'shape': (2,), 'output_key': 'velocity'},
    'torque': {'shape': (3,), 'output_key': 'torque'},
}

data_dir = 'tmp/test_dataset'

## remove the directory if it already exists
import shutil
shutil.rmtree(data_dir, ignore_errors=True)

# Create a new dataset in write mode with whole trajectory sampling
dataset = TrajectoryDataset(data_dir, io_mode='w', 
        obs_config=obs_shapes, act_config=action_shapes, 
        whole_trajectory=True)

# Add some trajectories
for _ in range(5):  # Add 5 trajectories
    traj_length = np.random.randint(50, 150)  # Random trajectory length
    traj_obs = {
        'rgb': np.random.randint(0, 256, size=(traj_length + 1, 64, 64, 3), dtype=np.uint8),
        'depth': np.random.randn(traj_length + 1, 64, 64).astype(np.float32),
        'particles': np.random.randn(traj_length + 1, 100, 3).astype(np.float32)
    }
    traj_actions = {
        'velocity': np.random.randn(traj_length, 2).astype(np.float32),
        'torque': np.random.randn(traj_length, 3).astype(np.float32)
    }
    dataset.add_trajectory(traj_obs, traj_actions)

# Print dataset info
print(f"Total trajectories: {len(dataset)}")
print(f"Total timesteps: {dataset.get_total_timesteps()}")
print(f"Trajectory lengths: {dataset.get_trajectory_lengths()}")
print(f"Observation types: {dataset.get_observation_types()}")
print(f"Action types: {dataset.get_action_types()}")

# To use the dataset for reading after writing, you need to reinitialize it in read mode
read_dataset = TrajectoryDataset(data_dir, io_mode='r', 
    whole_trajectory=True, obs_config=obs_shapes, act_config=action_shapes)

# Create a DataLoader and use the dataset
from torch.utils.data import DataLoader

dataloader = DataLoader(read_dataset, batch_size=1, shuffle=True, num_workers=4)

print()
print("Iterating through the DataLoader")
for data in dataloader:
    # obs_dict and actions_dict now contain whole trajectories
    # obs_dict['rgb'].shape: (batch_size, max_traj_length, 64, 64, 3)
    # actions_dict['velocity'].shape: (batch_size, max_traj_length - 1, 2)
    # Note: Trajectories in a batch will be padded to the length of the longest trajectory
    
    print()
    # print shapes for all fields
    print("Observation shapes:")
    for key, value in data['observation'].items():
        print(f"{key}: {value.shape}")
    print("Action shapes:")
    for key, value in data['action'].items():
        print(f"{key}: {value.shape}")
