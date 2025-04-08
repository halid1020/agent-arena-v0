import numpy as np
import cv2
import agent_arena as ag_ar
from agent_arena.utilities.trajectory_dataset import TrajectoryDataset
from agent_arena.utilities.perform_single import perform_single

obs_config = {'rgb': {'shape': (128, 128, 3),  'output_key': 'rgb'}}
act_config = {'norm-pixel-pick-and-place': {'shape': (2, 2), 'output_key': 'default'}}

total_trials = 5
data_path = 'exmaple_trajectory_data'
agent_name = "oracle-garment|mask-biased-pick-and-place"
arena_name = "softgym|domain:mono-square-fabric,initial:crumpled,action:pixel-pick-and-place(1),task:flattening"

config = ag_ar.retrieve_config(agent_name, arena_name, "")
arena = ag_ar.build_arena(arena_name + ',disp:False')
agent = ag_ar.build_agent(agent_name, config)

dataset = TrajectoryDataset(
    data_path=data_path, io_mode='a', whole_trajectory=True,
    obs_config=obs_config, act_config=act_config)

while dataset.num_trajectories() < total_trials:
    res = perform_single(arena, agent, mode='train', max_steps=3)
    observations, actions = {}, {}
    for k, v in obs_config.items():
        obs_data = [info['observation'][k] for info in res['information']]
        observations[k] = [cv2.resize(obs.astype(np.float32), v['shape'][:2]) for obs in obs_data]
    for k, v in act_config.items():
        act_data = [a[k] for a in res['actions']]
        #act_data = act_data['norm-pixel-pick-and-place']
        actions[k] = [np.stack([act['pick_0'], act['place_0']]).reshape(*v['shape']) \
            for act in act_data]

    dataset.add_trajectory(observations, actions)