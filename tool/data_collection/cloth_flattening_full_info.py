import sys
sys.path.insert(0, '../..')

import numpy as np
import h5py
from tqdm import tqdm

from src.environments.softgym_cloth_flatten_env \
    import  SoftGymClothFlattenEnv
from cloth_folding_IRL.src.policies.pick_and_place_rect_fabric_flattening_policies import *
from cloth_folding_IRL.src.policies.base_policies import *


# 0: random, 1: expert, 2: noisy_expert, 3: corner-biased, 4: mix
num_trial_types = 5
trials_ratio = [0.05, 0.2, 0.05, 0.2, 0.5]
#trials_ratio = [0, 0, 0.05, 0.2, 0.5]
create_policy = {
    0: (RandomPolicy, \
        {'action_dim': (1, 4)}),
    1: (PickAndPlaceExpertPolicy, \
        {'action_dim': (1, 4)}),
    2: (PickAndPlaceNoisyExpertPolicy, \
        {'action_dim': (1, 4)}),
    3: (PickAndPlaceCornerBiasedPolicy, \
         {'action_dim': (1, 4), 'pick_noise': 0.05, 'place_noise': 0.05, 'drag_noise': 0.05}),
    4: (PickAndPlaceMixPolicy, \
        {
            'action_dim': (1, 4), 
            'pick_noise': 0.05, 'place_noise': 0.05, 'drag_noise': 0.05,
            'policy_orders': [
                'expert', 'expert', 'expert', 'expert', 
                'expert', 'corner-biased', 'random', 
                'corner-biased', 'noisy-expert', 'noisy-expert']})
}


# 0: random, 1: corner-biased, 2-7: flattening, 8: folding
# 9-13: noisy-flattening, 14: noisy-folding
action_phase_to_scalar = {
    'no_op': -1,

    'random': 0,
    'corner-biased-random': 1,

    # Flattening Phases
    'no-valid-corners': 2,
    'some-corners-out-of-boundary': 3,
    'in-boundary-yet-multiple-hidden-corners': 4,
    'untwist': 5,
    'reveal-hidden-corner': 6, 
    'to-flatten': 7,

     # Flattening Phases
    'noisy-no-valid-corners': 8,
    'noisy-some-corners-out-of-boundary': 9,
    'noisy-in-boundary-yet-multiple-hidden-corners': 10,
    'noisy-untwist': 11,
    'noisy-reveal-hidden-corner': 12, 
    'noisy-to-flatten': 13
}

flush_every = 100

def main():
    
    ### Data collection parameters
    num_episodes = int(5e3)
    seed = 0
    h5py_file = 'softgym_cloth_flattening_without_particle_info:seed_{}_num_episodes_{}.hdf5'.format(seed, num_episodes)
    #h5py_file = 'example.hdf5'.format(seed, num_episodes)
    num_pickers = 1
    H, W = 64, 64
    rgb_dim = (H, W, 3) # RGBD
    depth_dim = (H, W)
    #particle_pos_dim = (64*64, 4) # x, z, y, inverse masss, corners are (0, 63, 64*63, 64*64-1)
    picker_pos_dim = (num_pickers, 3) # 0-2: picker positions
    picker_pos_dim = (num_pickers, 3) # 0-2: picker positions
    pick_and_place_dim = (num_pickers, 4) # 0-1: pick, 2-3: place
    control_signal_dim = (num_pickers, 4)  # 0-2: control for displacement, 3: control for grip/release

    pick_and_place_action_steps = 40
    control_steps = 2000


    #### Intialise Environment
    env_para =  {
        "headless": True,  # Shows GUI

        'initial_state': 'crumple',
        'use_cached_states': True,
        'save_cached_states': False,
        'cached_states_path': 'cloth_crumple.pkl',
        'num_variations': 1000,
        'random_seed': seed,
        
        'observation_image_shape': (H, W, 4),
        'reward_mode': 'hoque_ddpg', # Reward Option
        'action_horizon': pick_and_place_action_steps,
        'control_horizon': control_steps,
        'action_mode': 'pickerpickplace',
        'picker_low': [-1, -1, -1, -1],
        'pick_high': [1, 1, 1, 1],

        'motion_trajectory': 'triangle',
        'pick_height': 0.026,
        'place_height': 0.06,
        'intermidiate_height': 0.15,
        'num_pickers': num_pickers,

        'save_step_info': True,
        'save_image_dim': (H, W)

    }
    print('initialise_environement')
    environment = SoftGymClothFlattenEnv(env_para)
    environment.set_train()
    

    with h5py.File(h5py_file,'w') as f:

        # observations
        rgb = f.create_dataset('rgb', (num_episodes, control_steps+1, *rgb_dim), dtype='uint8')
        depth = f.create_dataset('depth', (num_episodes, control_steps+1, *depth_dim), dtype='float16')
        #particle_pos =  f.create_dataset('particle_pos', (num_episodes, control_steps+1, *particle_pos_dim), dtype='float16')
        picker_pos = \
            f.create_dataset('picker_pos', (num_episodes, control_steps+1, *picker_pos_dim), dtype='float16')
        normalised_coverage = f.create_dataset('evaluaion/normalised_coverage', (num_episodes, control_steps+1, 1), dtype='float16')
        normalised_improvement = f.create_dataset('evaluaion/normalised_improvement', (num_episodes, control_steps+1, 1), dtype='float16')


        # Actions
        control_signal = f.create_dataset('control_signal', \
            (num_episodes, control_steps, *control_signal_dim), dtype='float16')

        pick_and_place_actions = f.create_dataset('pick_and_place_action', \
            (num_episodes, control_steps, *pick_and_place_dim), dtype='float16')
        
        pick_and_place_action_steps = f.create_dataset('pick_and_place_action_step', \
            (num_episodes, control_steps, 1), dtype='uint8')

        phases = f.create_dataset('phase', (num_episodes, control_steps, 1), dtype='int8')
        
        # Performance
        rewards = f.create_dataset('reward', (num_episodes,  control_steps, 1), dtype='float16')
        
        
        # Trajectory Info
        flatten_coverage = f.create_dataset('flatten_coverage', (num_episodes, 1), dtype='float16')
        trajectory_types = f.create_dataset('trajectory_type', (num_episodes, 1), dtype='uint8')  # 0 represent expert trajectory

        e = 0
        for trial_type, ratio in enumerate(trials_ratio):
            num_eps = int(ratio*num_episodes)
            
            for _ in tqdm(range(num_eps)):
                policy = create_policy[trial_type][0](**create_policy[trial_type][1])
                data = perform(environment, policy)

            
                rgb[e:e+1] = data['rgbd'][:control_steps+1, :, :, :3].astype(np.uint8)
                depth[e:e+1] = data['rgbd'][:control_steps+1, :, :, 3].astype(np.float16)
                #particle_pos[e:e+1] = data['particle_pos'][:control_steps+1].astype(np.float16)
                picker_pos[e:e+1] = data['picker_pos'][:control_steps+1, :, :3].astype(np.float16)
                


                pick_and_place_actions[e:e+1] = data['pick_and_place_action'][:control_steps].astype(np.float16)
                pick_and_place_action_steps[e:e+1] = data['pick_and_place_action_step'][:control_steps].astype(np.uint8)

                control_signal[e:e+1] = data['control_signal'][:control_steps].astype(np.float16)
                phases[e:e+1] = data['phase'][:control_steps].astype(np.int8)

                rewards[e:e+1] = data['reward'][:control_steps].astype(np.float16)
                normalised_coverage[e:e+1] = data['normalised_coverage'][:control_steps+1].astype(np.float16)
                normalised_improvement[e:e+1] = data['normalised_improvement'][:control_steps+1].astype(np.float16)
                
                trajectory_types[e:e+1] = trial_type
                flatten_coverage[e:e+1] = data['flatten_coverage'].astype(np.float16)
                e += 1
            
        f.flush()
        f.close()

    print('Finish Collection')

def perform(env, policy):
    data = {}

    obs, _, _ = env.reset()
    state = env.get_state()
    
    #observations
    data['rgbd'] = [np.expand_dims(obs['image'], 0)]
    data['particle_pos'] = [np.expand_dims(state['particle_pos'], 0)]
    data['picker_pos'] = [np.expand_dims(state['picker_pos'], 0)]
    eval_data = env.evaluate()
    for k, v in eval_data.items():
        data[k] =  [np.asarray(v).reshape(-1, 1)]
    
    # Info
    res_data = {
        'flatten_coverage': env.get_flatten_coverage()
    }
    

    # actions
    data['pick_and_place_action'] = []
    data['pick_and_place_action_step'] = []
    data['control_signal'] = []
    data['phase'] = []


    # Evaluation
    data['reward'] = []
    


    t = 0
    done = False
    while not done:

        action = policy.sample_action(obs, env)
        action_type = policy.get_action_type()

        obs, reward, done = env.step(action)

        step_data = env.get_step_info()
        steps = step_data['reward'].shape[0]

        data['rgbd'].append(step_data['rgbd'])
        data['particle_pos'].append(step_data['particle_pos'])
        data['picker_pos'].append(step_data['picker_pos'])
        
        data['pick_and_place_action'].append(np.stack([action for _ in range(steps)]))
        data['pick_and_place_action_step'].append(np.full((steps, 1), t))
        data['control_signal'].append(step_data['control_signal'])
        data['phase'].append(np.full((steps, 1), action_phase_to_scalar[action_type]))
        
        data['normalised_coverage'].append(step_data['normalised_coverage'].reshape(-1, 1))
        data['normalised_improvement'].append(step_data['normalised_improvement'].reshape(-1, 1))
        data['reward'].append(step_data['reward'].reshape(-1, 1))
        
        t += 1
        if done:
            break
    


    for k, v in data.items():
        res_data[k] = np.concatenate(v, axis=0)

    return res_data


if __name__ == '__main__':
    main()