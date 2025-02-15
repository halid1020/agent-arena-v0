import sys
sys.path.insert(0, '..')

import h5py

from logger.visualisation_utils import *
from environments.softgym_cloth_diagonal_folding_env \
    import SoftGymClothDiagonalFoldingEnv


h5py_file = 'example.hdf5'

env_para =  {
        "headless": False,  # Shows GUI
        "render": True,

        'initial_state': 'crumple',
        'use_cached_states': True,
        'save_cached_states': False,
        'cached_states_path': 'cloth_crumple.pkl',
        'num_variations': 1000,
        
        'observation_image_shape': (64, 64, 4),
        'reward_mode': 'normalised_particle_distance', # Reward Option
        'action_horizon': 15,
        'control_horizon': None,
        'action_mode': 'pickerpickplace',
        'picker_low': [-1, -1, -1, -1],
        'pick_high': [1, 1, 1, 1],

        'action_mode': 'velocity_control',
        'num_pickers': 1,

        'save_step_info': False

    }
env = SoftGymClothDiagonalFoldingEnv(env_para)
env.set_train()


with h5py.File(h5py_file, 'r') as f:
    eid = 0
    control_steps = 1000
    env.reset()
    for step in range(control_steps)[1:]:
        #print('step', step)
        
        state = {
            'particle_pos': f['particle_pos'][eid][step],
            'picker_pos': f['picker_pos'][eid][step],
            'action_step': f['pick_and_place_action_step'][eid][step],
            'control_step': step,
            'current_action_coverage': f['coverage'][eid][step],
            'prior_action_coverage': f['coverage'][eid][step-1]
        }
        env.set_state(state)
    