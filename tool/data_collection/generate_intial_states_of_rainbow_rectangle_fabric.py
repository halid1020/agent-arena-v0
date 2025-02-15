import sys
sys.path.insert(0, '..')

import numpy as np

from environments.softgym_cloth_flatten_env \
    import SoftGymClothFlattenEnv
from cloth_folding_IRL.src.policies.pick_and_place_rect_fabric_flattening_policies \
    import PickAndPlaceExpertPolicy
from logger.visualisation_utils import plot_pick_and_place_trajectory

def main():
    
    # Intialise Environment
    env_para =  {
        "headless": True,  # Shows GUI
        'random_seed': 0,

        "use_cached_states": True,
        "save_cached_states": True,
        "cached_states_path": 'crumpled_random_rectangle_fabric_with_different_colours_and_positions.pkl',
        "num_variations": 5000,

        "reward_mode": 'hoque_ddpg', # Reward Option
        "action_mode": "pickerpickplace",
        "step_mode": "pixel_pick_and_place",
        "action_horizon": 10,
        "control_horizon": None,
        "picker_low": [-1, -1, -1, -1],
        "picker_high": [1, 1, 1, 1],

        "motion_trajectory": 'triangle',
        "pick_height": 0.025,
        "place_height": 0.06,
        "intermidiate_height": 0.15,
        "end_trajectory_move": True,
        "recolour_config": False,
        'context': {
            'state': True,
            'position': 0.6,
            'rotation': True,
            
            
            'colour': {
                'front_colour':
                {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },

                'back_colour':
                {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                } 
            },

            'size': {
                'width':{
                    'lower_bound': 0.2,
                    'upper_bound': 0.7
                },

                'length':{
                    'lower_bound': 0.2,
                    'upper_bound': 0.7
                }
            }
        },
        'context_cloth_colour': False
        
    }
    environment = SoftGymClothFlattenEnv(env_para)
    

    initial_observations = []
    environment.set_eval()
    for i in range(100):
        obs, _, _ = environment.reset(episode_id=i)
        initial_observations.append(obs['image'].copy())
    
    plot_pick_and_place_trajectory(obs=np.stack(initial_observations), show=False, save_png=True, title='initial_states', save_path='.')


if __name__ == '__main__':
    main()