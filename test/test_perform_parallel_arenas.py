import os

import agent_arena as ag_ar
from agent_arena.utilities.utils import create_message_logger
from agent_arena.utilities.visual_utils import plot_pick_and_place_trajectory as pt
from agent_arena.utilities.perform_parallel import setup_arenas, perform_parallel
import ray

import numpy as np
from time import time


def main():
    arena_name = 'softgym|domain:clothfunnels-real2sim-longsleeve,task:flattening,horizon:5'
    agent_name = 'cloth-funnel'
    config_name = 'place_only'
    log_dir = 'test_results'
    log_dir = os.path.join(log_dir, arena_name, agent_name, config_name)

    ray.init()

    config = ag_ar.retrieve_config(agent_name, arena_name, config_name)
    arenas = setup_arenas(arena_name, num_processes=4)
    agent = ag_ar.build_agent(agent_name, config=config)
    
    for arena in arenas:
        #arena = ray.get(arena_ref)
        arena.set_log_dir.remote(log_dir)
    agent.set_log_dir(log_dir)

    start = time()

    results = perform_parallel(arenas, agent)

    ## time taken in seconds
    print(f'Time taken{time()-start} seconds')
    
    for i, result in enumerate(results):
        
        if isinstance(result['actions'][0], dict):
            pick_actions = []
            place_actions = []
            for action in result['actions']:
                action = action['norm-pixel-pick-and-place']
                pick_actions.append(action['pick_0'])
                place_actions.append(action['place_0'])
            pick_actions = np.stack(pick_actions)
            place_actions = np.stack(place_actions)
            result['actions'] = np.concatenate([pick_actions, place_actions], axis=1)
            T = result['actions'].shape[0]
            N = 1
        
        result['actions'] = result['actions'].reshape(T, N, 2, -1)[:, :, :, :2]
        print('keys of information', result['informations'][0].keys())
        rgbs = np.stack([info['observation']['rgb'] for info in result['informations']])

        pt(
            rgbs, result['actions'].reshape(T, -1, 4), # TODO: this is envionrment specific
            title='Result {}'.format(i), 
            # rewards=result['rewards'], 
            save_png = True, save_path=os.path.join(log_dir, 'performance_visualisation'), col=5)



if __name__ == '__main__':
    main()