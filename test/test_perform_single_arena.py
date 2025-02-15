import os

import agent_arena as ag_ar
from agent_arena.utilities.utils import create_message_logger
from agent_arena.utilities.visual_utils import plot_pick_and_place_trajectory as pt
from agent_arena.utilities.perform_single import perform_single
import ray

import numpy as np
from time import time


def main():
    arena_name = 'softgym|domain:cloth-funnel-longsleeve,task:flattening,horizon:30'
    agent_name = 'cloth-funnel'
    config_name = 'place_only'
    log_dir = 'test_results'

    # ray.init(local_mode=True)

    config = ag_ar.retrieve_config(agent_name, arena_name, config_name, log_dir)
    arena = ag_ar.build_arena(arena_name + ',disp:0')
    agent = ag_ar.build_agent(agent_name, config=config)
    
    arena.set_log_dir(log_dir)
    
    agent.set_log_dir(log_dir)

    start = time()

    perform_single(arena, agent)

    ## time taken in seconds
    print(f'Time taken{time()-start} seconds')
    
if __name__ == '__main__':
    main()