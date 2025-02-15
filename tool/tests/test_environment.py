import os
import sys
sys.path.insert(0, '..')

import argparse
import ruamel.yaml as yaml
from pathlib import Path

from yamls.settings import *
from environments.env_builder import EnvBuilder
from policies.base_policies import RandomPolicy

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='default')
    parser.add_argument('--eid', default=0)
    #parser.add_argument('--policy', default='default')
    args = parser.parse_args()
    cwd = os.getcwd() # get current working directory
    src_index = cwd.find("src") # find the index of "src" in the path
    if src_index != -1: # if "src" is found in the path
        truncated_path = cwd[:src_index+3] # truncate the path at "src"
    else:
        raise Exception("Could not find src directory. Make sure you are running the script from the root of the repository.")
    

    # Environment
    print()
    print('Initialising Environment {}'.format(args.env))
    env = EnvBuilder.build(args.env)
    env.set_eval()
    
    

    # Initialise Expert Policy
    action_space = env.get_action_space()
    print('action space', action_space)
    policy = RandomPolicy(
        action_dim = action_space.shape,
        action_lower_bound = action_space.low,
        action_upper_bound = action_space.high
    )


    # Starta an Eval Episode
    env.set_train()
    info = env.reset(episode_id=args.eid)
    done = False

    while not info['done']:
        action = policy.act(info['observation'], env)
        print('action', action)
        info = env.step(action)
        print('evaluation', env.evaluate())


if __name__ == '__main__':
    main()