import os
import sys
sys.path.insert(0, '..')

from dotmap import DotMap
import ruamel.yaml as yaml
from tqdm import tqdm
from pathlib import Path
import wandb
import argparse

# from yamls.settings import *
from utilities.loggers import *
from utilities.utils import perform

from environments.env_builder import EnvBuilder
from agent.policies.builder import PolicyBuilder

def main():

    ### Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='')
    args = parser.parse_args()
    


    # Environment
    print()
    print('Initialising Environment {}'.format(args.env))
    env = EnvBuilder.build(args.env + ',gui:True')
    env.set_eval()
        
    ### Starta an Eval Episode
    initial_observations = []
    for i in range(100):
        obs, _, _ = env.reset(episode_id=i)
        initial_observations.append(obs['image'].copy())
    
    plot_trajectory(obs=np.stack(initial_observations), show=False, save_png=True, title='{}_initial_states'.format(args.env), save_path='.')


if __name__ == '__main__':
    main()