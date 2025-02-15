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
    parser.add_argument('--policy', default='')
    parser.add_argument('--eid', default=0)
    parser.add_argument('--camera', default='default_camera')
    args = parser.parse_args()
    

    ### Initiliase Policy
    print()
    print('Initialising policy {}'.format(args.policy))
    policy = PolicyBuilder.build(args.policy)


    # Environment
    print()
    print('Initialising Environment {}'.format(args.env))
    env = EnvBuilder.build(args.env)
    env.set_eval()
        
    ### Starta an Eval Episode
    env.set_train()
    info = env.reset(episode_id=int(args.eid))
    print(env.evaluate())
    done = False

    while not info['done']:
        action = policy.act(info['observation'], env)
        info = env.step(action)
        image = env.render(camera_name=args.camera, resolution=(128, 128))
        plt.imshow(image)
        plt.show()
        image = env.get_cloth_mask(camera_name=args.camera, resolution=(128, 128))
        print(env.evaluate())


if __name__ == '__main__':
    main()