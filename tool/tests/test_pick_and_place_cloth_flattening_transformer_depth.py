import sys
sys.path.insert(0, '..')

import numpy as np
from dotmap import DotMap
import ruamel.yaml as yaml
from tqdm import tqdm
from pathlib import Path
import wandb
import argparse

from scripts.settings import *
from logger.visualisation_utils import plot_pick_and_place_trajectory

def main():

    ### Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', default='pick-and-place-cloth-flattening-old-d')
    parser.add_argument('--agent',  default='planet-pick')
    parser.add_argument('--policy', default='')
    parser.add_argument('--hyper_para',  default='z-norm-test')
    args = parser.parse_args()
    if args.policy == '':
        args.policy = args.setting

    
    ### Initilise Project
    configs = yaml.safe_load(Path('../scripts/yamls/{}/{}/{}.yaml'.format(args.agent, args.setting, args.hyper_para)).read_text())
    configs = DotMap(configs)

     ### Initialise Transformer
    transform = name_to_transformer[args.setting](configs)

    ### Initiliase dataset
    dataset_params = yaml.safe_load(Path('../scripts/yamls/dataset.yaml').read_text())[args.setting]
    train_dataset = name_to_dataset[args.setting](config=configs, **dataset_params, mode='train', transform=transform)

   


    ### Plot Trajectory after Transforming
    for i in range(1):
        episode = train_dataset.get_episode(i, transform=True, train=True)
        
        ## Post process observation for plotting
        obs =  episode['observation'].detach().cpu().numpy().transpose(0, 2, 3, 1)
        print('obs shape', obs.shape)
        
        print('obs mean', np.mean(obs))
        print('obs std', np.std(obs))
        print('obs min', np.min(obs))
        print('obs max', np.max(obs))

        plot_pick_and_place_trajectory(obs=obs, show=False, save_png=True, save_path='.')



if __name__ == '__main__':
    main()