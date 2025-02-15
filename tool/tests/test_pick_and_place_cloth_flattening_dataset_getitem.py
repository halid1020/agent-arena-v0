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
    parser.add_argument('--setting', default='mono-square-fabric-legacy-pick-and-place-flattening')
    parser.add_argument('--agent',  default='planet-pick')
    parser.add_argument('--policy', default='')
    parser.add_argument('--hyper_para',  default='test-transformer')
    args = parser.parse_args()
    if args.policy == '':
        args.policy = args.setting

    
    ### Initilise Project
    configs = yaml.safe_load(Path('../scripts/yamls/{}/{}/{}.yaml'.format(args.agent, args.setting, args.hyper_para)).read_text())
    configs = DotMap(configs)

     ### Initialise Transformer
    transform = name_to_transformer[configs.transform.name](configs.transform.params)

    ### Initiliase dataset
    dataset_params = yaml.safe_load(Path('../scripts/yamls/dataset.yaml').read_text())[args.setting]
    train_dataset = name_to_dataset[configs.dataset.name](
                **dataset_params, 
                config=configs.dataset.params,
                transform=transform,
                mode='train')

   


    ### Plot Trajectory after Transforming
    for i in range(10):
        episode = train_dataset[i]

        
        ## Post process observation for plotting
        obs =  episode['observation'].detach().cpu().numpy().transpose(0, 2, 3, 1)
        obs = ((obs+0.5)*255).clip(0, 255).astype(np.int)
        act = episode['action'].detach().cpu().numpy()

        plot_pick_and_place_trajectory(obs=obs, acts1=act, rewards=episode['terminal'], show=False, save_png=True, save_path='.', title='data_item_{}'.format(i))
        




if __name__ == '__main__':
    main()