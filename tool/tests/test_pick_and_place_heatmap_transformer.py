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
    parser.add_argument('--setting', default='pick-and-place-cloth-diagonal-folding-from-crumpled')
    parser.add_argument('--agent',  default='rssm-bc-pick-and-place-heatmap')
    parser.add_argument('--policy', default='')
    parser.add_argument('--hyper_para',  default='heatmap-test')
    args = parser.parse_args()
    if args.policy == '':
        args.policy = args.setting

    
    ### Initilise Project
    configs = yaml.safe_load(Path('../scripts/yamls/{}/{}/{}.yaml'.format(args.agent, args.setting, args.hyper_para)).read_text())
    configs = DotMap(configs)

     ### Initialise Transformer
    transform = name_to_transformer[configs.bc_transform.name](configs.bc_transform.params)

    ### Initiliase dataset
    dataset_params = yaml.safe_load(Path('../scripts/yamls/dataset.yaml').read_text())[configs.bc_dataset.name]
    dataset = name_to_dataset[configs.bc_dataset.name](
        **dataset_params,
        config=configs.bc_dataset.params,
        transform=transform,
        mode='train')


    ### Plot Trajectory after Transforming
    for i in range(1):
        episode = dataset.get_episode(i, transform=True, train=True)
        
        ## Post process observation for plotting
        obs = ((episode['observation'].detach().cpu().numpy().transpose(0, 2, 3, 1) + 0.5)*255).astype(np.uint8)
        pick_heatmap = episode['pick_heatmap'].unsqueeze(-1).detach().cpu().numpy()
        place_heatmap = episode['place_heatmap'].unsqueeze(-1).detach().cpu().numpy()
        actions = episode['action'].detach().cpu().numpy()

        print('actions shape', actions.shape)

        plot_pick_and_place_trajectory(obs=obs, acts1=actions, show=False, save_png=True, save_path='.', title='obs')
        plot_pick_and_place_trajectory(obs=pick_heatmap, show=False, save_png=True, save_path='.', title='pick_heatmap')
        plot_pick_and_place_trajectory(obs=place_heatmap, show=False, save_png=True, save_path='.', title='place_heatmap')



if __name__ == '__main__':
    main()