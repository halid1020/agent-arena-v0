import sys
sys.path.insert(0, '..')

from dotmap import DotMap
import ruamel.yaml as yaml
from pathlib import Path
import argparse

from scripts.settings import *


def main():
    ### Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pick-and-place-cloth-flattening-old-d')
    args = parser.parse_args()
    
    ### Initiliase dataset
    dataset_params = yaml.safe_load(Path('../scripts/yamls/dataset.yaml').read_text())[args.dataset]
    config = DotMap({'num_episodes': 20000, 'rotation_degree': 0, 'fip_vertical': False}) ## TODO refactor dataset to not need this
    train_dataset = name_to_dataset[args.dataset](**dataset_params, mode='train', config=config)

    for i in range(10):
        data = train_dataset[i]
        # print observation min max mean std var with printout message
        print('observation min: ', data['observation'].min())
        print('observation max: ', data['observation'].max())
        print('observation mean: ', data['observation'].mean())
        print('observation std: ', data['observation'].std())

        print()


if __name__ == '__main__':
    main()