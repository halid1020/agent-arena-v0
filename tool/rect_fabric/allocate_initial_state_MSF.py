"""

For testing the script, simply run

python allocate_initial_state.py

"""

import numpy as np
import argparse
from tqdm import tqdm

import api as ag_ar


def main():
  
    # have arguements for domain
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default="mono-square-fabric")
    args = parser.parse_args()

    arena = ag_ar.build_arena('softgym|domain:{},initial:crumple,action:pixel-pick-and-place(1),task:flattening,gui:False'.format(args.domain))
    arena.set_eval()

    bins = {
        '0': [0.9, 1.0],
        '1': [0.65, 0.85],
        '2': [0.55, 0.65],
        '3': [0.4, 0.45],
        '4': [0, 0.35]
    }

    res = {k: [] for k, v in bins.items()}

    for i in tqdm(range(100)):
        arena.reset({'eid': i, 'save_video': False})
        nc = arena.get_normalised_coverage()

        for k, v in bins.items():
            if v[0] <= nc <= v[1]:
                res[k].append((i, nc))
                break
    
    ## print the results with mean and std, min and max, and episode numbers
    for k, v in res.items():
        nc = [x[1] for x in v]
        eps = [x[0] for x in v]
        print('\ntier', k, ', mean:', np.mean(nc), ', std:', np.std(nc), ', min:', np.min(nc), ', max:', np.max(nc))
        print('count', len(v), 'eps', eps)


   



if __name__ == '__main__':
    main()