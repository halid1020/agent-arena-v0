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

    arena = ag_ar.build_arena('softgym|domain:rainbow-square-fabric,initial:crumple,action:pixel-pick-and-place(1),task:flattening,disp:False')
    arena.set_eval()

    
    ## There will be 8 bins for allocating tiers, where the id is represeted by binary numbers in 3 bits.
    ## The first bit is for differentiating the intial normalised coverage of  > 0.65  and < 0.4
    ## The second bit is for differentiating the length:witdth ratio of the fabric is <= 1.2 and >= 1.5
    ## The third bit is for differentiating the area of the fabric <= 0.15 and >= 0.25

    bins = {i : [] for i in range(4)}

    for i in tqdm(range(200)):
        arena.reset({'eid': i, 'save_video': False})
        nc = arena.get_normalised_coverage()
        length, width = arena.get_cloth_dim()
        
        a = length * width
        print('length', length, 'nc', nc)

        ## calculate bin number
        num = 0
        if nc > 0.6:
            num += 0
        elif nc < 0.4:
            num += 2
        else:
            continue

        if length >= 0.5:
            num += 1
        elif length <= 0.4:
            num += 0
        else:
            continue

        bins[num].append((i, nc, length))

    print(bins)
    ## print the results with mean and std, min and max, and episode numbers
    for k, v in bins.items():
        vv = v[:7]
        nc = [x[1] for x in vv]
        length = [x[2] for x in vv]
        eps = [x[0] for x in vv]
        
        try:
            print('\ntier', k)
            print('nc mean:', np.mean(nc), ', std:', np.std(nc), ', min:', np.min(nc), ', max:', np.max(nc))
            print('length mean:', np.mean(length), ', std:', np.std(length), ', min:', np.min(length), ', max:', np.max(length))
            print('count', len(v), 'eps', eps)
        except:
            continue


   



if __name__ == '__main__':
    main()