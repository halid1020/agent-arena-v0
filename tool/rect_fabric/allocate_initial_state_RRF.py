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

    arena = ag_ar.build_arena('softgym|domain:rainbow-rect-fabric,initial:crumple,action:pixel-pick-and-place(1),task:flattening,gui:False')
    arena.set_eval()

    
    ## There will be 8 bins for allocating tiers, where the id is represeted by binary numbers in 3 bits.
    ## The first bit is for differentiating the intial normalised coverage of  > 0.65  and < 0.4
    ## The second bit is for differentiating the length:witdth ratio of the fabric is <= 1.2 and >= 1.5
    ## The third bit is for differentiating the area of the fabric <= 0.15 and >= 0.25

    bins = {i : [] for i in range(8)}

    for i in tqdm(range(500)):
        arena.reset({'eid': i, 'save_video': False})
        nc = arena.get_normalised_coverage()
        length, width = arena.get_cloth_dim()
        if length < width:
            length, width = width, length
        r = length / width
        a = length * width
        print('area', a, 'ratio', r, 'nc', nc)

        ## calculate bin number
        num = 0
        if nc > 0.6:
            num += 0
        elif nc < 0.4:
            num += 4
        else:
            continue

        if r >= 1.5:
            num += 2
        elif r <= 1.2:
            num += 0
        else:
            continue

        if a >= 0.25:
            num += 1
        elif a <= 0.18:
            num += 0
        else:
            continue

        bins[num].append((i, nc, r, a))

    print(bins)
    ## print the results with mean and std, min and max, and episode numbers
    for k, v in bins.items():
        vv = v[:7]
        nc = [x[1] for x in vv]
        rates = [x[2] for x in vv]
        areas = [x[3] for x in vv]
        eps = [x[0] for x in vv]
        
        try:
            print('\ntier', k)
            print('nc mean:', np.mean(nc), ', std:', np.std(nc), ', min:', np.min(nc), ', max:', np.max(nc))
            print('rate mean:', np.mean(rates), ', std:', np.std(rates), ', min:', np.min(rates), ', max:', np.max(rates))
            print('area mean:', np.mean(areas), ', std:', np.std(areas), ', min:', np.min(areas), ', max:', np.max(areas))
            print('count', len(v), 'eps', eps)
        except:
            continue


   



if __name__ == '__main__':
    main()