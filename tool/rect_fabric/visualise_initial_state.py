"""

For testing the script, simply run

python allocate_initial_state.py

"""

import numpy as np
import argparse
from tqdm import tqdm

import api as ag_ar
from utilities.visualisation_utils import plot_image_trajectory as pt

def main():
  
    # have arguements for domain
    domain = 'rainbow-square-fabric'
    arena = ag_ar.build_arena('softgym|domain:{},initial:crumple,action:pixel-pick-and-place(1),task:flattening,disp:False'.format(domain))
    arena.set_eval()

    
    ## There will be 8 bins for allocating tiers, where the id is represeted by binary numbers in 3 bits.
    ## The first bit is for differentiating the intial normalised coverage of  > 0.65  and < 0.4
    ## The second bit is for differentiating the length:witdth ratio of the fabric is <= 1.2 and >= 1.5
    ## The third bit is for differentiating the area of the fabric <= 0.15 and >= 0.25
    rgbs = []
    for i in tqdm(range(1, 10)):
        info = arena.reset({'eid': i, 'save_video': False})
        rgb = arena.render()
        rgbs.append(rgb)
    
    pt(
        rgbs, # TODO: this is envionrment specific
        title='{}'.format(domain), 
        # rewards=result['rewards'], 
        save_png=True, col=3, save_path=".")


   



if __name__ == '__main__':
    main()