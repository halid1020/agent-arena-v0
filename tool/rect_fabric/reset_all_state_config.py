"""

For testing the script, simply run

python reset_all_state_config.py

"""

import numpy as np
import argparse
from tqdm import tqdm

import api as ag_ar
from utilities.utils import create_message_logger

def main():
  
    # have arguements for domain
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default="mono-square-fabric")
    parser.add_argument('--intial', default='crumple')
    args = parser.parse_args()

    create_message_logger('tmp', 'info')

    arena = ag_ar.build_arena('softgym|domain:{},initial:{},action:pixel-pick-and-place(1),task:flattening,gui:False'\
                              .format(args.domain, args.intial))
    arena.set_eval()

    
    
    arena.set_train()
    for i in tqdm(range(899, 0, -1)):
        arena.reset({'eid': i, 'save_video': False})

    arena.set_eval()
    for i in tqdm(range(100)):
        arena.reset({'eid': i, 'save_video': False})


if __name__ == '__main__':
    main()