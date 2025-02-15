import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from agent_arena.utilities.visual_utils import save_video as sv
from agent_arena.utilities.visual_utils import save_numpy_as_gif as sg
from agent_arena.utilities.visual_utils import plot_pick_and_place_trajectory as pt

def pick_and_place_fabric_manupilation_logger(
    eps_param, res, save_dir, filename=None):

    tier, eid, save_video = eps_param['tier'], eps_param['eid'], eps_param['save_video']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if filename is None:
        filename = 'manupilation'
    
    if not os.path.exists(os.path.join(save_dir, filename)):
        os.makedirs(os.path.join(save_dir, filename))

    df_dict = {
        'tier': [tier],
        'episode_id': [eid],
        'return': [res['return']]
    }

    evaluations = res['evaluation']
    evaluation_keys = list(evaluations .keys())
    for k in evaluation_keys:
        df_dict['evaluation/'+ k] = [evaluations[k]]

    
    df = pd.DataFrame.from_dict(df_dict)
    performance_file = \
        os.path.join(save_dir, filename, 'performance.csv'.format(filename))
    written = os.path.exists(performance_file)

    
    df.to_csv(
        performance_file, 
        mode= ('a' if written else 'w'), 
        header= (False if written else True)
    )

    # print('res action shape', res['actions'].shape)
    T = res['actions'].shape[0]
    res['actions'] = res['actions'].reshape(T, 2, -1)[:, :, :2]
    pt(
        res['rgb'], acts1=res['actions'].reshape(-1, 4), # TODO: this is envionrment specific 
        title='Episode {}'.format(eid), 
        # rewards=res['rewards'], 
        save_png = True, save_path=os.path.join(save_dir, filename, 'performance_visualisation'))
    
    if save_video and 'frames' in res:    
        sv(res['frames'], 
            os.path.join(save_dir, filename, 'performance_visualisation'),
            'episode_{}_tier_{}'.format(eid, tier))

    if save_video and 'frames' in res:    
        sg(
            res['frames'], 
            os.path.join(
                save_dir, 
                filename, 
                'performance_visualisation',
                'episode_{}_tier_{}'.format(eid, tier)
            )
        )
    
    if save_video and 'frames' in res:
        save_results(res, os.path.join(save_dir, filename, 'performance_npy'), 
                        'episode_{}_tier_{}'.format(eid, tier))
        
def pick_and_place_rect_fabric_all_task_manupilation_logger(
    eps_param, res, save_dir):

    pick_and_place_fabric_manupilation_logger(
        eps_param, res, save_dir, filename=eps_param['task'])
        

def save_goal_logger(eps_param, res, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    file_dir = os.path.join(save_dir, 'rgb.png')
    plt.imsave(file_dir, res['rgb'][-1])

    # save depth with 1 dimension
    file_dir = os.path.join(save_dir, 'depth.npy')
    np.save(file_dir, res['depth'][-1])


def save_results(res, save_dir, filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ## res is a dictionary 
    for k, v in res.items():
        np.save(os.path.join(save_dir, '{}_{}'.format(filename, k)), v)