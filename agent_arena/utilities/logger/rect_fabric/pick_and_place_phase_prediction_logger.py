import os
import numpy as np
import pandas as pd

from agent_arena.utilities.logger.logger_interface import Logger
from agent_arena.utilities.visual_utils import save_video as sv
from agent_arena.utilities.visual_utils import save_numpy_as_gif as sg
from agent_arena.utilities.visual_utils import plot_pick_and_place_trajectory as pt

class PickAndPlacePhasePredictionLogger(Logger):
    def __init__(self, log_dir):
        super().__init__(log_dir)
    
    def __call__(self, episode_config, result, filename=None):

        tier, eid, save_video = episode_config['tier'], episode_config['eid'], episode_config['save_video']
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        if filename is None:
            filename = 'manupilation'
        
        if not os.path.exists(os.path.join(self.log_dir, filename)):
            os.makedirs(os.path.join(self.log_dir, filename))

        df_dict = {
            'tier': [tier],
            'episode_id': [eid],
            #'return': [result['return']]
        }

        evaluations = result['evaluation']
        evaluation_keys = list(evaluations .keys())
        for k in evaluation_keys:
            df_dict['evaluation/'+ k] = [evaluations[k]]

        
        df = pd.DataFrame.from_dict(df_dict)
        performance_file = \
            os.path.join(self.log_dir, filename, 'performance.csv'.format(filename))
        written = os.path.exists(performance_file)

        
        df.to_csv(
            performance_file, 
            mode= ('a' if written else 'w'), 
            header= (False if written else True)
        )

        # print('result action shape', result['actions'].shape)
        #result['actions'] = np.stack([a.flatten() for a in result['actions']])
        result['actions'] = np.stack(result['actions'])
        T = result['actions'].shape[0]
        N = result['actions'].shape[1]
        result['actions'] = result['actions'].reshape(T, N, 2, -1)[:, :, :, :2]
        
        pt(
            result['rgb'], result['actions'].reshape(T, -1, 4), # TODO: this is envionrment specific
            info_ = result['phases'],
            title='Episode {}'.format(eid), 
            # rewards=result['rewards'], 
            save_png = True, 
            save_path=os.path.join(self.log_dir, filename, 'performance_visualisation'))
        
        if save_video and 'frames' in result:    
            sv(result['frames'], 
                os.path.join(self.log_dir, filename, 'performance_visualisation'),
                'episode_{}_tier_{}'.format(eid, tier))

        if save_video and 'frames' in result:    
            sg(
                result['frames'], 
                os.path.join(
                    self.log_dir, 
                    filename, 
                    'performance_visualisation',
                    'episode_{}_tier_{}'.format(eid, tier)
                )
            )

    def check_exist(self, episode_config, filename=None):
        tier, eid = episode_config['tier'], episode_config['eid']

        if filename is None:
            filename = 'manupilation'
        
        performance_file = \
            os.path.join(self.log_dir, filename, 'performance.csv')
        #print('performance_file', performance_file)

        if not os.path.exists(performance_file):
            return False
        df = pd.read_csv(performance_file)
        if len(df) == 0:
            return False
        
        ## Check if there is an entry with the same tier and eid, and return True if there is
        return len(df[(df['tier'] == tier) & (df['episode_id'] == eid)]) > 0