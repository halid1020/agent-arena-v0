import os
import numpy as np
import pandas as pd

from actoris_harena.utilities.logger.logger_interface import Logger
from actoris_harena.utilities.visual_utils import save_video as sv
from actoris_harena.utilities.visual_utils import save_numpy_as_gif as sg
from actoris_harena.utilities.visual_utils import plot_pick_and_place_trajectory as pt

class VideoLogger(Logger):
    
    def __call__(self, episode_config, result, filename=None, wandb_logger=None):

        eid, save_video = episode_config['eid'], episode_config['save_video']
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        if filename is None:
            filename = 'manupilation'
        
        if not os.path.exists(os.path.join(self.log_dir, filename)):
            os.makedirs(os.path.join(self.log_dir, filename))

        df_dict = {
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

        if save_video and 'frames' in result:    
            sv(result['frames'], 
                os.path.join(self.log_dir, filename, 'performance_visualisation'),
                'episode_{}'.format(eid))

        if save_video and 'frames' in result:    
            sg(
                result['frames'], 
                path=os.path.join(
                    self.log_dir, 
                    filename, 
                    'performance_visualisation'
                ),
                filename=f"episode_{eid}"
            )

    def check_exist(self, episode_config, filename=None):
        eid = episode_config['eid']

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
        return len(df[(df['episode_id'] == eid)]) > 0