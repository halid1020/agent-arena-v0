import os
import pandas as pd
import matplotlib.image as mpimg

from agent_arena.utilities.logger.logger_interface import Logger
from agent_arena.utilities.visual_utils import save_video as sv
from agent_arena.utilities.visual_utils import save_numpy_as_gif as sg
from agent_arena.utilities.visual_utils import plot_image_trajectory as pt

class StandardLogger(Logger):
    def __init__(self):
        super().__init__()
    
    def __call__(self, episode_config, result, filename=None):

        eid, save_video = episode_config['eid'], episode_config['save_video']

        save_goal = False
        if 'save_goal' in episode_config:
            save_goal = episode_config['save_goal']

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
        if filename is None:
            filename = 'manupilation'
        
        if not os.path.exists(os.path.join(self.log_dir, filename)):
            os.makedirs(os.path.join(self.log_dir, filename))

        df_dict = {
            'episode_id': [eid],
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

        # pt(
        #     result['rgb'], # TODO: this is envionrment specific 
        #     title='Episode {}'.format(eid), 
        #     # rewards=result['rewards'], 
        #     save_png = True, save_path=os.path.join(self.log_dir, filename, 'performance_visualisation'))
        
        if save_goal:

            mpimg.imsave(
                os.path.join(
                    self.log_dir, 
                    filename, 
                    'performance_visualisation',
                    'episode_{}_goal.png'.format(eid)
                ), 
                result['goal']['rgb']
            )
        
        if save_video and 'frames' in result:    
            sv(result['frames'], 
                os.path.join(self.log_dir, filename, 'performance_visualisation'),
                'episode_{}'.format(eid))

        if save_video and 'frames' in result:    
            sg(
                result['frames'], 
                os.path.join(
                    self.log_dir, 
                    filename, 
                    'performance_visualisation',
                    'episode_{}'.format(eid)
                )
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

    