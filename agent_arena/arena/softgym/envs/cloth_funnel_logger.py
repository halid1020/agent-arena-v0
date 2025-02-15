import os
import numpy as np
import pandas as pd

from agent_arena.utilities.logger.logger_interface import Logger
from agent_arena.utilities.visual_utils import save_video as sv
from agent_arena.utilities.visual_utils import save_numpy_as_gif as sg
from agent_arena.utilities.visual_utils import plot_pick_and_place_trajectory as pt

class ClothFunnelLogger(Logger):
    
    def __call__(self, episode_config, result, filename=None):

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

        plot_pick_and_place = True
        #if result['actions'][0] is is instance of dict
        if isinstance(result['actions'][0], dict):
            pick0_actions = []
            pick1_actions = []
            place_actions = []
            for action in result['actions']:
                if 'norm-pixel-pick-and-place' in action.keys():
                    action_ = action['norm-pixel-pick-and-place']
                elif 'norm-pixel-pick-and-fling' in action.keys():
                    action_ = action['norm-pixel-pick-and-fling']
                    plot_pick_and_place = False
                else:
                    action_ = action
                pick0_actions.append(action_['pick_0'])

                if 'place_0' in action_.keys():
                    place_actions.append(action_['place_0'])
                
                if 'pick_1' in action_.keys():
                    place_actions.append(action_['pick_1'])
                    plot_pick_and_place = False

              
            pick0_actions = np.stack(pick0_actions)
            place_actions = np.stack(place_actions)
            result['actions'] = np.concatenate([pick0_actions, place_actions], axis=1)
            T = result['actions'].shape[0]
            N = 1
        else:
            result['actions'] = np.stack(result['actions'])
            T = result['actions'].shape[0]
            N = result['actions'].shape[1]
        result['actions'] = result['actions'].reshape(T, N, 2, -1)[:, :, :, :2]
       
        if plot_pick_and_place:
            rgbs = np.stack([info['observation']['rgb'] for info in result['informations']])
            pt(
                rgbs, result['actions'].reshape(T, -1, 4), # TODO: this is envionrment specific
                title='Episode {}'.format(eid), 
                # rewards=result['rewards'], 
                save_png = True, save_path=os.path.join(self.log_dir, filename, 'performance_visualisation'), col=5)
            
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