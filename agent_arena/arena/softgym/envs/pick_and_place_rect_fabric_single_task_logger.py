import os
import numpy as np
import pandas as pd

from agent_arena.utilities.logger.logger_interface import Logger
from agent_arena.utilities.visual_utils import save_video as sv
from agent_arena.utilities.visual_utils import save_numpy_as_gif as sg
from agent_arena.utilities.visual_utils import plot_pick_and_place_trajectory as pt
from agent_arena.utilities.visual_utils import plot_image_trajectory

class PickAndPlaceRectFabricSingleTaskLogger(Logger):

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
        
        internal_stattes = result['internal_states'] ## list of dictionaries
        #print('internal_stattes', internal_stattes[0].keys())

        if 'denoise_action_rgb' in internal_stattes[0]:
            denoise_action_rgbs = np.stack([state['denoise_action_rgb'] for state in internal_stattes])
            print('denoise_action_rgbs', denoise_action_rgbs.shape)
            plot_image_trajectory(
                denoise_action_rgbs, 
                save_path=os.path.join(self.log_dir, filename, 'performance_visualisation'), 
                title='episode_{}_tier_{}_denoise_action_rgb'.format(eid, tier)
            )
        
        if 'denoise_action_input_obs_rgb' in internal_stattes[0]:
            denoise_action_input_obs_rgbs = np.stack([state['denoise_action_input_obs_rgb'] for state in internal_stattes])
            plot_image_trajectory(
                denoise_action_input_obs_rgbs, 
                save_path=os.path.join(self.log_dir, filename, 'performance_visualisation'), 
                title='episode_{}_tier_{}_denoise_action_input_obs_rgb'.format(eid, tier)
            )
        
        if 'denoise_depth_frames' in internal_stattes[0]:
            for i in range(len(internal_stattes)):
                frames = internal_stattes[i]['denoise_depth_frames']
                
                ## duplicate the first and last frame for 20 times and padd to the orginal frame
                frames = np.concatenate([np.tile(frames[0], (20, 1, 1, 1)), frames, np.tile(frames[-1], (20, 1, 1, 1))], axis=0)
                print('frames', frames.shape)
                sg(
                    frames, 
                    os.path.join(
                        self.log_dir, 
                        filename, 
                        'performance_visualisation',
                        'episode_{}_tier_{}_denoise_action_step_{}'.format(eid, tier, i)
                    ),
                    fps=20,
                )

        
        df = pd.DataFrame.from_dict(df_dict)
        performance_file = \
            os.path.join(self.log_dir, filename, 'performance.csv'.format(filename))
        written = os.path.exists(performance_file)

        
        df.to_csv(
            performance_file, 
            mode= ('a' if written else 'w'), 
            header= (False if written else True)
        )

        
        if isinstance(result['actions'][0], dict):
            pick_actions = []
            place_actions = []
            for action in result['actions']:
                pick_actions.append(action['pick_0'])
                place_actions.append(action['place_0'])
            pick_actions = np.stack(pick_actions)
            place_actions = np.stack(place_actions)
            result['actions'] = np.concatenate([pick_actions, place_actions], axis=1)
            T = result['actions'].shape[0]
            N = 1
        else:
            result['actions'] = np.stack(result['actions'])
            T = result['actions'].shape[0]
            N = result['actions'].shape[1]
        result['actions'] = result['actions'].reshape(T, N, 2, -1)[:, :, :, :2]
        #print('keys of information', result['informations'][0].keys())
        rgbs = np.stack([info['observation']['rgb'] for info in result['informations']])
        pt(
            rgbs, result['actions'].reshape(T, -1, 4), # TODO: this is envionrment specific
            title='Episode {}'.format(eid), 
            # rewards=result['rewards'], 
            save_png = True, save_path=os.path.join(self.log_dir, filename, 'performance_visualisation'), col=5)
        
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