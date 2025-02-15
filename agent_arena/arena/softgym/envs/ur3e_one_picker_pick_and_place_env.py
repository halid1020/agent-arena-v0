import os
import numpy as np

from ..picker_action_wrappers.pixel_pick_and_place import PixelPickAndPlace
from .cloth_funnel_env import ClothFunnelEnv
from .camera_utils import pixel2world
import ray


import os
import numpy as np
import pandas as pd
import pyflex
from softgym.action_space.action_space import Picker

from agent_arena.utilities.logger.logger_interface import Logger
from agent_arena.utilities.visual_utils \
    import save_video as sv
from agent_arena.utilities.visual_utils \
    import save_numpy_as_gif as sg
from agent_arena.utilities.visual_utils \
    import plot_pick_and_place_trajectory as pt


global CLOTH_FUNNEL_ENV_NUM
CLOTH_FUNNEL_ENV_NUM = 0


class CustomLogger(Logger):
    
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
        
        # internal_stattes = result['internal_states'] ## list of dictionaries
        # print('internal_stattes', internal_stattes[0].keys())

        
        df = pd.DataFrame.from_dict(df_dict)
        performance_file = \
            os.path.join(self.log_dir, filename, 'performance.csv'.format(filename))
        written = os.path.exists(performance_file)

        
        df.to_csv(
            performance_file, 
            mode= ('a' if written else 'w'), 
            header= (False if written else True)
        )

        #if result['actions'][0] is is instance of dict
        if isinstance(result['actions'][0], dict):
            pick_actions = []
            place_actions = []
            for action in result['actions']:
                if 'norm-pixel-pick-and-place' in action.keys():
                    action_ = action['norm-pixel-pick-and-place']
                else:
                    action_ = action
                pick_actions.append(action_['pick_0'])
                place_actions.append(action_['place_0'])
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
        workspace_mask = result['informations'][0]['observation']['workspace_mask']
        # apply workspace mask on the rgbs with interpolation value 0.5
        alpha = 0.5
        rgbs = (1 - alpha) * (1.0*rgbs/255.0) + alpha * np.stack([workspace_mask]*3, axis=2).astype(np.float32)
        rgbs = (rgbs * 255).astype(np.uint8)
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
    

class UR3eOnePickerPickAndPlaceEnv(ClothFunnelEnv):

    """"
        This envionrment can be initialised with string
        'softgym|domain:ur3e-<object_name>,task:flattening,horizon:<N>'
    """
    name = 'ur3e_one_picker_pick_and_place_env'

    def __init__(self, config):

        ## Agent Arena setups, TODO: eval_params and val_params.
        self.config = config
        self.info = {}
        self.save_video = False
        self.name = f'ur3e-sim-{config.object}'
        
        self.random_reset = False
        self.set_id(0)
        
        # Softgym Setup
        self._get_sim_config()
        self._setup_camera()

        self.scene_config = self.default_config['scene_config']
        
        #print('disp', config.disp)
        headless = not config.disp
        #print('headless', headless)
        pyflex.init(headless, 
                    True, 
                    self.camera_config['cam_size'][0], 
                    self.camera_config['cam_size'][1],
                    )

        self.pickers = Picker(
            2, picker_radius= self.config.picker_radius, 
            particle_radius=self.scene_config['radius'],
            picker_threshold=self.config.picker_threshold,
            picker_low=self.config.picker_low, 
            picker_high=self.config.picker_high,
            grasp_mode=(self.config.grasp_mode if 'grasp_mode' in self.config.keys() else {'closest': 1.0}),
        )
        self.particle_radius = self.scene_config['radius']

        self.logger = CustomLogger()
        self.action_tool = PixelPickAndPlace(
            action_horizon=self.config.horizon,
            single_operator=True,
            pick_height=0.025,
            drag_vel=0.01)

        self._observation_image_shape = config.observation_image_shape \
            if 'observation_image_shape' in config else (480, 480, 3)
        self._process_workspace_mask()

        self._get_init_state_keys()
       
    
    def _get_obs(self):
        super_obs = super()._get_obs()
        super_obs['workspace_mask'] = self.workspace_mask
        return super_obs

    def _process_workspace_mask(self):
         # camera position is [0, 0.3, 0.75], the camera and robot base distance is 0.5 on y axis
        self.robot_base = np.array([0, 0.35, 0])
        #print('camera_extrinsic_matrix', self.camera_extrinsic_matrix)
        self.far_raidus= 0.52
        self.near_radius = 0.24

        # go through each pixel on the image and check if it is in the workspace
        W, H = self.camera_config['cam_size']

        # get the x, y positions of each pixel
        pixels = np.indices((H, W)).reshape(2, -1).T
        depths = [self.camera_config['cam_position'][1]]*pixels.shape[0]
        world_poses = pixel2world(pixels, self.camera_intrinsic_matrix, 
                                  self.camera_extrinsic_matrix, depths).reshape(H, W ,3)
        distances = np.linalg.norm(world_poses - self.robot_base, axis=2)
        
        self.workspace_mask = np.zeros((H, W))
        self.workspace_mask[distances < self.far_raidus] = 1
        self.workspace_mask[distances < self.near_radius] = 0
        self.workspace_mask = self.workspace_mask.astype(np.bool)
        # plt workspace mask

        # from matplotlib import pyplot as plt
        # plt.imshow(self.workspace_mask)
        # plt.savefig('workspace_mask.png')

    
    def _get_sim_config(self):
        super()._get_sim_config()
        self.default_config['camera_params']['default_camera'] = {
            'render_type': ['cloth'],
            'cam_position': [0, 0.75, 0], #[0, 0.75, 0],
            'cam_angle': [0, -90 / 180. * np.pi, 0.], #[np.pi/2, -np.pi / 2, 0],
            'cam_size': [848, 480], #[1280, 720],
            'cam_fov': 50.0 / 180 * np.pi
        }


    def step(self, action): 
        ## get pixel actino on the observation image, but we need to swap the x and y before passing to the action tool
        self.last_info = self.info
        self.evaluate_result = None
        #print('action', action) 
        #TODO: check if the action lies in the workspace
        W, H = self.camera_config['cam_size']
        if 'norm-pixel-pick-and-place' in action.keys():
            action = action['norm-pixel-pick-and-place']
        #print(f'camera height: {H}, width: {W}')
        pick_0 = (action['pick_0'] + 1)/2*np.array([H, W]).clip(np.array([0, 0]), np.array([H-1, W-1]))
        place_0 = (action['place_0'] + 1)/2*np.array([H, W]).clip(np.array([0, 0]), np.array([H-1, W-1]))
        pick_x, pick_y = int(pick_0[0]), int(pick_0[1])
        place_x, place_y = int(place_0[0]), int(place_0[1])
       

        if not self.workspace_mask[pick_x][pick_y] or \
            not self.workspace_mask[place_x][place_y]:
            print(f'Reject Action.\nRobot is trying to pick {pick_0} or place {place_0} outside the workspace')
            self.action_tool.action_step += 1
            info = {'done': self.action_tool.action_step >= self.action_tool.action_horizon} 
        else:
            # action['pick_0'] = action['pick_0'][::-1]
            # action['place_0'] = action['place_0'][::-1]
            info = self.action_tool.step(self, action)
        self.info = self._process_info(info)
        return self.info
    

   

@ray.remote(num_gpus=0.05)
class UR3eOnePickerPickAndPlaceEnvRay(UR3eOnePickerPickAndPlaceEnv):
    
    def __init__(self, config):
        super().__init__(config)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU ID
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Make CUDA calls synchronous