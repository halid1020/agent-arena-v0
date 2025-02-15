import numpy as np
from gym.spaces import Box

from ...softgym.picker_action_wrappers.world_pick_and_fling \
    import WorldPickAndFling

from .utils import pixel_to_world

class PixelPickAndFling():

    def __init__(self, 
        lowest_cloth_height=0.1,
        max_grasp_dist=0.7,
        stretch_increment_dist=0.02,
        
        pregrasp_height=0.3,
        pregrasp_vel=0.1,
        tograsp_vel=0.05,
        prefling_height=0.7,#
        prefling_vel=0.01,
        fling_pos_y=0.3,
        lift_vel=0.01,
        adaptive_fling_momentum=0.8,
        action_horizon=20,
        hang_adjust_vel=0.01,
        stretch_adjust_vel=0.01,
        fling_vel=0.008,
        release_vel=0.01,
        drag_vel=0.01,
        lower_height=0.06,

        pick_lower_bound=[-1, -1],
        pick_upper_bound=[1, 1],
        place_lower_bound=[-1, -1],
        place_upper_bound=[1, 1],
        pick_height=0.025, #0.02,

        **kwargs):
        
        ### Environment has to be WorldPickAndFlingWrapper
        self.action_tool = WorldPickAndFling(**kwargs) 
        
        self.action_horizon = action_horizon
        self.lowest_cloth_height = lowest_cloth_height
        self.max_grasp_dist = max_grasp_dist
        self.stretch_increment_dist = stretch_increment_dist
        self.fling_vel = fling_vel
        self.pregrasp_height = pregrasp_height
        self.pregrasp_vel = pregrasp_vel
        self.tograsp_vel = tograsp_vel
        self.prefling_height = prefling_height
        self.prefling_vel = prefling_vel
        self.fling_pos_y = fling_pos_y
        self.lift_vel = lift_vel
        self.adaptive_fling_momentum = adaptive_fling_momentum
        self.pick_height = pick_height
        self.hang_adjust_vel = hang_adjust_vel
        self.stretch_adjust_vel = stretch_adjust_vel
        self.release_vel = release_vel
        self.drag_vel = drag_vel
        self.lower_height = lower_height

        self.num_pickers = 2


        

        space_low = np.concatenate([pick_lower_bound, place_lower_bound]*self.num_pickers)\
            .reshape(self.num_pickers, -1).astype(np.float32)
        space_high = np.concatenate([pick_upper_bound, place_upper_bound]*self.num_pickers)\
            .reshape(self.num_pickers, -1).astype(np.float32)
        self.action_space = Box(space_low, space_high, dtype=np.float32)

    
    def get_no_op(self):
        return self.no_op
        
    def sample_random_action(self):
        return self.action_space.sample()

    def get_action_space(self):
        return self.action_space
    
    def get_action_horizon(self):
        return self.action_horizon
    
    def reset(self, env):
        return self.action_tool.reset(env)
    
    def process(self, action):
        #action = action['norm_pixel_pick_and_fling']
        p0 = np.asarray(action['pick_0'])
        p1 = np.asarray(action['pick_1'])

        # p0[0], p0[1] = -0.7, 0.9
        # p1[0], p1[1] = 0.3, -0.6

        # print('p0:', p0)
        # print('p1:', p1)

        ## Assuming top-down view
        # p0 = p0 * self.camera_to_world_ratio * self.camera_height
        # p1 = p1 * self.camera_to_world_ratio * self.camera_height
        # p0 = np.concatenate([p0, [self.pick_height]])
        # p1 = np.concatenate([p1, [self.pick_height]])

        # convert to world coordinate
        p0 = pixel_to_world(p0, self.camera_height-self.pick_height, 
            self.camera_intrinsics, self.camera_pose, self.camera_size)
        p1 = pixel_to_world(p1, self.camera_height-self.pick_height, 
            self.camera_intrinsics, self.camera_pose, self.camera_size)

        # print('p0:', p0)
        # print('p1:', p1)

        

        return {
            'pick_0_position': p0,
            'pick_1_position': p1,
            'pregrasp_height': self.pregrasp_height,
            'pregrasp_vel': self.pregrasp_vel,
            'tograsp_vel': self.tograsp_vel,
            'prefling_height': self.prefling_height,
            'prefling_vel': self.prefling_vel,
            'lift_vel': self.lift_vel,

            'fling_pos_y': self.fling_pos_y,
            'hang_adjust_vel': self.hang_adjust_vel, # for hang and stretch
            'stretch_adjust_vel': self.stretch_adjust_vel, # for hang and stretch
            'fling_vel': self.fling_vel, # for fling 
            'release_vel': self.release_vel, # for fling and release
            'drag_vel': self.drag_vel, # for  fling and release
            'lower_height': self.lower_height, # for lower the picker before release
        }
    
    ## It accpet action has shape (num_picker, 2, 3), where num_picker can be 1 or 2
    def step(self, env, action):
        self.camera_height = env.camera_height
        # self.camera_to_world_ratio = env.pixel_to_world_ratio
        self.camera_intrinsics = env.camera_intrinsic_matrix
        self.camera_pose = env.camera_extrinsic_matrix
        self.camera_size = env.camera_size
        action_ = self.process(action)
        return self.action_tool.step(env, action_)