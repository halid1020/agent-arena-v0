import numpy as np
from gym.spaces import Dict, Discrete, Box

from ...softgym.picker_action_wrappers.world_pick_and_place \
    import WorldPickAndPlace
from ..envs.camera_utils import norm_pixel2world

class PixelPickAndDrag():

    def __init__(self, 
                 action_horizon=20,
                 pick_height=0.025,
                 place_height=0.06,
                 pick_lower_bound=[-1, -1],
                 pick_upper_bound=[1, 1],
                 place_lower_bound=[-1, -1],
                 place_upper_bound=[1, 1],
                 pregrasp_height=0.05,
                 pre_grasp_vel=0.05,
                 drag_vel=0.05,
                 lift_vel=0.05,
                 single_operator=False,
                 **kwargs):
        
        ### Environment has to be WorldPickAndFlingWrapper
        self.action_tool = WorldPickAndPlace(**kwargs) 
        

        #### Define the action space
        #self.num_picker = self.env.get_num_picker()
        self.num_pickers = 2
        #self.num_picker = self.env.get_num_picker()
        # space_low = np.concatenate([pick_lower_bound, place_lower_bound]*self.num_pickers)\
        #     .reshape(self.num_pickers, -1).astype(np.float32)
        # space_high = np.concatenate([pick_upper_bound, place_upper_bound]*self.num_pickers)\
        #     .reshape(self.num_pickers, -1).astype(np.float32)
        # self.action_space = Box(space_low, space_high, dtype=np.float32)
        pick_lower_bound = np.array(pick_lower_bound)
        pick_upper_bound = np.array(pick_upper_bound)
        place_lower_bound = np.array(place_lower_bound)
        place_upper_bound = np.array(place_upper_bound)
        self.action_space = Dict({
            'pick_0': Box(pick_lower_bound, pick_upper_bound, dtype=np.float32),
            'place_0': Box(place_lower_bound, place_upper_bound, dtype=np.float32),
        })
        if not single_operator:
            self.action_space.spaces['pick_1'] = Box(pick_lower_bound, pick_upper_bound, dtype=np.float32)
            self.action_space.spaces['place_1'] = Box(place_lower_bound, place_upper_bound, dtype=np.float32)
        
        ### Each parameters has its class variable
        self.pick_height = pick_height
        self.place_height = place_height

        self.action_step = 0
        self.action_horizon = action_horizon
        self.kwargs = kwargs
        self.action_mode = 'pixel-pick-and-place'
        #self.horizon = self.action_horizon
        #self.logger_name = 'standard_logger' #'pick_and_place_fabric_single_task_logger'
        self.single_operator = single_operator
        self.pregrasp_height = pregrasp_height
        self.pre_grasp_vel = pre_grasp_vel
        self.drag_vel = drag_vel
        self.lift_vel = lift_vel
        


    def get_no_op(self):
        return self.no_op
        
    def sample_random_action(self):
        return self.action_space.sample()

    def get_action_space(self):
        return self.action_space
    
    def get_action_horizon(self):
        return self.action_horizon
    
    def reset(self, env):
        self.action_step = 0
        return self.action_tool.reset(env)
        
    def process(self, action):
        swap = action['swap'] if 'swap' in action else False
        #print('swap:', swap)
        pick_0 = np.asarray(action['pick_0'])
        place_0 = np.asarray(action['place_0'])
        if swap:
            pick_0 = pick_0[::-1]
            place_0 = place_0[::-1]
        pick_0_depth = action['pick_0_d'] if 'pick_0_d' in action else self.camera_height  - self.pick_height
        place_0_depth = action['place_0_d'] if 'place_0_d' in action else self.camera_height  - self.place_height

 
        pick_1 = np.asarray(action['pick_1'])
        place_1 = pick_1.copy()
        if swap:
            pick_1 = pick_1[::-1]
            #place_1 = place_1[::-1]
        pick_1_depth = action['pick_1_d'] if 'pick_1_d' in action else self.camera_height  - self.pick_height
        place_1_depth = self.camera_height - self.pick_height
        action['single_operator'] = False
       
        
        ref_a = np.array([1, -1])
        ref_b = np.array([1, 1])

        if np.linalg.norm(pick_1[:2] - ref_a) > np.linalg.norm(pick_0[:2] - ref_a):
            pick_0, pick_1 = pick_1, pick_0
            place_0, place_1 = place_1, place_0

        action_ = np.concatenate([pick_0, place_0, pick_1, place_1]).reshape(-1, 2)

        depths = np.array([
            pick_0_depth, place_0_depth, 
            pick_1_depth, place_1_depth])

        # action_ = action_ * self.camera_to_world_ratio * depths

        convert_action = np.zeros((4, 3))
        W, H = self.camera_size
        # for i, (a, d )in enumerate(zip(action_, depths)):
        #     # print('a:', a)
        #     # print('d:', d)
        #     convert_action[i] = norm_pixel2world(a, d,
        #         self.camera_intrinsics, self.camera_pose)
        #print('before cnovertionaction:', action_)
        convert_action = norm_pixel2world(
                action_, np.asarray([H, W]),  
                self.camera_intrinsics, self.camera_pose, depths) 
        convert_action = convert_action.reshape(2, 2, 3)
        #print('convert_action in worldspace:', convert_action)

        return {
            'pick_0_position': convert_action[0, 0],
            'place_0_position': convert_action[0, 1],
            'pick_1_position': convert_action[1, 0],
            'place_1_position': convert_action[1, 1],
            'tograsp_vel': self.pre_grasp_vel,
            'drag_vel': self.drag_vel,
            'lift_vel': self.lift_vel,
            'pregrasp_height': self.pregrasp_height,
            'single_operator': action['single_operator']
        }

    ## It accpet action has shape (num_picker, 2, 3), where num_picker can be 1 or 2
    def step(self, env, action):
        #action = action['norm_pixel_pick_and_place']
        self.camera_height = env.camera_height
        # self.camera_to_world_ratio = env.pixel_to_world_ratio
        self.camera_intrinsics = env.camera_intrinsic_matrix
        self.camera_pose = env.camera_extrinsic_matrix
        self.camera_size = env.camera_size

        action_ = self.process(action)
        #print('action_:', action_)
        info = self.action_tool.step(env, action_)
        self.action_step += 1
        info['done'] = self.action_step >= self.action_horizon
        #print(f"Pixel Step: {self.action_step}, Done: {info['done']}")
        return info
        