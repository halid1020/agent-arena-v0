import numpy as np
import logging
from gym.spaces import Box
import matplotlib.pyplot as plt

from ...softgym.picker_action_wrappers.world_position_with_velocity_and_grasping_control \
    import WorldPositionWithVelocityAndGraspingControl



class WorldPickAndPlace():

    def __init__(self, 
                 action_horizon=20,
                 pick_lower_bound=[-1, -1, 0],
                 pick_upper_bound=[1, 1, 1],

                 place_lower_bound=[-1, -1, 0],
                 place_upper_bound=[1, 1, 1],

                 ready_pos = [[1, 1, 0.6], [-1, 1, 0.6]],
                 
                 lift_vel=0.05,
                 drag_vel=0.05,
                 tograsp_vel=0.05,

                 **kwargs):
        
        ### Environment has to be WorldPickAndFlingWrapper
        self.action_tool = WorldPositionWithVelocityAndGraspingControl()
       

        #### Define the action space
        self.action_dim = 2
        space_low = np.concatenate([pick_lower_bound, place_lower_bound]*self.action_dim)\
            .reshape(self.action_dim, -1)
        space_high = np.concatenate([pick_upper_bound, place_upper_bound]*self.action_dim)\
            .reshape(self.action_dim, -1)
        self.action_space = Box(space_low, space_high, dtype=np.float64)

        self.ready_pos = np.asarray(ready_pos)

        # if self.no_op.shape[0] == 2:
        #     self.no_op = self.no_op.reshape(2, 2, -1)
        #     self.no_op[1, :, 0] *= -1
        #     if self.no_op.shape[2] == 3:
        #         self.no_op[:, :, 2] = 0.2
        #     self.no_op = self.no_op.reshape(*self.action_space.shape)
        

        ### Each parameters has its class variable
        self.lift_vel = lift_vel
        self.drag_vel = drag_vel
        self.tograsp_vel = tograsp_vel
        self.no_cloth_vel = 0.3

        self.action_step = 0
        self.action_horizon = action_horizon
        self.action_mode = 'world-pick-and-place'
        self.horizon = self.action_horizon
        self.logger_name = 'standard_logger'
    
    def get_no_op(self):
        return self.ready_pos
        
    def sample_random_action(self):
        return self.action_space.sample()

    def get_action_space(self):
        return self.action_space
    
    def get_action_horizon(self):
        return self.action_horizon
    
    def _process_info(self, info):
        info['no_op'] = self.ready_pos
        info['action_space'] = self.action_space
        #info['arena'] = self
        return info

    def reset(self, env):
        self.action_step = 0
        return env.get_info()
    
    def get_step(self):
        return self.action_step
    
    def process(self, action):
        if 'tograsp_vel' not in action:
            action['tograsp_vel'] = self.tograsp_vel
        if 'lift_vel' not in action:
            action['lift_vel'] = self.lift_vel
        if 'drag_vel' not in action:
            action['drag_vel'] = self.drag_vel
        if 'pregrasp_height' not in action:
            action['pregrasp_height'] = self.pregrasp_height
        if 'single_operator' not in action:
            action['single_operator'] = False
        
        return action
    
    def step(self, env, action):
        action = self.process(action)
        self.camera_height = env.camera_height
        pick_positions = np.stack(
            [action['pick_0_position'], action['pick_1_position']]
        )

        place_positions = np.stack(
            [action['place_0_position'], action['place_1_position']]
        )

        pre_pick_positions = pick_positions.copy()
        pre_pick_positions[:, 2] = action['pregrasp_height']

        place_raise = place_positions.copy()
        place_raise[:, 2] = 0.1

        if action['single_operator']:
            pick_positions[1] = self.ready_pos[0, :3]
            pre_pick_positions[1] = self.ready_pos[0, :3]

        self.action_tool.movep(env, pre_pick_positions, self.no_cloth_vel)
        self.action_tool.movep(env, pick_positions, action['tograsp_vel'])
        self.action_tool.grasp(env)
        self.action_tool.movep(env, pre_pick_positions, action['lift_vel'])
        self.action_tool.movep(env, place_positions, action['drag_vel'])
        self.action_tool.open_gripper(env)
        self.action_tool.movep(env, place_raise, action['lift_vel'])
        self.action_tool.open_gripper(env)

        self.action_tool.movep(env, self.ready_pos, self.no_cloth_vel)

        info = env.wait_until_stable()
        
        self.action_step += 1
        info['done'] = self.action_step >= self.action_horizon
        #print(f"World Step: {self.action_step}, Done: {info['done']}")
        return self._process_info(info)