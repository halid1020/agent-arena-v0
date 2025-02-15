## This code is highly referenced from https://github.com/real-stanford/cloth-funnels/blob/main/cloth_funnels/environment/simEnv.py#L1523
import numpy as np

from ...softgym.picker_action_wrappers.world_position_with_velocity_and_grasping_control \
    import WorldPositionWithVelocityAndGraspingControl


## This action primitive should be decoupled from the environment.

class WorldPickAndFling():

    def __init__(self, 
        lowest_cloth_height=0.1,
        max_grasp_dist=0.7,
        stretch_increment_dist=0.02,
        fling_vel=8e-3,
        pregrasp_height=0.3,
        pregrasp_vel=0.06,
        tograsp_vel=0.005,
        prefling_height=0.7,#
        prefling_vel=6e-3,
        fling_pos_y=0.3,
        lift_vel=0.009,
        hang_adjust_vel=0.005,
        stretch_adjust_vel=0.005,
        release_vel=1e-2,
        ready_pos = [[1, 1, 0.6], [1, 1, 0.6]],
        no_cloth_vel = 0.3,
        adaptive_fling_momentum=1,
        lower_height=0.015,
        action_horizon=20,

        **kwargs):

        # Only support 2-picker pick-and-fling
        
        ### Environment has to be WorldPickAndFlingWrapper
        self.action_tool = WorldPositionWithVelocityAndGraspingControl(**kwargs)
        

        #### Define the action space
        self.kwargs = kwargs
        self.action_mode = 'world-pick-and-place'
        self.action_horizon = action_horizon
        self.logger_name = 'standard_logger'


        self.lowest_cloth_height = lowest_cloth_height
        self.max_grasp_dist = max_grasp_dist
        self.stretch_increment_dist = stretch_increment_dist
        self.fling_vel = fling_vel
        self.pregrasp_height = pregrasp_height
        self.pregrasp_velocity = pregrasp_vel
        self.tograsp_velocity = tograsp_vel
        self.prefling_height = prefling_height
        self.prefling_vel = prefling_vel
        self.fling_pos_y = fling_pos_y
        self.ready_pos = np.asarray(ready_pos)
        self.lift_vel = lift_vel
        self.hang_adjust_vel = hang_adjust_vel
        self.stretch_adjust_vel = stretch_adjust_vel
        self.no_cloth_vel = no_cloth_vel
        self.adaptive_fling_momentum = adaptive_fling_momentum
        self.lower_height = lower_height
        self.num_picker = 2
        self.grasping = False
        self.action_step = 0
    
    def get_no_op(self):
        return self.ready_pos
        
    def sample_random_action(self):
        return self.action_space.sample()

    def get_action_space(self):
        return self.action_space
    
    def get_action_horizon(self):
        return self.action_horizon


    def reset(self, env):
        self.action_step = 0
        
        
        return self.reset_pickers(env)
        
    
    def get_step(self):
        return self.action_step
    

    def process(self, action):
       
        return {'pick_0_position': action['pick_0_position'],
                'pick_1_position': action['pick_1_position'],
                'fling_vel': (action['fling_vel'] if 'fling_vel' in action.keys() else self.fling_vel),
                'pregrasp_height': (action['pregrasp_height'] if 'pregrasp_height' in action.keys() else self.pregrasp_height),
                'pregrasp_vel': (action['pregrasp_vel'] if 'pregrasp_vel' in action.keys() else self.pregrasp_velocity),
                'tograsp_vel': (action['tograsp_vel'] if 'tograsp_vel' in action.keys() else self.tograsp_velocity),
                'prefling_height': (action['prefling_height'] if 'prefling_height' in action.keys() else self.prefling_height),
                'prefling_vel': (action['prefling_vel'] if 'prefling_vel' in action.keys() else self.prefling_vel),
                'fling_pos_y': (action['fling_pos_y'] if 'fling_pos_y' in action.keys() else self.fling_pos_y),
                'lift_vel': (action['lift_vel'] if 'lift_vel' in action.keys() else self.lift_vel),
                'hang_adjust_vel': (action['hang_adjust_vel'] if 'hang_adjust_vel' in action.keys() else self.hang_adjust_vel),
                'stretch_adjust_vel': (action['stretch_adjust_vel'] if 'stretch_adjust_vel' in action.keys() else self.stretch_adjust_vel),
                'release_vel': (action['release_vel'] if 'release_vel' in action.keys() else self.release_vel),
                'drag_vel': (action['drag_vel'] if 'drag_vel' in action.keys() else self.drag_vel),
                'lower_height': (action['lower_height'] if 'lower_height' in action.keys() else self.lower_height),
        }
    
    ## expect to recieve action as an dictionary with at least pick_0 and pick_1 positions
    def step(self, env, action):
        action = self.process(action)
        #print('action', action)
        #print('action', action_)
        pick_0_position = action['pick_0_position']
        pick_1_position = action['pick_1_position']
        #self.camera_height = env.camera_height
        #action = action_['action']
        
        
        pick_positions = np.stack([pick_0_position, pick_1_position], axis=0)
        #pick_vels = np.array([action['tograsp_vel']]*self.num_picker).reshape(self.num_picker, -1)
        
        pregrasp_positions = pick_positions.copy()
        pregrasp_positions[:, 2] = action['pregrasp_height']
 
        lift_positions = pick_positions.copy()
        lift_positions[:, 2] = action['prefling_height']
      
        grasp_dist = np.linalg.norm(pick_positions[0, :2] - pick_positions[1, :2])
        prefling_positions = lift_positions.copy()
        prefling_positions[:, 1] = action['fling_pos_y']
        prefling_positions[0, 0] = grasp_dist/2
        prefling_positions[1, 0] = -grasp_dist/2
 
        ## go to pregrasp position
        info = self.action_tool.movep(env, pregrasp_positions, self.no_cloth_vel)
        ## lower the picker
        info = self.action_tool.movep(env, pick_positions, action['tograsp_vel'])
        #print('pick_positions', pick_positions)
        ## pick the object
        info = self.action_tool.grasp(env)
        ## lift the cloth to the prefling height
        info = self.action_tool.movep(env, lift_positions, action['lift_vel'])
        ## go to fling position
        info = self.action_tool.movep(env, prefling_positions, action['prefling_vel'])

        #return self._process_info(info)

        
        ## continue to lift until the the lowest partcile of the cloth is above the minumn height
        hang_height, info = self.hang_cloth(env,
            prefling_positions[0, 2], grasp_dist, action['fling_pos_y'], 
            adjust_vel=action['hang_adjust_vel'])
        
        

        ## stretch the cloth until the cloth is stable
        stretch_dist, info = self.stretch_cloth(env, hang_height, grasp_dist, action['fling_pos_y'], 
            adjust_vel=action['stretch_adjust_vel'], max_grasp_dist=self.max_grasp_dist,
            increment_dist=self.stretch_increment_dist)
        
        info = env.wait_until_stable(max_wait_step=50)

        self.fling(env, stretch_dist, hang_height, lower_height=action['lower_height'],
                   fling_vel=action['fling_vel'], release_vel=action['release_vel'], drag_vel=action['drag_vel'])

        info = self.reset_pickers(env)

        info = env.wait_until_stable()

        self.action_step += 1
        done = self.action_step >= self.action_horizon
        info['done'] = done
        return info
    
    def reset_pickers(self, env):
        
        self.action_tool.movep(env, self.ready_pos, self.no_cloth_vel)
        return self.action_tool.open_gripper(env)
    

    ## ALERT!!! this cannot be done in real world, as it reqruiest the get the information of the cloth
    ## This mean this is a research direction.
    def fling(self, env, grasp_dist, hang_height, lower_height, 
              fling_vel=8e-3, release_vel=1e-2, drag_vel=5e-3):
        cloth_positions = env.get_particle_positions()

        # max height - min height
        cloth_height = np.max(cloth_positions[:, 2]) - np.min(cloth_positions[:, 2])

        # Fling the cloth
        fling_y = cloth_height * 0.5

        back_fling_pos = np.array([
            [grasp_dist/2, fling_y, hang_height],
            [-grasp_dist/2, fling_y, hang_height]
        ])
        front_fling_pos = back_fling_pos.copy()
        front_fling_pos[:, 1] = -fling_y
        
        info = self.action_tool.movep(env, back_fling_pos, fling_vel)
        info = self.action_tool.movep(env, front_fling_pos, fling_vel)
        

        ## Lower the cloth

        release_y = fling_y * 0.9 * self.adaptive_fling_momentum
        drag_y = fling_y * self.adaptive_fling_momentum

        self.action_tool.movep(env, [[grasp_dist/2,  release_y, lower_height],
                    [-grasp_dist/2, release_y, lower_height]], release_vel)
        self.action_tool.movep(env,[[grasp_dist/2, drag_y, lower_height],
                    [-grasp_dist/2, drag_y, lower_height]], drag_vel)
        
        # release the cloth, TODO: refactor
        info = self.action_tool.open_gripper(env)

        # move up the picker a bit
        self.action_tool.movep(env,[[grasp_dist/2, drag_y, lower_height+0.1],
                    [-grasp_dist/2, drag_y, lower_height+0.1]], drag_vel)
        
        return info

    ## ALERT!!! this cannot be done in real world, as it reqruiest the get the information of the cloth
    ## This mean this is a research direction.
    def stretch_cloth(self, env, hang_height, grasp_dist, hang_pos_y, adjust_vel=0.001, increment_dist=0.02, max_grasp_dist=0.7):

        ## get picker positions
        picker_pos = env.get_picker_position()

        mid_pos = (picker_pos[0] + picker_pos[1])/2
        mid_pos[2] = hang_height

        direction = (picker_pos[0] - picker_pos[1])/np.linalg.norm(picker_pos[0] - picker_pos[1])

        hang_positios = np.array([
            [grasp_dist/2, hang_pos_y, hang_height],
            [-grasp_dist/2, hang_pos_y, hang_height]])

        info = self.action_tool.movep(env, hang_positios, adjust_vel)
        cloth_mid_pos = (picker_pos[0] + picker_pos[1])/2
        max_steps = 100
        stretch_steps = 0
        stable_steps = 0
        while True:
            cloth_positions = env.get_particle_positions() # N, [x, y, z]

            high_cloth_positions = cloth_positions[cloth_positions[:, 2] > hang_height-0.1, ...]

            if len(high_cloth_positions) == 0:
                break

            # sort cloth particles regarding to the distance to the mid_pos on x-y plane
            sorted_cloth_positions = high_cloth_positions[np.argsort(np.linalg.norm(high_cloth_positions[:, :2] - mid_pos[:2], axis=1))]
            
            new_cloth_mid_pos = sorted_cloth_positions[0]
            stable = np.linalg.norm(new_cloth_mid_pos - cloth_mid_pos) < 3e-2
            if stable:
                stable_steps += 1
            else:
                stable_steps = 0
            
            stretched = stable_steps > 2
            if stretched or stretch_steps > max_steps:
                break

            stretch_steps += 1
            
            cloth_mid_pos = new_cloth_mid_pos
            grasp_dist += increment_dist

            left_pos = mid_pos - direction*grasp_dist/2
            right_pos = mid_pos + direction*grasp_dist/2

            info = self.action_tool.movep(env, np.stack([right_pos, left_pos]), adjust_vel)

            if grasp_dist > max_grasp_dist:
                break

        return grasp_dist, info


    ## ALERT!!! this cannot be done in real world, as it reqruiest the get the information of the cloth
    ## This mean this is a research direction.
    def hang_cloth(self, env, hang_height, grasp_dist, hang_pos_y, adjust_vel=0.001):
        
        hang_positions = np.array([
            [grasp_dist/2, hang_pos_y, hang_height],
            [-grasp_dist/2, hang_pos_y, hang_height]])

        info = self.action_tool.movep(env, hang_positions, adjust_vel)
        max_steps = 100
        hang_steps = 0 
        while hang_steps > max_steps:
            #print('adjusting height')
            positions = env.get_particle_positions()
            min_z = np.min(positions[:, 2])
            #print('min cloth height', min_z)
            if min_z > self.lowest_cloth_height + 0.05:
                hang_height -= 0.05
            elif min_z < self.lowest_cloth_height - 0.05:
                hang_height += 0.05
            else:
                break
            hang_positions[:, 2] = hang_height
            
            info = self.action_tool.movep(env, hang_positions, adjust_vel)
            hang_steps += 1
        
        return hang_height, info