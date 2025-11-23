import numpy as np

class WorldPositionWithVelocityAndGraspingControl():

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.grasping = [False, False]

    def step(self, env, actions):
        ## this function requires the 
        ## env to have `get_picker_position` and `step` method
        total_steps = 0
        info = {}
        
        for action in actions:
            #start_time = time.time()
            
            pickers_position = env.get_picker_position()
            #print('picker_pos', pickers_position)
            target_position = action[:, :3].copy()
            #target_position[:, [1, 2]] = target_position[:, [2, 1]]
            velocity = action[:, 3]

            delta = target_position - pickers_position
            #print('delta', delta)
            distance = np.linalg.norm(delta, axis=1)
            num_step = np.ceil(np.max(distance / velocity)).astype(int) + 1

            delta /= num_step
            norm_delta = np.linalg.norm(delta, axis=1, keepdims=True)

            curr_pos = pickers_position.copy()
            
            #print('num sub step', num_step)

            for i in range(num_step):
                #print('!!!! small step')
                dist = np.linalg.norm(target_position - curr_pos, axis=1, keepdims=True)
                mask = dist < norm_delta
                delta = np.where(mask, target_position - curr_pos, delta)
                
                control_signal = np.hstack([delta, action[:, 4:5]])
                #print('control_signal', control_signal)
                info = env.control_picker(control_signal, process_info=False)#, process_info=(i == num_step-1))
                curr_pos += delta
                total_steps += 1
            
            #print(f'action {action} num steps {num_step} took {time.time() - start_time:.6f} seconds')
        
        info['total_control_steps'] = total_steps
        #info = self._process_info(info)
        return info

    
    def movep(self, env, pos, vel):
        grasp_sign = 1 if not self.grasping else -1
        grasp_signs = np.where(self.grasping, -1, 1).reshape(2, 1)
        # TODO grasp_sign = 1 if not self.grasping else -1
        action =  np.concatenate(
            [
                pos, 
                np.array([vel]*2).reshape(2, -1),
                grasp_signs
            ],     
            axis=1
        )
        info = self.step(env, [action])
        return info
    
    def both_grasp(self, env):
        self.grasping[0], self.grasping[1] = True, True
        picker_pos = env.get_picker_position()
        #print('picker_pos before grasp', picker_pos)
        return self.movep(env, picker_pos, 0.1)
    
    def open_both_gripper(self, env):
        self.grasping[0], self.grasping[1] = False, False
        picker_pos = env.get_picker_position()
        return self.movep(env, picker_pos, 0.1)