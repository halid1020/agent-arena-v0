import logging
import numpy as np
import gym
import math

from agent.oracle.rect_fabric.pick_and_place_flattening_policies \
    import OracleTowelPnPFlattening

class OnePickerOneStepFoldingPolicy(OracleTowelPnPFlattening):
    """
        Pick a random coner and put the corner on any point in the area, where the area is defined as the
        arc of the circle with the center at the corner and the radius is the length of the 1.5 time diagonal of the fabric.

        This policy assumes the rectangular fabric is flattened.
    
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.no_op = np.asarray(kwargs['no_op'])
        self.action_space = gym.spaces.Box(
            low=np.asarray(kwargs['action_low']), 
            high=np.asarray(kwargs['action_high']), 
            shape=tuple(kwargs['action_dim']), dtype=np.float32)
        self.fold_steps = 0
        self.phase = 'folding'
    
    def get_phase(self):
        return self.phase

    def success(self, info):
        logging.debug('[oracle, cross folding] largest_particle_distance {}'.format(info['largest_particle_distance']))
        #print('diagonal foldig, info largest_particle_distance {}'.format(info['largest_particle_distance']))
        flg = info['largest_particle_distance'] < 0.005 and self.fold_steps >= 1
        return flg
    
    def reset(self, info=None):
        self.fold_steps = 0

    def _is_point_inside_arc(self, center, radius, start_angle, end_angle, point):
        """
            Check if a point is inside a circle arc.

            Parameters:
            - center: Tuple (x, y) representing the center of the circle.
            - radius: Radius of the circle.
            - start_angle: Start angle of the arc in degrees.
            - end_angle: End angle of the arc in degrees.
            - point: Tuple (x, y) representing the point to be checked.

            Returns:
            - True if the point is inside the arc, False otherwise.
        """

        # Calculate the angle of the point with respect to the center
        angle = math.degrees(math.atan2(point[1] - center[1], point[0] - center[0]))

        # Normalize the angle to be between 0 and 360 degrees
        angle = (angle + 360) % 360

        # Normalize the start and end angles to be between 0 and 360 degrees
        start_angle = (start_angle + 360) % 360
        end_angle = (end_angle + 360) % 360

        if start_angle < end_angle:
            # Standard case where the arc does not span the boundary
            if start_angle <= angle <= end_angle:
                distance = math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
                if distance <= radius:
                    return True
        else:
            # Special case where the arc spans the boundary (e.g., start_angle = 270, end_angle = 30)
            if start_angle <= angle or angle <= end_angle:
                distance = math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
                if distance <= radius:
                    return True

        return False
        
    def act(self, info):
        arena = info['arena']

        
        action = np.clip(
                    self.no_op.astype(float).reshape(*self.action_dim), 
                    self.action_space.low,
                    self.action_space.high)\
        .reshape(self.action_dim[0], 2, -1)[:, :, :2]\
        .reshape(self.action_dim[0], -1)

        # Case 1: After the first step do nothing
        if self.success(info):
            print('success no op')
            self.phase = 'success'
            return np.clip(
                    self.no_op.astype(float).reshape(*self.action_dim), 
                    self.action_space.low,
                    self.action_space.high)
        
        # Case 2: Pick a random corner
        #print('pick a random corner')
        ### Get random sampler from a seed
        self.phase = 'folding'
        seed = arena.get_episode_id()
        self.random_sampler = np.random.RandomState(seed) ## TODO: avoid eval and train overlapping
        
        corner_world_positions = arena.get_corner_positions()
        _, pcp = arena.get_visibility(
                corner_world_positions,
                resolution=(128, 128))
        pcp = pcp[0]
        corner_id = self.random_sampler.choice(len(pcp))
        action[0, :2] = pcp[corner_id]
        
        diagonal_length = np.linalg.norm(pcp[0] - pcp[3])
        arc_radius = 1 * diagonal_length
        arc_center = pcp[corner_id]
        opposite_corner_id = 3-corner_id

        ## Get diagonal's angle wrt to center
        diagonal_angle = math.degrees(
            math.atan2(pcp[opposite_corner_id][1] - arc_center[1], 
            pcp[opposite_corner_id][0] - arc_center[0]))
        
        start_angle = diagonal_angle - 30
        end_angle = diagonal_angle + 30
        


        while True:
            ## generate a random action place_x, place y in the range of -1 to 1
            action[0, 2:] = self.random_sampler.uniform(-1, 1, size=(self.action_dim[0], 2))

            ## check if the action is in the arc
            if self._is_point_inside_arc(
                    arc_center, 
                    arc_radius, 
                    start_angle,
                    end_angle, 
                    action[0, 2:]):
                
                break
        self.fold_steps += 1
        
        return self._process_action(action, arena)
    
    def terminate(self):
        return self.fold_steps >= 1