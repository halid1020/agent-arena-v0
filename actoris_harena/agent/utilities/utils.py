import numpy as np
import math
import time
from tensorboardX import SummaryWriter as SumWriterX


# class SummaryWriter(SumWriterX):
#     def add_scalars(self, scalars):
#         for (name, value, step) in scalars:
#             self.add_scalar(name, value, step)


## TODO: Make the res collection more general.


def normalise_actions(actions, action_lower_bound, action_upper_bound):
        normal_actions = (actions - (action_upper_bound + action_lower_bound)/2) * 2 / (action_upper_bound - action_lower_bound)
        return normal_actions

def denormalise_action(normal_action, action_lower_bound, action_upper_bound):
    action = normal_action * (action_upper_bound - action_lower_bound)/2 +  (action_lower_bound + action_upper_bound)/2
    return action

def preprocess_observation(observation, bit_depth=5, noise=True):
        observation = np.floor(observation/(2 ** (8 - bit_depth)))/(2 ** bit_depth) - 0.5
        if noise:
            observation = observation + (np.random.normal(size=observation.shape)/(2 ** bit_depth))
        
        return observation

def postprocess_observation(observation, bit_depth):
    return (observation + 0.5) * 255.0


def valid_entry(position, shape):
    # Check if the position in the shape range
    for p, s in zip(position, shape):
        if p >= s or p < 0:
            return False
    return True

def find_closest_pickpoint(cloth_mask, action):
    pixel_actions = ((action + 1)/2  * 64).astype(int)
    #cloth_mask = cloth_mask.transpose(0, 1)
    y, x = pixel_actions[0], pixel_actions[1]

    if y == 64:
        y = 63
    if x == 64:
        x = 63
    
    if cloth_mask[x][y]:
        return action
    
    mind = 10000000000
    rx, ry = x, y

    for i in range(64):
        for j in range(64):
            if cloth_mask[i][j] == True:
                d = math.sqrt( (x - i)**2 + (y - j)**2 )
                if d < mind:
                    rx = i
                    ry = j
                    mind = d
    return np.asarray([1.0*ry*2/64 - 1, 1.0*rx* 2/64 - 1])
    

def annotate(image, pixel_positions, size=(3, 3), colour=[0, 0, 255]):
    # Image: H*W*3, np int8, [0, 255].
    # Size: kernel_size
    # Colour: Colour with which to do the annotation.

    H, W = image.shape[0], image.shape[1]
    ret_image = image.copy()

    for p in pixel_positions:
        for dx in range(-size[0]//2, -size[0]//2 + size[0]):
            for dy in range(-size[1]//2, -size[1]//2 + size[1]):
                if valid_entry((p[0]+dx, p[1]+dy), (H, W)):
                    ret_image[p[0]+dx][p[1]+dy] = colour
    
    return ret_image


def denormalise_action(normal_action, action_upper_bound, action_lower_bound):
    action = normal_action * (action_upper_bound - action_lower_bound)/2 +  (action_lower_bound + action_upper_bound)/2
    return action

def normalise_action(action, action_upper_bound, action_lower_bound):
    normal_action = (action - (action_upper_bound + action_lower_bound)/2) * 2 / (action_upper_bound - action_lower_bound)
    return normal_action


def search_shortest_sum_distance_pair(distances):
        M = len(distances[0])
        N = len(distances)
        vis = [False]*M

        def dfs(x, sum_distance, pairs):
            if x == N:
                return pairs.copy(), sum_distance
            ret_pairs = None
            ret_sum_distance = None

            for y in range(M):
                if vis[y]:
                    continue

                vis[y] = True
                pairs[x] = (y, distances[x][y])               
                cur_pairs, cur_sum_distance = dfs(x+1, sum_distance + distances[x][y], pairs)
                if ret_sum_distance == None or cur_sum_distance < ret_sum_distance:
                    ret_sum_distance = cur_sum_distance
                    ret_pairs = cur_pairs
                
                vis[y] = False
            
            return ret_pairs, ret_sum_distance
                
        
        res_pairs, _ = dfs(0, 0, {})
        return res_pairs