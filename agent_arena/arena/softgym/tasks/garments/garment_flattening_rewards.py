
from .utils import deformable_distance
import numpy as np

def normalised_coverage_reward(last_info, action, info):
    """
        The hueristic used in VCD and Huang's MEDOR.
    """
    return info['evaluation']['normalised_coverage'] # 0-1

def coverage_differance_reward(last_info, action, info):
    """
        FlingBot uses this reward function.
    """
    if last_info is None:
        last_info = info
    return info['evaluation']['normalised_coverage'] - last_info['evaluation']['normalised_coverage']#

def max_IoU_reward(last_info, action, info):
    """
        The reward function used in the original implementation.
    """
    return info['evaluation']['max_IoU'] # 0-1

def canon_IoU_reward(last_info, action, info):
    """
        The reward function used in the original implementation.
    """
    return info['evaluation']['canon_IoU'] # 0-1

def max_IoU_differance_reward(last_info, action, info):
    """
        The reward function used in the original implementation.
    """
    if last_info is None:
        last_info = info
    return info['evaluation']['max_IoU'] - last_info['evaluation']['max_IoU'] # -1 to 1

def canon_IoU_differance_reward(last_info, action, info):
    """
        The reward function used in the original implementation.
    """
    if last_info is None:
        last_info = info
    return info['evaluation']['canon_IoU'] - last_info['evaluation']['canon_IoU'] # -1 to 1

def canon_l2_tanh_reward(last_info, action, info):
    cur_pos = info['observation']['particle_position'][:, :2]
    goal_pos = info['goal']['particle_position'][:, :2]
    flipped_goal_pos = goal_pos.copy()
    flipped_goal_pos[:, 0] =  -1 * flipped_goal_pos[:, 0]


    distances = np.linalg.norm(cur_pos - goal_pos, axis=1)
    l2_distance = np.mean(distances)
    

    flipped_distances = np.linalg.norm(cur_pos - flipped_goal_pos, axis=1)
    flipped_l2_distance = np.mean(flipped_distances)

    min_distance = min(l2_distance, flipped_l2_distance)
    #print('l2_distance', min_distance)

    return 1 - np.tanh(min_distance) # 0-1

def planet_clothpick_hueristic_reward(last_info, action, info):
    misgrasping_threshold = 1.0
    misgrasping_penalty = -0.5
    penalise_action_threshold = 0.7
    extreme_action_penalty = -0.5
    unflatten_threshold = 0.98
    unflatten_penalty = -0.5
    flattening_threshold = 0.98
    flatten_bonus = 0.5

    if last_info is None:
        last_info = info
    
    if action is None:
        action = np.zeros(4)
    else:
        action = action['norm_pixel_pick_and_place']
        action = np.stack([action['pick_0'], action['place_0']])

    reward = info['evaluation']['normalised_coverage'] - last_info['evaluation']['normalised_coverage']
    if reward < 1e-4 and info['evaluation']['normalised_coverage'] < misgrasping_threshold:
        reward = misgrasping_penalty
    
    if np.max(np.abs(action)) > penalise_action_threshold:
        reward = extreme_action_penalty

    if last_info['evaluation']['normalised_coverage'] > unflatten_threshold \
        and info['evaluation']['normalised_coverage'] < unflatten_threshold:
        reward = unflatten_penalty

    if info['evaluation']['normalised_coverage'] > flattening_threshold:
        reward  = flatten_bonus
    
    return reward

def clothfunnel_reward(last_info, action, info):
    """
        Borrow the code from the original implementation.
    """
    DELTA_WEIGHTED_REWARDS_MEAN = -0.0018245290312917787
    DELTA_WEIGHTED_REWARDS_STD = 0.072
    DELTA_L2_STD = 0.019922712535836946
    DELTA_POINTWISE_REWARDS_STD = 0.12881897698788683

    if last_info is None:
        last_info = info
    arena = info['arena']
    # pre_pos = last_info['observation']['particle_position']
    # post_pos = info['observation']['particle_position']
    # cloth_area = arena.get_cloth_area()
    # init_pos = arena.get_c_particle_position()
    deformable_weight = 0.65
    
    pre_deform_distance = last_info['evaluation']['deform_l2_distance']
    pre_rigid_distance = last_info['evaluation']['rigid_l2_distance']
    post_deform_distance = info['evaluation']['deform_l2_distance']
    post_rigid_distance = info['evaluation']['rigid_l2_distance']
    pre_l2_distance = last_info['evaluation']['canon_l2_distance']
    post_l2_distance = info['evaluation']['canon_l2_distance']
    # print('cloth_area', cloth_area)
    # pre_weighted_distance, pre_deform_distance, pre_rigid_distance, pre_l2_distance, _ = \
    #     deformable_distance(
    #         init_pos, pre_pos, 
    #         cloth_area, deformable_weight)
    
    # post_weighted_distance, post_deform_distance, post_rigid_distance, post_l2_distance, _ = \
    #     deformable_distance(
    #         init_pos, post_pos, 
    #         cloth_area, deformable_weight)
    
    delta_deform_distance = pre_deform_distance - post_deform_distance
    delta_rigid_distance = pre_rigid_distance - post_rigid_distance
    delta_l2_distance = pre_l2_distance - post_l2_distance

    deform_reward = delta_deform_distance / DELTA_WEIGHTED_REWARDS_STD
    rigid_reward = delta_rigid_distance / DELTA_WEIGHTED_REWARDS_STD
    l2_reward = delta_l2_distance / DELTA_L2_STD
    weighted_reward = deform_reward * deformable_weight + rigid_reward * (1 - deformable_weight)
    weighted_reward = np.clip(weighted_reward, -1, 1) ### Halid Added this line.

    tanh_deform_reward = 1 - (np.tanh(-delta_deform_distance) + 1)/2
    tanh_rigid_reward = 1 - (np.tanh(-delta_rigid_distance) + 1)/2#
    tanh_weighted_reward = tanh_deform_reward * deformable_weight + tanh_rigid_reward * (1 - deformable_weight)

    return weighted_reward, tanh_weighted_reward



def learningTounfold_reward(last_info, actino, info):
    """
        How the theta_p is calculated is unknown.
        We use principle component analysis to calculate the orientation difference between the current and goal positions.
        we make alpha as 1, and get rid of vertical facing orientation in our implementation.
    """
    lam = 0.55
    alpha = 1.0

    theta_p = calculate_orientation_difference(
        info['observation']['particle_position'], #
        info['goal']['particle_position'])
    
    theta_f = 0 # Unknown how to calculate
    theta = alpha*theta_p + (1-alpha)*theta_f
    reward_c = info['evaluation']['normalised_coverage']
    # reward_o is 1 - tanh^2(theta)
    reward_o = 1 - np.tanh(theta)**2
    reward = lam*reward_c + (1-lam)*reward_o
    return reward # 0-1

def speedFolding_reward(last_info, action, info):
    """
        In the original paper, it used a pretrained smoothness classifier to calculate the smoothness of the folding.
        Here, we use the max IoU to approximate the smoothness.
    """
    if last_info is None:
        last_info = info
    delta_coverage = info['evaluation']['normalised_coverage'] - last_info['evaluation']['normalised_coverage'] # -1 to 1

    smoothness = info['evaluation']['max_IoU'] - last_info['evaluation']['max_IoU'] # -1 to 1

    alpha = 2
    beta = 1

    return max(np.tanh(alpha*delta_coverage + beta*smoothness), 0) # 0-1

def coverage_alignment_reward(last_info, action, info):

    if last_info is None:
        last_info = info
    r_cs = speedFolding_reward(last_info, action, info)
    dc = info['evaluation']['normalised_coverage'] - last_info['evaluation']['normalised_coverage']
    ds = info['evaluation']['max_IoU'] - last_info['evaluation']['max_IoU']
    nc = info['evaluation']['normalised_coverage']
    iou = info['evaluation']['max_IoU']
    epsilon_c = 1e-4
    epsilon_s = 1e-4
    max_c = 0.99
    max_iou = 0.85
    
    if dc < epsilon_c and ds < epsilon_s and nc < max_c and iou < max_iou:
        return 0
    
    if nc >= max_c and iou >= max_iou:
        return 1
    
    return r_cs






def calculate_orientation_difference(cur_pos, goal_pos):
    # Calculate the center of mass for current and goal positions
    cur_center = np.mean(cur_pos[:, :2], axis=0)
    goal_center = np.mean(goal_pos[:, :2], axis=0)

    # Center the positions
    cur_centered = cur_pos[:, :2] - cur_center
    goal_centered = goal_pos[:, :2] - goal_center

    # Calculate the covariance matrices
    cur_cov = np.cov(cur_centered.T)
    goal_cov = np.cov(goal_centered.T)

    # Calculate the principal axes (eigenvectors)
    _, cur_evecs = np.linalg.eig(cur_cov)
    _, goal_evecs = np.linalg.eig(goal_cov)

    # Get the primary axis (first eigenvector) for current and goal
    cur_axis = cur_evecs[:, 0]
    goal_axis = goal_evecs[:, 0]

    # Calculate the angle between the primary axes
    dot_product = np.dot(cur_axis, goal_axis)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Convert angle to degrees
    angle_degrees = np.degrees(angle)

    return angle_degrees