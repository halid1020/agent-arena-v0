import numpy as np
import math
import time
import os
import sys
import logging
import psutil
import cv2

from scipy.ndimage import distance_transform_edt, sobel

from tensorboardX import SummaryWriter as SumWriterX

class TrainWriter(SumWriterX):
    def add_scalars(self, scalars):
        for (name, value, step) in scalars:
            self.add_scalar(name, value, step)
            
def print_dict_tree(dictionary, indent='', is_last=True):
    for index, (key, value) in enumerate(dictionary.items()):
        connector = '└── ' if is_last and index == len(dictionary) - 1 else '├── '
        print(f"{indent}{connector}{key}")
        
        if isinstance(value, dict):
            new_indent = indent + ('    ' if is_last and index == len(dictionary) - 1 else '│   ')
            print_dict_tree(value, new_indent, index == len(dictionary) - 1)
        else:
            value_connector = '└── ' if index == len(dictionary) - 1 else '├── '
            print(f"{indent}{'│   ' if not is_last else '    '}{value_connector}{value}")


def adjust_points(points, mask, min_distance=2):
    """
    Adjust points to be at least min_distance pixels away from the mask border.
    
    :param points: List of (x, y) coordinates
    :param mask: 2D numpy array where 0 is background and 1 is foreground
    :param min_distance: Minimum distance from the border (default: 2)
    :return: List of adjusted (x, y) coordinates
    """
    # from matplotlib import pyplot as plt
    
    # plt.imshow(mask.astype(np.float32))
    # plt.savefig('tmp/mask.png')

    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)

    if np.sum(mask) == 0:
        return points, mask
    
    # Compute distance transform
    dist_transform = distance_transform_edt(mask)
    
    # Create a new mask where pixels < min_distance from border are 0
    eroded_mask = (dist_transform >= min_distance).astype(np.uint8)
    
   
    # plt.imshow(eroded_mask.astype(np.float32))
    # plt.savefig('tmp/eroded_mask.png')

    adjusted_points = []
    for x, y in points:
        #print('x, y', x, y)
        if eroded_mask[x, y] == 0:  # If point is too close to border
            # Find the nearest valid point
            x_indices, y_indices = np.where(eroded_mask == 1)
            distances = np.sqrt((x - x_indices)**2 + (y - y_indices)**2)
            nearest_index = np.argmin(distances)
            new_x, new_y = x_indices[nearest_index], y_indices[nearest_index]
            adjusted_points.append((new_x, new_y))
        else:
            adjusted_points.append((x, y))
    
    return adjusted_points, eroded_mask

def perform(arena, agent, mode='eval', episode_config=None,
    collect_frames=False,
    update_agent_from_arena=lambda ag, ar: None):

    print('save video', episode_config['save_video'])

    if mode == 'eval':
        arena.set_eval()
    elif mode == 'train':
        arena.set_train()
    elif mode == 'val':
        arena.set_val()
    else:
        raise ValueError('mode must be either train, eval, or val')
    
    from ..agent.trainable_agent import TrainableAgent
    if isinstance(agent, TrainableAgent):
        agent.set_train()
        if mode in ['eval', 'val']:
            agent.set_eval()

    res = {}
    rgbs = []
    depths = []
    internal_states = []
    informations = []
    #rewards = []
    actions = []
    phases = []
    action_time = []
    res['evaluation'] = {}
    
    #arena.set_save_control_step_info(collect_frames)
    if episode_config['save_video']:
        frames = []
       

    #total_reward = 0
    # done = False

    logging.info('[ag_ar.perform] ########################################################')
    logging.info('[ag_ar.perform] ########################################################')

    eid = episode_config['eid']
    agent.reset()
    information = arena.reset(episode_config)
    
    # ## Debugging #################
    # import matplotlib.pyplot as plt
    # key_points = arena.get_keypoint_positions()

    # vis_key_points, project_pos = arena.get_visibility(key_points)
    # project_pos = project_pos[0]
    # print('vis_key_points', vis_key_points)
    # # print('project_pos', project_pos)

    # ### project pos is in shape N*2, where each entry is between [-1, 1] pixel space
    # ### draw the key points on rgb image

    # rgb = information['observation']['rgb']
    # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    # H, W = rgb.shape[:2]
    # print('H, W', H, W)
    # for i in range(len(project_pos)):
    #     x, y = project_pos[i][0], project_pos[i][1]
    #     # print('x, y', x, y)
    #     x = int((x + 1) * W / 2)
    #     y = int((y + 1) * H / 2)
    #     cv2.circle(rgb, (x, y), 1, (0, 255, 0), -1)
    # cv2.imwrite('tmp/key_points.png', rgb)


    ##################################

    information['done'] = False
    informations.append(information)
    agent.init(information)
    #internal_states.append(agent.get_state().copy())
    #information['reward'] = 0
    evals = arena.evaluate()

    #import cv2
    #cv2.imwrite('test.png', information['observation']['rgb'])


    logging.info('[ag_ar.perform] Start Episode {}'.format(eid))
    logging.info('[ag_ar.perform] evaluations: {}'.format(evals))

    for k, v in evals.items():
        res['evaluation'][k] = [v]
    
    # rgbs.append(information['observation']['rgb'])
    

    update_agent_from_arena(agent, arena)

    #agent.init_state(information)

    if ('save_goal' in episode_config) and episode_config['save_goal']:
        res['goal'] = arena.get_goal()
        #print('goal keys', res['goal'].keys())
   
    while not information['done']:
        start_time = time.time()
        
        action = agent.act(information)
        #print('perform action', action)
        phase = agent.get_phase()
        phases.append(phase)
        internal_states.append(agent.get_state().copy())

        end_time = time.time()
        elapsed_time = (end_time - start_time)
        action_time.append(elapsed_time)
        information = arena.step(action)
        informations.append(information)

        check_memory_usage()
        #print('info keys', information.keys())
        
        if episode_config['save_video']:
            frame = np.asarray(arena.get_frames())
            ## resize frames to 256x256
            frame = np.stack([cv2.resize(f, (256, 256)) for f in frame])
            #print('frame shape', frame.shape)
            frames.append(frame[:, :, :, :3])
            arena.clear_frames()

        #print('action', action)
        actions.append(action)
        #rewards.append(information['reward'])
        #total_reward += information['reward']
        # rgbs.append(information['observation']['rgb'])
        #depths.append(external_state['observation']['depth'])
        evals = arena.evaluate()
        
        if 'normalised_coverage' in evals:
            print('evals', evals['normalised_coverage'])
        

        # logging.info('\n[ag_ar.perform] Step {}'.format(len(actions)))
        # logging.info('[ag_ar.perform] actions: {}'.format(action))
        # logging.info('[ag_ar.perform] Evaluations: {}'.format(evals))
        
        agent.update(information, action)
        
        information['done'] = information['done'] or agent.success() or arena.success() or agent.terminate()
        if not information['done']:
            logging.info('[ag-ar.perform] not finished, continue to next step ...')

        for k, v in evals.items():
            res['evaluation'][k].append(v)

       

    #res['return'] = total_reward
    # res['rgb'] = np.stack(rgbs)
    #res['depth'] = np.stack(depths)
    #res['rewards'] = np.asarray(rewards)
    res['actions'] = actions #np.stack(actions)
    res['action_durations'] = np.asarray(action_time)
    internal_states.append(agent.get_state().copy())
    res['phases'] = np.stack(phases)
    res['informations'] = informations
    if episode_config['save_video']:
        res['frames'] = np.concatenate(frames, axis=0)
    res['internal_states'] = internal_states
    return res

def check_memory_usage():
    # Get the current memory usage
    memory_percent = psutil.virtual_memory().percent

    # Print the current memory usage (optional)
    #print(f"Current memory usage: {memory_percent}%")

    # Check if memory usage is above 90%
    if memory_percent > 90:
        print("Memory usage is above 90%. Stopping the program.")
        # You can add additional cleanup or logging steps here if needed
        exit()


def create_message_logger(path, verbose):
    verbose2level = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    if verbose == 'silence':
        pass 
    else:
        os.makedirs(path, exist_ok=True)
        logging.basicConfig(
            level=verbose2level[verbose],
            format="[%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(path, "output.log")), 
                logging.StreamHandler(sys.stdout)],
        )