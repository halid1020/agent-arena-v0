import numpy as np
import time
import cv2

from .utils import check_memory_usage
from .visual_utils import save_video, save_numpy_as_gif

def perform_single(arena, agent, mode='eval', episode_config=None,
    collect_frames=False,
    save_info=False, save_internal_states=False,
    update_agent_from_arena=lambda ag, ar: None,
    env_success_stop=True, policy_terminate=True,
    max_steps=None, debug=False):
    """
        Return dictionary of lists
    """

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
    if save_internal_states:
        internal_states = []
    if save_info:
        information_list = []
    actions = []
    phases = []
    action_time = []
    res['evaluation'] = {}
    
    #arena.set_save_control_step_info(collect_frames)
    frames = None
    if episode_config is not None and episode_config['save_video']:
        frames = []
       
    if max_steps is not None:
        assert max_steps > 0, 'max_steps must be greater than 0'
   
    agent.reset([arena.id]) # reset the agent for the single default arena
    information = arena.reset(episode_config)
    

    ##################################

    information['done'] = False
    if save_info:
        information_list.append(information)
    agent.init([information])

    evals = information['evaluation']


    for k, v in evals.items():
        res['evaluation'][k] = [round(v, 4)]
    res['evaluation']['success'] = [0]

    update_agent_from_arena(agent, arena)

    if episode_config is not None and ('save_goal' in episode_config) and episode_config['save_goal']:
        res['goal'] = arena.get_goal()
        #print('goal keys', res['goal'].keys())

    done = information['done']
    steps = 0
   
    while not done:
        start_time = time.time()
        
        action = agent.single_act(information, update=False)
        steps += 1
        #print('perform action', action)
        phase = agent.get_phase()[0]
        phases.append(phase)
        if save_internal_states:
            internal_states.append(agent.get_state()[arena.id].copy())

        end_time = time.time()
        elapsed_time = (end_time - start_time)
        action_time.append(elapsed_time)
        information = arena.step(action)
        if save_info:
            information_list.append(information)

        check_memory_usage()
        #print('info keys', information.keys())
        
        if episode_config is not None and episode_config['save_video']:
            frame = np.asarray(arena.get_frames())
            if len(frame) != 0:
                #print('frame shape', frame.shape)
                ## resize frames where shorter side 256
                H, W = frame[0].shape[0], frame[0].shape[1]
                if H < W:
                    frame = np.stack([cv2.resize(f, (256 * W // H, 256)) for f in frame])
                else:
                    frame = np.stack([cv2.resize(f, (256, 256 * H // W)) for f in frame])
                #print('frame shape', frame.shape)
                frames.append(frame[:, :, :, :3])
                arena.clear_frames()

        actions.append(action)
        evals = arena.evaluate()
        
        #print('evaluations', evals)
        if debug:
            print('[agent-arena, perform_single] evaluations', evals)
            if frames is not None and len(frame) > 0:
                frames_ = np.concatenate(frames)
                save_video(frames_, path='./tmp', title='perform_single')
                save_numpy_as_gif(frames_, path='./tmp', filename='perform_single')

        agent.update([information], [action])
        
        done = information['done']
        if policy_terminate:
            done |= agent.terminate()[arena.id]
        done = done or (max_steps is not None and steps >= max_steps)
        success = agent.success()[arena.id] or arena.success()
        if env_success_stop:
            done = done or success  
        for k, v in evals.items():
            res['evaluation'][k].append(round(v, 4))
        res['evaluation']['success'].append(int(success)) # convert to int because of json dumping not accepting bool.

       
    res['actions'] = actions #np.stack(actions)
    res['action_durations'] = np.asarray(action_time)
    if save_internal_states:
        internal_states.append(agent.get_state()[arena.id].copy())
    res['phases'] = np.stack(phases)
    if save_info:
        res['information'] = information_list
    if episode_config is not None and episode_config['save_video'] and len(frames) > 0:
        res['frames'] = np.concatenate(frames, axis=0)
    if save_internal_states:
        res['internal_states'] = internal_states
    return res