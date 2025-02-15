import numpy as np
from time import time
import cv2

import ray
import torch
import agent_arena.api as ag_ar

from .utils import check_memory_usage


def setup_arenas(arena_name, num_processes=16):

    gpu_per_process = torch.cuda.device_count() / num_processes
    print('gpu_per_process', gpu_per_process)
    arenas = [ray.remote(ag_ar.build_arena).options(
        num_gpus=gpu_per_process,
        num_cpus=0.2).remote(f'{arena_name},disp:0', ray=True)
        for _ in range(num_processes)]
    arenas = ray.get(arenas)
    #arenas = [ray.get(e) for e in arenas]
    
    #ray.get([e.setup_ray.remote(e) for e in arenas])

    # os.environ['CUDA_VISIBLE_DEVICES'] = original_visible_devices
    return arenas


def step_arenas(all_arenas, arean_ids, ready_arenas, ready_actions, waiting_infos, 
                wait_num=1, wait_time=0.01):
    # Step 1: Initiate steps for ready arenas
    waiting_infos.extend(
        [e.step.remote(a) for e, a in zip(ready_arenas, ready_actions)])
    
    step_retval = []
    start = time()
    total_time = 0

    # Step 2: Wait for results
    while True:
        if wait_num == -1:
            ready, waiting_infos = ray.wait(waiting_infos, num_returns=len(waiting_infos), timeout=wait_time)
        else:
            ready, waiting_infos = ray.wait(
                waiting_infos, num_returns=wait_num, timeout=wait_time)
        
        if len(ready) == 0:
            continue
        
        step_retval.extend(ready)
        waiting_infos = [info for info in waiting_infos if info not in step_retval]
        
        #waiting_infos = [info for info in waiting_infos if info not in ready]

        total_time = time() - start
        if (total_time > wait_time and len(step_retval) > 0)\
                or len(step_retval) == len(all_arenas):
            break

    # Step 3: Process results
    infos = []
    ready_arenas = []
    
    for info in ray.get(step_retval):
        infos.append(info.copy())
        idx = arean_ids.index(info['arena_id'])
        ready_arenas.append(all_arenas[idx])
    

    return ready_arenas, infos, waiting_infos

def perform_parallel(arenas, agent, 
    mode='eval', episode_configs=None, evaluate=False,
    update_agent_from_arena=lambda ag, ar: None):

    """
        This function performs a parallel episode on multiple arenas.
        It terminates when all arenas are done.
        It returns a list of result dictionaries, one for each arena, where each result dictionary contains the following:
            * 'internal_states': a list of internal states of the agent
            * 'informations': a list of information dictionaries returned by the arenas
            * 'actions': a list of actions taken by the agent
            * 'phases': a list of phases of the agent
            * 'action_durations': a list of time taken for each action
            * 'frames': a list of frames of the arena
            * 'evaluation': a dictionary of evaluation results

    """

    ## 1. Initialise data structures for returning results
    for i, arena in enumerate(arenas):
        arena_id = arena._actor_id.hex()
        print(f"Actor {i} Ray ID: {arena_id}")
        ref = arena.set_id.remote(arena_id)
        ray.get(ref)
    
    arena_ids = ray.get([e.get_id.remote() for e in arenas])

    results = [{} for _ in range(len(arenas))]
    internal_states = [[] for _ in range(len(arenas))]
    informations = [[] for _ in range(len(arenas))]
    actions = [[] for _ in range(len(arenas))]
    phases = [[] for _ in range(len(arenas))]
    action_time = [[] for _ in range(len(arenas))]
    if evaluate:
        for res in results:
            res['evaluation'] = {}
    if episode_configs is None:
        episode_configs = [None] * len(arenas)
    frames = [[] for _ in range(len(arenas))]

    ## 2. Set mode for all arenas and agents
    if mode == 'eval':
        [e.set_eval.remote() for e in arenas]
    elif mode == 'train':
        [e.set_train.remote() for e in arenas]
    elif mode == 'val':
        [e.set_val.remote() for e in arenas]
    else:
        raise ValueError('mode must be either train, eval, or val')
    
    from agent_arena.agent.trainable_agent import TrainableAgent
    if isinstance(agent, TrainableAgent):
        agent.set_train()
        if mode in ['eval', 'val']:
            agent.set_eval()

    ## 3. Reset all arenas and the agent.
    

    waiting_infos = [e.reset.remote(episode_config) for e, episode_config in zip(arenas, episode_configs)]
    ready_infos = ray.get(waiting_infos)
    
    ready_arenas = []
    all_dones = [False] * len(arenas)
    for info in ready_infos:
        arena_id = info['arena_id']
        idx = arena_ids.index(arena_id)
        
        informations[idx].append(info)
        all_dones[idx] = info['done']
    
    ready_arenas = arenas

    #print('arena_ids', ray.get(arena_ids))
    agent.reset(arena_ids)
    agent.init(ready_infos)
    waiting_infos = []

    ## 4. get the evaluation results of the arenas at the beginning
    if evaluate:
        evals = [e.evaluate.remote() for e in arenas]
        for ar, e in zip(arena_ids, ray.get(evals)):
            # print('ar', ar)
            # print('e', e)
            idx = arena_ids.index(ar)
            for k, v in e.items():
                results[idx]['evaluation'][k] = [v]
    
    ## 5. Perform the parallel episode
    total_actions = 0
    total_num_readies = 0
    while not all(all_dones):

        print('#############################')
        print('Iteration!!!!!!')
        
        ready_actions = agent.act(ready_infos)
        total_actions += len(ready_actions)
        #agent.update(ready_infos, ready_actions)
        
        for info, a in zip(ready_infos, ready_actions):
            arena_id = info['arena_id']
            idx = arena_ids.index(arena_id)
            actions[idx].append(a)
        
        ready_arenas, ready_infos, waiting_infos = \
            step_arenas(arenas, arena_ids, ready_arenas, ready_actions, waiting_infos)
        
        total_num_readies += len(ready_arenas)
        
        ## 5.2 Update the results
        agent_phases = agent.get_phase()
        agent_internal_states = agent.get_state()
        for i, e_ in enumerate(ready_infos):
            e = arena_ids.index(e_['arena_id'])
            phases[e].append(agent_phases[e_['arena_id']])#
            internal_states[e].append(agent_internal_states[e_['arena_id']].copy())
            
        for info in ready_infos:
            e = info['arena_id']
            e = arena_ids.index(e)    
            all_dones[e] = info['done']
            if info['done']:
                ready_arenas.remove(arenas[e])
            informations[e].append(info)
        
        print('len(read_infos)', len(ready_infos))
        
        print('all_dones', all_dones)
        
        if evaluate:
            ready_evals = ray.get([e.evaluate.remote() for e in ready_arenas])
            for info, e in zip(ready_infos, ready_evals):
                ar = info['arena_id']
                ar = arena_ids.index(ar)
                for k, v in e.items():
                    results[ar]['evaluation'][k].append(v)
            
        #print('ready_arenas', ready_arenas_id)

        ## 5.2 Save the frames if required TODO
        print('total_actions', total_actions)
        print('total_num_readies', total_num_readies)
        check_memory_usage()

    ## 6. Return the results
    for i, res in enumerate(results):
        res['internal_states'] = internal_states[i]
        res['informations'] = informations[i]
        print('len(informations)', len(informations[i]))
        res['actions'] = actions[i]
        res['phases'] = phases[i]
        res['action_durations'] = action_time[i]
        # if episode_configs[i]['save_video']:
        #     res['frames'] = np.concatenate(frames[i], axis=0)
    
    return results
            
        
