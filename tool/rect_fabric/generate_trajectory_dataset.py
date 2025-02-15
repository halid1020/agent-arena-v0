"""

For testing the script, simply run

python generate_trajectory_dataset.py

"""


import os

import numpy as np
import h5py
from tqdm import tqdm
import argparse
import ruamel.yaml as yaml
from pathlib import Path
import dotmap
import json

from agent.oracle.mix_policy import MixPolicy
from arena.builder import ArenaBuilder
import api as ag_ar


def main():

    ### Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='mono-square-fabric-almost-flatten-one-picker-mix-policies-heuristic-z-vision-only')
    parser.add_argument('--num_episodes', default=5, type=int)
    parser.add_argument('--save_dir', default='.')
    parser.add_argument('--eval', default=False)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
  
    
    
    config = yaml.safe_load(Path('{}/configuration/data_collection/{}.yaml'.\
                                 format(os.environ['AGENT_ARENA_PATH'], args.config)).read_text())
    config = dotmap.DotMap(config)
    config.num_episodes = args.num_episodes

    # Set random seed
    random_sampler = np.random.RandomState(args.seed)


    flush_every = config.flush_every

    # Environment
    arena = ag_ar.build_arena(config.environment + ',disp:False' + ',seed:{}'.format(args.seed))
    if args.eval:
        print('Set environment to eval.')
        arena.set_eval()
    else:
        arena.set_train()

    # Initialise Policies
    policies = []
    #action_types = []
    policies_ratio = config['policies_ratio']
    policy_names = []
    # action_types = []
    for i in range(len(config['policies'])):
        
        # Mix policy
        if isinstance(config['policies'][i], dotmap.DotMap):
            mix_policies = []
            for sub_policy_name in config['policies']:
                if isinstance(sub_policy_name , dotmap.DotMap):
                    continue
                sub_policy = ag_ar.build_agent(sub_policy_name)
                mix_policies.append(sub_policy)
            
            policies.append(MixPolicy(
                mix_policies, 
                config['policies'][i].policy_weights,
                config['policies'][i].action_dim,
                random_seed=args.seed))
            policy_names.append('mix_policy')
        
        ## Non-mix policy
        else:
            policy = ag_ar.build_agent(config['policies'][i])
            policies.append(policy)
            #action_types.extend(policy.get_action_types())
            policy_names.append(config['policies'][i])


    
    ## Unique  action types
    # action_types = list(set(action_types))
    # action_types.sort()
    json_to_save = {
        # 'action_types': {n: i for i, n in enumerate(action_types)},
        'policies_ratio': policies_ratio,
        'policies': {n: i for i, n in enumerate(policy_names)},
        'save_names': config['save_names'],
        'save_types': config['save_types'],
        'save_dimensions': config['save_dimensions']
    }
    ## Save the action types as json from string to id with format

    ## create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, '{}.json'.format(args.config)), 'w') as f:
        json.dump(json_to_save, f, ensure_ascii=False, indent=4)
    
    ### Data collection parameters
    num_episodes = config['num_episodes']
    
    h5py_file = os.path.join(args.save_dir, 
        '{}:eps_{}_seed_{}_{}.hdf5'.format(
        args.config, args.num_episodes, args.seed, 
        'eval' if args.eval else 'train'))
       

    file_data = {}
    with h5py.File(h5py_file,'w') as f:
        
        for i in range(len(config['save_names'])):
            name = config['save_names'][i]
            type = config['save_types'][i]
            dimension = config['save_dimensions'][i]
            steps = config['save_steps'][i]
            file_data[name] = f.create_dataset(name, (num_episodes, steps, *tuple(dimension)), dtype=type)

        e = 0
        qbar = tqdm(total=num_episodes)
        while (e < config['num_episodes']):
            policy_id = random_sampler.choice(range(len(policies)), p=policies_ratio)
            policy = policies[policy_id]
            #print('Policy: {}'.format(config['policies'][policy_id]))

            eps_data = perform(arena, policy, save_info_keys=config['save_names'], initial_coverage=config['initial_coverage'])

            for i in range(len(config['save_names'])):
                name = config['save_names'][i]
                type = config['save_types'][i]
                dimension = config['save_dimensions'][i]
                steps = config['save_steps'][i]

                if name in eps_data.keys():
                    file_data[name][e:e+1] = eps_data[name].reshape(-1, *dimension)[:steps].astype(type)
                elif name == 'policy':
                    file_data[name][e:e+1] = np.asarray([policy_id]).reshape(steps, *dimension).astype(type)
                elif name == 'success':
                    file_data[name][e:e+1] = np.asarray(arena.success()).reshape(steps, *dimension).astype(type)
                else:
                    raise Exception('Unknown save name {}'.format(name))
                

            e += 1
            qbar.update(1)
            
            if e % flush_every == 0:
                f.flush()

            
        f.flush()
        f.close()

    print('Finish Collection')

def perform(env, policy, eid=None, save_info_keys=None,  initial_coverage=[0, 1]):
    data = {}
    env.set_save_control_step_info(False)
    while True:

        info = env.reset({'eid': eid, 'save_video': False})
        #obs = info['observation']
        policy.reset()
        info_ = env.get_info(save_info_keys)
        policy.init(info)

        if env.get_normalised_coverage() < initial_coverage[0] or env.get_normalised_coverage() > initial_coverage[1]:
            continue
        else:
            break
    
    #observations
    if save_info_keys:
        for key in info_.keys():
            if 'control' in key:
                data[key] = [np.array(info_[key])]
            else:
                data[key] = [np.expand_dims(info_[key], 0)]

    t = 0
    #control_step = 0
    done = False
    while not info['done']:
        
        
        action = policy.act(info)
        info = env.step(action)
        policy.update(action, info)
        info_ = env.get_info(save_info_keys)
    
        if save_info_keys:
            for key in info_.keys():
                if key not in data.keys():
                    data[key] = []
                if 'control' in key:
                    data[key].append(np.array(info_[key]))
                    #print('key {} len {} shape {}'.format(key, len(data[key][-1]), data[key][-1].shape))
                else:
                    data[key].append(np.expand_dims(info_[key], 0))

        if 'action' in save_info_keys:
            if t == 0:
                data['action']= [np.expand_dims(action, 0)]
            else:
                data['action'].append(np.expand_dims(action, 0))

        # if 'action_type' in save_info_keys:
        #     ## Get the index of the action type
        #     idx = action_types.index(policy.get_action_type())

        #     if t == 0:
        #         data['action_type']= [np.expand_dims(idx, 0)]
        #     else:
        #         data['action_type'].append(np.expand_dims(idx, 0))
        
        if 'reward' in save_info_keys:
            if t == 0:
                data['reward']= [np.expand_dims(info['reward'], 0)]
            else:
                data['reward'].append(np.expand_dims(info['reward'], 0))

        if 'action_step' in save_info_keys:
            if t == 0:
                data['action_step']= [np.full(data['control_signal'][-1].shape[0], t)]
            else:
                data['action_step'].append(np.full(data['control_signal'][-1].shape[0], t))
        
        t += 1
        if done:
            break
    
    new_data = {}
    for k, v in data.items():
        new_data[k] = np.concatenate(v, axis=0)
        print('k: {}, v : {}'.format(k, new_data[k].shape))

    return new_data



if __name__ == '__main__':
    main()