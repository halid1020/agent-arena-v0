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

def save_checkpoint(e, args, h5py_file):
    checkpoint = {
        'episode': e,
        'args': vars(args),
        'h5py_file': h5py_file
    }
    checkpoint_file = os.path.join(args.save_dir, \
        f'{args.config}_seed_{args.seed}_eps_{args.num_episodes}_checkpoint.json')
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f)

def load_checkpoint(args):
    checkpoint_file = os.path.join(args.save_dir, \
        f'{args.config}_seed_{args.seed}_eps_{args.num_episodes}_checkpoint.json')
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint
    return None

def perform(env, policy, eid=None, save_info_keys=None, initial_coverage=[0, 1]):
    data = {}
    env.set_save_control_step_info(False)
    while True:
        info = env.reset({'eid': eid, 'save_video': False})
        policy.reset()
        info_ = env.get_info(save_info_keys)
        policy.init(info)
        if env.get_normalised_coverage() < initial_coverage[0] or env.get_normalised_coverage() > initial_coverage[1]:
            continue
        else:
            break

    if save_info_keys:
        for key in info_.keys():
            if 'control' in key:
                data[key] = [np.array(info_[key])]
            else:
                data[key] = [np.expand_dims(info_[key], 0)]

    t = 0
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
                else:
                    data[key].append(np.expand_dims(info_[key], 0))
                    
                    # from matplotlib import pyplot as plt
                    # if key == 'depth':
                    #     print('max', np.max(info_[key]))
                    #     print('min', np.min(info_[key]))
                    #     plt.imshow(info_[key])
                    #     plt.savefig(f'tmp/col_depth.png')
                    #     plt.close()
                    # if key == 'mask':
                    #     plt.imshow(info_[key])
                    #     plt.savefig(f'tmp/col_mask.png')
                    #     plt.close()
                    


        if 'action' in save_info_keys:
            if t == 0:
                data['action'] = [np.expand_dims(action, 0)]
            else:
                data['action'].append(np.expand_dims(action, 0))

        if 'reward' in save_info_keys:
            if t == 0:
                data['reward'] = [np.expand_dims(info['reward'], 0)]
            else:
                data['reward'].append(np.expand_dims(info['reward'], 0))

        if 'action_step' in save_info_keys:
            if t == 0:
                data['action_step'] = [np.full(data['control_signal'][-1].shape[0], t)]
            else:
                data['action_step'].append(np.full(data['control_signal'][-1].shape[0], t))

        t += 1
        if done:
            break

    new_data = {}
    for k, v in data.items():
        new_data[k] = np.concatenate(v, axis=0)
    return new_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='mono-square-fabric-almost-flatten-one-picker-mix-policies-heuristic-z-vision-only')
    parser.add_argument('--num_episodes', default=5, type=int)
    parser.add_argument('--save_dir', default='.')
    parser.add_argument('--eval', default=False)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    config = yaml.safe_load(Path('{}/configuration/data_collection/{}.yaml'.format(os.environ['AGENT_ARENA_PATH'], args.config)).read_text())
    config = dotmap.DotMap(config)
    config.num_episodes = args.num_episodes

    random_sampler = np.random.RandomState(args.seed)
    flush_every = config.flush_every

    arena = ag_ar.build_arena(config.environment + ',disp:False' + ',seed:{}'.format(args.seed))
    if args.eval:
        print('Set environment to eval.')
        arena.set_eval()
    else:
        arena.set_train()

    policies = []
    policies_ratio = config['policies_ratio']
    policy_names = []

    for i in range(len(config['policies'])):
        if isinstance(config['policies'][i], dotmap.DotMap):
            mix_policies = []
            for sub_policy_name in config['policies']:
                if isinstance(sub_policy_name, dotmap.DotMap):
                    continue
                sub_policy = ag_ar.build_agent(sub_policy_name)
                mix_policies.append(sub_policy)
            policies.append(MixPolicy(mix_policies, config['policies'][i].policy_weights, config['policies'][i].action_dim, random_seed=args.seed))
            policy_names.append('mix_policy')
        else:
            policy = ag_ar.build_agent(config['policies'][i])
            policies.append(policy)
            policy_names.append(config['policies'][i])

    json_to_save = {
        'policies_ratio': policies_ratio,
        'policies': {n: i for i, n in enumerate(policy_names)},
        'save_names': config['save_names'],
        'save_types': config['save_types'],
        'save_dimensions': config['save_dimensions']
    }

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, '{}.json'.format(args.config)), 'w') as f:
        json.dump(json_to_save, f, ensure_ascii=False, indent=4)

    checkpoint = load_checkpoint(args)
    if checkpoint:
        start_episode = checkpoint['episode']
        h5py_file = checkpoint['h5py_file']
        print(f"Resuming from episode {start_episode}")
    else:
        start_episode = 0
        h5py_file = os.path.join(args.save_dir, '{}:eps_{}_seed_{}_{}.hdf5'.format(args.config, args.num_episodes, args.seed, 'eval' if args.eval else 'train'))

    # Verify if the HDF5 file is corrupted
    try:
        with h5py.File(h5py_file, 'a') as f:
            pass
    except OSError:
        print(f"File {h5py_file} is corrupted. Creating a new file.")
        os.remove(h5py_file)
        with h5py.File(h5py_file, 'w') as f:
            pass

    with h5py.File(h5py_file, 'a') as f:
        file_data = {}
        for i in range(len(config['save_names'])):
            name = config['save_names'][i]
            dtype = config['save_types'][i]
            dimension = config['save_dimensions'][i]
            steps = config['save_steps'][i]
            if name not in f:
                file_data[name] = f.create_dataset(name, (args.num_episodes, steps, *tuple(dimension)), dtype=dtype)
            else:
                file_data[name] = f[name]

        e = start_episode
        qbar = tqdm(total=args.num_episodes, initial=start_episode)

        while e < config['num_episodes']:
            policy_id = random_sampler.choice(range(len(policies)), p=policies_ratio)
            policy = policies[policy_id]
            eps_data = perform(arena, policy, save_info_keys=config['save_names'], initial_coverage=config['initial_coverage'])

            for i in range(len(config['save_names'])):
                name = config['save_names'][i]
                dtype = config['save_types'][i]
                dimension = config['save_dimensions'][i]
                steps = config['save_steps'][i]
                if name == 'depth':
                    print(e, eps_data[name].shape)
                if name in eps_data.keys():
                    file_data[name][e:e+1] = eps_data[name].reshape(-1, *dimension)[:steps].astype(dtype)
                elif name == 'policy':
                    file_data[name][e:e+1] = np.asarray([policy_id]).reshape(steps, *dimension).astype(dtype)
                elif name == 'success':
                    file_data[name][e:e+1] = np.asarray(arena.success()).reshape(steps, *dimension).astype(dtype)
                else:
                    raise Exception('Unknown save name {}'.format(name))

            e += 1
            qbar.update(1)

            if e % flush_every == 0:
                f.flush()
                save_checkpoint(e, args, h5py_file)

        f.flush()
        save_checkpoint(e, args, h5py_file)
    os.remove(os.path.join(args.save_dir, f'{args.config}_checkpoint.json'))
    print('Finish Collection')

if __name__ == '__main__':
    main()
