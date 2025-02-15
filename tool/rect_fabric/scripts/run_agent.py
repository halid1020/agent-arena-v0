import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../../../OpenSTL')

from dotmap import DotMap
import ruamel.yaml as yaml
from tqdm import tqdm
from pathlib import Path
import wandb
import argparse

from yamls.settings import *
from utilities.loggers import *
from utilities.utils import perform
from environments.env_builder import EnvBuilder
from agent.policies.builder import PolicyBuilder

def main():

    ### Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', default='pick-and-place-cloth-flattening-old')
    parser.add_argument('--agent',  default='default') ### TODO: force it let it have input
    parser.add_argument('--policy', default='')
    parser.add_argument('--hyper_para',  default='')
    parser.add_argument('--log_dir', default='/data/planet-pick-project')
    parser.add_argument('--eval_episodes', default=30, type=int)
    parser.add_argument('--save_file', default='manupulation')

    args = parser.parse_args()
    if args.policy == '':
        args.policy = args.setting

    cwd = os.getcwd() # get current working directory
    src_index = cwd.find("src") # find the index of "src" in the path
    if src_index != -1: # if "src" is found in the path
        truncated_path = cwd[:src_index+3] # truncate the path at "src"
    else:
        raise Exception("Could not find src directory. Make sure you are running the script from the root of the repository.")
    

    ### Initilise Project
    configs = yaml.safe_load(Path('{}/yamls/agents/{}/{}/{}.yaml'.\
        format(truncated_path, args.agent, args.setting, args.hyper_para)).read_text())
    dict_configs = configs
    configs = DotMap(dict_configs)



    ### Initiliase datasets and transform
    datasets = {}
    if 'datasets' in dict_configs:
        for dataset_dict in dict_configs['datasets']:
            key = dataset_dict['key']
            print()
            print('Initialising dataset {} from name {}'.format(key, dataset_dict['name']))

            dataset_params = yaml.safe_load(
                Path('{}/yamls/datasets/{}.yaml'.\
                    format(truncated_path, dataset_dict['name'])).read_text())
            dataset_params.update(dataset_dict['params'])
            
            transform_params = DotMap(dataset_dict['transform']['params'])
            transform = name_to_transformer[dataset_dict['transform']['name']](transform_params)
            dataset_params['transform'] = transform
            
            
            
            dataset = name_to_dataset[dataset_dict['name']](
                **dataset_params)

            datasets[key] = dataset
        

    ### Initialise Agent
    if args.hyper_para == '':
        args.hyper_para = args.setting   

    configs['save_dir'] = os.path.join(args.log_dir, args.setting, args.agent, args.hyper_para)
    configs = DotMap(dict(configs))
    #configs.transform=transform
    print()
    print('Initialising agent {}'.format(args.agent))
    agent = name_to_agent[args.agent](configs)

    # Environment
    print()
    print('Initialising Environment {}'.format(args.setting))
    env = EnvBuilder.build(args.setting + ',gui:False')


    ### Train Agent
    print()
    print('load agent {}'.format(args.agent))
    agent.load_model()
    print('Finished training Agent {}'.format(args.agent))


    ### Initiliase Policy
    policy_params = configs.policy.params
    if configs.policy.name == 'self':
        policy = agent
    else:
        policy_params['agent'] = agent
        policy = PolicyBuilder.build(configs.policy.name, policy_params)


    # Environment
    print()
    print('Initialising Environment {}'.format(args.setting))
    env.set_eval()
        
    # ### Evaluate Agent on the Environment
    print()
    print('Running Agent {} on Environment {}'.format(args.agent, args.setting))
    env_eval_para = env.get_eval_para()

    if args.eval_episodes is None:
        tiers = env_eval_para['eval_tiers']

        qbar = tqdm(total=sum([len(tiers[k]) for k in tiers.keys()]))
        written = False
        for tier in reversed(list(tiers.keys())):
            for eid in tiers[tier]:   
                collect_frames = (eid in env_eval_para['video_episodes'])
                save_frames = (eid in env_eval_para['video_episodes'][:3])
                res = perform(env, eid, policy, collect_frames=collect_frames)

                if 'z' in args.setting:
                    res['actions'] = res['actions'][:, 0].reshape(-1, 2, 3)[:, :, :2].reshape(-1, 4)

                manupilation_logger(tier, eid, res, configs, 
                                    written, env_eval_para['video_episodes'],
                                    save_frames=save_frames,
                                    filename=args.save_file)
                written = True
                qbar.update(1)
    else:
        written = False
        action_durations = []
        for eid in tqdm(range(args.eval_episodes)): 
            # collect_frames = (eid in env_eval_para['video_episodes'])
            # save_frames = (eid in env_eval_para['video_episodes'][:3])
            res = perform(env, eid, policy)
            action_durations.extend(res['action_durations'])

            if 'z' in args.setting:
                res['actions'] = res['actions'][:, 0].reshape(-1, 2, 3)[:, :, :2].reshape(-1, 4)

            manupilation_logger(0, eid, res, configs, 
                                written, env_eval_para['video_episodes'],
                                save_frames=False,
                                filename=args.save_file)
            written = True
        
        print('Average action duration: {}'.format(np.mean(action_durations)))
        print('Std action duration: {}'.format(np.std(action_durations)))


if __name__ == '__main__':
    main()