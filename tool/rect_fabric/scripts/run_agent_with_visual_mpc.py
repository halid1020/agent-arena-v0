import os
import sys
sys.path.insert(0, '..')

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
    parser.add_argument('--setting', default='')
    parser.add_argument('--agent',  default='planet-pick') ### TODO: force it let it have input
    parser.add_argument('--policy', default='')
    parser.add_argument('--hyper_para',  default='')
    args = parser.parse_args()
    if args.policy == '':
        args.policy = args.setting

    
    ### Initilise Project
    cwd = os.getcwd() # get current working directory
    src_index = cwd.find("src") # find the index of "src" in the path
    if src_index != -1: # if "src" is found in the path
        truncated_path = cwd[:src_index+3] # truncate the path at "src"
    else:
        raise Exception("Could not find src directory. Make sure you are running the script from the root of the repository.")
    
    configs = yaml.safe_load(Path('{}/yamls/agents/{}/{}/{}.yaml'.\
        format(truncated_path, args.agent, args.setting, args.hyper_para)).read_text())
    dict_configs = configs
    configs = DotMap(dict_configs)



    
    ### Initialise Agent
    if args.hyper_para == '':
        args.hyper_para = args.setting   

    configs['save_dir'] = os.path.join('tmp', args.setting, args.agent, args.hyper_para)
    configs = DotMap(dict(configs))
    #configs.transform=transform
    print()
    print('Initialising agent {}'.format(args.agent))
    agent = name_to_agent[args.agent](configs)





    ### Initiliase Policy
    policy_params = configs.policy.params
    policy_params['agent'] = agent
    
    
    
    # Environment
    print()
    print('Initialising Environment {}'.format(args.setting))
    env = EnvBuilder.build(args.setting + ',gui:False')
    env.set_eval()
    env_eval_para = env.get_eval_para()
    tiers = env_eval_para['eval_tiers']
    
    

    
    # ### Evaluate Agent on the Environment
    for ph in [2, 1, 4]:
        for candidates in [1000, 100]:
            for iterations in [10, 100]:
                for act in [1.0, 0.5]:
                    policy_params.update({
                        'candidates': candidates,
                        'planning_horizon': ph,
                        'iterations': iterations,
                        'cost_fn': 'L2',
                        'image_dim': (64, 64),
                        'action_lower_bound': -act,
                        'action_upper_bound': act,
                        'no_op': [1.0, 1.0, 1.0, 1.0],
                        'action_dim': (1, 4),
                        'clip': True
                    })

                    policy = PolicyBuilder.build('visual_mpc_cem', policy_params)

                    update_policy_from_env = lambda p, e: p.set_goal_image(e.get_goal_observation())

                    qbar = tqdm(total=sum([len(tiers[k]) for k in tiers.keys()]))
                    written = False
                    for tier in reversed(list(tiers.keys())):
                        for eid in tiers[tier]:
                            collect_frames = False
                            res = perform(env, eid, policy, 
                                        collect_frames=collect_frames, 
                                        update_policy_from_env=update_policy_from_env)
                            if 'z' in args.setting:
                                res['actions'] = res['actions'][:, 0].reshape(-1, 2, 3)[:, :, :2].reshape(-1, 4)

                            manupilation_logger(tier, eid, res, configs, written, filename='visual_mpc_ph_{}_pop_{}_it_{}_act_{}'.format(ph, candidates, iterations, act))
                            written = True
                            qbar.update(1)


if __name__ == '__main__':
    main()