import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')


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
    parser.add_argument('--log_dir', default='/data/planet-pick-project')
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

    configs['save_dir'] = os.path.join(args.log_dir, args.setting, args.agent, args.hyper_para)
    configs = DotMap(dict(configs))
    #configs.transform=transform
    print()
    print('Initialising agent {}'.format(args.agent))
    agent = name_to_agent[args.agent](configs)

    # ### Train Agent
    # print()
    # print('Training agent {}'.format(args.agent))
    # agent.train(datasets, loss_logger, eval_logger)
    # print('Finished training Agent {}'.format(args.agent))

    # ### Visualise Agent
    # print()
    # print('Visualising Agent {}'.format(args.agent))
    # agent.visualise(datasets)

    
    
    # Environment
    print()
    print('Initialising Environment {}'.format(args.setting))
    env = EnvBuilder.build(args.setting + ',gui:False')
    env.set_eval()
    env_eval_para = env.get_eval_para()
    tiers = env_eval_para['eval_tiers']
    

    
    # ### Evaluate Agent on the Environment
    for action_bound in [1]:
        for ph in [1, 2]:
            for pop in [5000, 1000]:
                for it in [100, 10]:
                    for th in [0.99, 1.0, 0.95, 0.97] :
                        for clip in [True, False]:
                            for cloth_mask in ['from_model', 'from_env']:
                                if cloth_mask == 'from_model':
                                    for cth in [0.45, 0.48, 0.5]:

                                save_file = 'cloth_mask_mpc_action_{}_ph_{}_pop_{}_it_{}_flatten_{}_clip_{}_{}'.\
                                                                format(action_bound, ph, pop, it, th, clip, cloth_mask)
                                if os.path.exists(os.path.join(configs.save_dir, save_file)):
                                    continue

                                print('Evaluating Agent {} on Environment {} with action bound {}", planning horizon {}, population {}, iterations {}, flatten threshold {}'.\
                                    format(args.agent, args.setting, action_bound, ph, pop, it, th))

                                policy_params = configs.policy.params
                                policy_params['agent'] = agent
                                policy_params['action_lower_bound'] = -action_bound
                                policy_params['action_upper_bound'] = action_bound
                                policy_params['no_op'] = [action_bound, action_bound, action_bound, action_bound]
                                policy_params['planning_horizon'] = ph
                                policy_params['candidates'] = pop
                                policy_params['iterations'] = it
                                policy_params['flatten_threshold'] = th
                                policy_params['clip'] = clip
                                policy_params['cloth_mask'] = cloth_mask
                                if cloth_mask == 'from_model':
                                    policy_params['cloth_mask_threshold'] = 0.48

                                policy = PolicyBuilder.build('rect_fabric_cloth_mask_mpc', policy_params)
                                        



                                qbar = tqdm(total=sum([len(tiers[k]) for k in tiers.keys()]))
                                written = False
                                for tier in reversed(list(tiers.keys())):
                                    for eid in tiers[tier]:
                                        res = perform(env, eid, policy, collect_frames=False)
                                        if 'z' in args.setting:
                                            res['actions'] = res['actions'][:, 0].reshape(-1, 2, 3)[:, :, :2].reshape(-1, 4)

                                        manupilation_logger(tier, eid, res, configs, written, 
                                                            env_eval_para['video_episodes'],
                                                            filename=save_file)
                                        written = True
                                        qbar.update(1)


if __name__ == '__main__':
    main()