import os
import sys
sys.path.insert(0, '..')

from dotmap import DotMap
import ruamel.yaml as yaml
from tqdm import tqdm
from pathlib import Path
import wandb
import argparse

from cloth_folding_IRL.src.policies.pick_and_place_rect_fabric_flattening_policies import RectFabricPickPlaceReadjustMPC
from scripts.settings import *
from utilities.loggers import loss_wandb_logger, eval_wandb_logger, manupilation_wandb_logger
from utilities.utils import perform



def main():

    ### Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', default='pick-and-place-cloth-flattening-old')
    parser.add_argument('--agent',  default='') ### TODO: force it let it have input
    parser.add_argument('--hyper_para',  default='')
    args = parser.parse_args()
    if args.hyper_para == '':
        args.hyper_para = args.setting

    def run():
        run = wandb.init()
        config = wandb.config
        

        ### Initiliase dataset
        dataset_params = yaml.safe_load(Path('yamls/dataset.yaml').read_text())[args.setting]
        train_dataset = name_to_dataset[args.setting](config=config, **dataset_params)
        test_dataset = name_to_dataset[args.setting](config=config, **dataset_params)
        
        ### Initialise Agent
        config.save_dir = os.path.join('tmp', args.setting, args.agent, run.name)
        model = name_to_agent[args.agent](config)
        ### Train Agent
        model.train(train_dataset, test_dataset, loss_wandb_logger, eval_wandb_logger)



        ### Initiliase Policy ### TODO: this is not general.
        mpc_params = yaml.safe_load(Path('yamls/policy.yaml').read_text())[args.setting]
        mpc_params['cost_fn'] = model.cost_fn
        mpc_params['unroll_action'] = model.unroll_action
        mpc_params['model'] = model
        config.reward_pred = model.reward_pred()
        mpc_params['configs'] = config
        policy = RectFabricPickPlaceReadjustMPC(**mpc_params)
        
        # Environment
        env_para = yaml.safe_load(Path('yamls/softgym_env.yaml').read_text())[args.setting]
        env_para['headless'] = True
        env = name_to_env[args.setting](env_para)
        env.set_eval()
        

        
        ### Evaluate Agent on the Environment
        tiers = env_para['eval_tiers']
        qbar = tqdm(total=sum([len(tiers[k]) for k in tiers.keys()]))
        written = False
        for tier in reversed(list(tiers.keys())):
            for eid in tiers[tier]:   
                res = perform(model, env, eid, policy, config, collect_frames=True, 
                    init_state=model.init_state, update_state=model.update_state)
                #print('here')
                manupilation_wandb_logger(tier, eid, res, config, written, env_para['video_episodes'])
                written = True
                qbar.update(1)

    
    ### Initilise Project
    sweep_configuration = yaml.safe_load(Path('yamls/{}/{}/sweeps/{}.yaml'.format(args.agent, args.setting, args.hyper_para)).read_text())
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project=args.agent + '_on_' + args.setting + '_with_' + args.hyper_para)

    wandb.agent(sweep_id, function=run, count=10)

if __name__ == '__main__':
    main()