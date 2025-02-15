import os
import os.path as osp
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../../OpenSTL/')

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
from torch.utils.data import DataLoader
import torch

from openstl.core.metrics import *

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
    agent.load_models(os.path.join(agent.config.save_dir, 'model/model.pth'))

    ### Initiliase datasets and transform
    datasets = {}

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
    
    history_frames = 10
    future_frames = 10
    batchsize = 100
    agent.set_eval()
    dataloader = DataLoader(
            datasets['test'],
            batch_size=batchsize)
    
    metrics = ['mae', 'mse', 'rmse', 'psnr', 'ssim']
    results = {}
    folder_path = osp.join(agent.config.save_dir, 'vp_results', 'saved')
    data = next(iter(dataloader))

    data = agent.preprocess(data)
    init_belief = torch.zeros(
        batchsize, 
        agent.config.deterministic_latent_dim).to(agent.config.device)

    init_state = torch.zeros(
        batchsize,
        agent.config.stochastic_latent_dim).to(agent.config.device)
    
    past_actions = data['action'][:history_frames-1, :]
    past_observations = data['observation'][:history_frames, :]
    future_actions = data['action'][history_frames-1:history_frames+future_frames-1, :]
    future_observations = data['observation'][history_frames:history_frames+future_frames]

    past_beliefs, past_states, _ = agent._unroll_state_action(
        past_observations[1:], past_actions,
        init_belief, init_state, None)
    future_beliefs, future_states = agent._unroll_action(
        future_actions, past_beliefs[-1], past_states['sample'][-1])
    
    future_observations_pred = agent.reconstruct(future_beliefs, future_states)

    ## Image is in B*T*C*H*W, resize it to B*T*C*128*128
    past_observations = torch.nn.functional.interpolate(
        past_observations.reshape(-1, 3, 64,64), size=(128, 128)).reshape(10, batchsize, 3, 128, 128)
    
    future_observations = torch.nn.functional.interpolate(
        future_observations.reshape(-1, 3, 64,64), size=(128, 128)).reshape(10, batchsize, 3, 128, 128)
    
    future_observations_pred = torch.nn.functional.interpolate(
        future_observations_pred.reshape(-1, 3, 64,64), size=(128, 128)).reshape(10, batchsize, 3, 128, 128)

    results['input_images'] = past_observations.cpu().numpy().transpose(1, 0, 2, 3, 4) + 0.5
    results['trues'] = future_observations.cpu().numpy().transpose(1, 0, 2, 3, 4) + 0.5
    results['preds'] = future_observations_pred.detach().cpu().numpy().transpose(1, 0, 2, 3, 4) + 0.5
    
    eval_res, eval_log = metric(results['preds'], results['trues'],
            0, 1,
            metrics=metrics, channel_names=None, spatial_norm=False)

    results['metrics'] = np.array([eval_res[m] for m in metrics])
    print(eval_res)
    if not osp.exists(folder_path):
        os.makedirs(folder_path)
    for np_data in ['metrics', 'input_images', 'trues', 'preds']:
        np.save(osp.join(folder_path, np_data + '.npy'), results[np_data])


if __name__ == '__main__':
    main()