# extend upon https://github.com/Xingyu-Lin/softagent/blob/master/planet/models.py
import os
import numpy as np
import cv2
from tqdm import tqdm
from typing import Optional, List
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torch.distributions as td
from pathlib import Path
import ruamel.yaml as yaml
import matplotlib.image as mpimg
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import logging

from agent_arena.agent.utilities.torch_utils import *
from agent_arena.registration.dataset import *
from agent_arena.agent.oracle.builder import OracleBuilder
from agent_arena.utilities.visual_utils import plot_pick_and_place_trajectory
from agent_arena.utilities.utils import TrainWriter
from agent_arena.utilities.transform.register import DATA_TRANSFORMER
from agent_arena import TrainableAgent
from agent_arena.utilities.logger.logger_interface import Logger

from .networks import ImageEncoder, ImageDecoder
from .memory import ExperienceReplay
from .contrastive import ContrastiveEncoder
from .logger import *
from .cost_functions import *
from .model import *


class RSSM(TrainableAgent):

    def __init__(self, config):
        self.config = config
        self.input_obs = self.config.input_obs
        
        self.no_op = np.asarray(config.no_op).flatten()
        self.model = dict()
        self.model['transition_model'] = TransitionModel(
            belief_size=self.config.deterministic_latent_dim,
            state_size=self.config.stochastic_latent_dim,
            action_size = np.prod(np.array(self.config.action_dim)), 
            hidden_size=self.config.hidden_dim,
            embedding_size=self.config.embedding_dim,
            activation_function=self.config.activation,
            min_std_dev=self.config.min_std_dev,
            embedding_layers=self.config.trans_layers
        ).to(self.config.device)

        self.model['observation_model'] = ImageDecoder(
            image_dim=config.output_obs_dim,
            belief_size=config.deterministic_latent_dim,
            state_size=config.stochastic_latent_dim,
            embedding_size=config.embedding_dim,
            activation_function=config.activation,
            batchnorm=config.decoder_batchnorm
        ).to(config.device)

        self.model['reward_model'] = RewardModel(
            belief_size=self.config.deterministic_latent_dim,
            state_size=self.config.stochastic_latent_dim,
            hidden_size=self.config.hidden_dim,
            activation_function=self.config.activation,
        ).to(self.config.device)

        if self.config.encoder_mode == 'contrastive':
            self.model['encoder'] = ContrastiveEncoder(
                self.config).to(self.config.device)
        elif self.config.encoder_mode == 'default':
            self.model['encoder'] = ImageEncoder(
                image_dim=self.config.input_obs_dim,
                embedding_size=self.config.embedding_dim,
                activation_function=self.config.activation,
                batchnorm=self.config.encoder_batchnorm,
                residual=self.config.encoder_residual
            ).to(config.device)
        else:
            raise NotImplementedError

        params = [list(m.parameters()) for m in self.model.values()]
        self.param_list = []
        for p in params:
            self.param_list.extend(p)
        
        # Count the number of all parameters in the model
        num_parameters = 0
        for k, v in self.model.items():
            n = sum(p.numel() for p in v.parameters())
            print(f"Number of parameters in {k}: {n}")

            num_parameters += n

        print(f"Number of all parameters in the model: {num_parameters}")


        optimiser_params = self.config.optimiser_params.copy()
        optimiser_params['params'] = self.param_list
        self.optimiser = OPTIMISER_CLASSES[self.config.optimiser_class](**optimiser_params)
        self.loaded = False
        self.symlog = self.config.symlog

        #Dot map to dict
        transform_config = self.config.transform
        self.transform = DATA_TRANSFORMER[transform_config.name](transform_config.params)
        self.apply_transform_in_dataset = self.config.get('apply_transform_in_dataset', False)


        planning_config = self.config.policy.params
        planning_config.model = self
        planning_config.action_space = self.config.action_space
        planning_config.no_op = self.no_op
        
        import agent_arena.api as ag_ar
        self.planning_algo = ag_ar.build_agent(
            self.config.policy.name,
            config=planning_config)
        
        
        self.internal_states = {}
        self.logger = Logger()
        self.cur_state = {}

    def set_log_dir(self, logdir):
        super().set_log_dir(logdir)
        self.save_dir = logdir
        self.writer = TrainWriter(self.save_dir)
    
        
    def reset(self, areana_ids):
        for arena_id in areana_ids:
            self.cur_state[arena_id] = {}
            self.internal_states[arena_id] = {}

    def get_state(self):
        return self.internal_states
        
    def act(self, infos, update=False):
        actions = []
        for info in infos:
            action =  self.planning_algo.act([info])[0].flatten()
            plan_internal_state = self.planning_algo.get_state()[info['arena_id']]
            
            for k, v in plan_internal_state.items():
                self.internal_states[info['arena_id']][k] = v
            
            ## covert self.config.action_output to dict and copy it
            ret_action = self.config.action_output.copy().toDict()
            action = action.flatten()

            ## recursively goes down the dictionary tree, when encounter list of integer number
            ## replace list with corresponding indexed values in `action`

            def replace_action(action, ret_action):
                for k, v in ret_action.items():
                    if isinstance(v, dict):
                        replace_action(action, v)
                    elif isinstance(v, list):
                        #print('v', v)
                        ret_action[k] = action[v]

            replace_action(action, ret_action)

            actions.append(ret_action)
       
        return actions

    def get_name(self):
        return "RSSM PlaNet"
    
    def train(self, update_steps, arena) -> bool:
        torch.backends.cudnn.benchmark = True
     
        if self.config.train_mode == 'offline':
            datasets = {}
            if 'datasets' in self.config:
                if 'initialised_datasets' in self.config and self.config.initialised_datasets:
                    datasets = self.config.datasets
                else:
                    for dataset_dict in self.config['datasets']:
                        key = dataset_dict['key']
                        print()
                        print('Initialising dataset {} from name {}'.format(key, dataset_dict['name']))

                        dataset_params = dataset_dict['params']
                        
                        
                        dataset = name_to_dataset[dataset_dict['name']](
                            **dataset_params)

                        datasets[key] = dataset
            else:
                raise NotImplementedError

            for key, dataset in datasets.items():
                if self.apply_transform_in_dataset:
                    dataset.set_transform(self.transform)
                
            self._train_offline(datasets, update_steps)

        elif self.config.train_mode == 'online':
            policy_params = self.config.explore_policy.params
            policy_params['base_policy'] = self
            policy_params['action_space'] = arena.get_action_space()
            self.explore_policy = OracleBuilder.build(
                self.config.explore_policy.name, arena, policy_params)


            self.train_online(arena, self.explore_policy, update_steps)
        else: 
            raise NotImplementedError
    
        return True
    
    def reconstruct_observation(self, state):

        return ts_to_np(bottle(
            self.model['observation_model'], 
            (state['deter'], state['stoch']['sample'])))
    
    def _update_helper(self, info, action):
        arena_id = info['arena_id']
        obs = info['observation']
        mask = obs['mask']
        self.no_op = self.no_op.flatten()

        
        if self.config.input_obs == 'gc-depth':
            obs_ = np.concatenate([obs['depth'], obs['goal_depth']], axis=-1)
            goal_mask = obs['goal_mask']
        elif self.config.input_obs == 'gc-rgb':
            obs_ = np.concatenate([obs['rgb'], obs['goal_rgb']], axis=-1)
            goal_mask = obs['goal_mask']
        elif self.config.input_obs == 'gc-rgbd':
            obs_ = np.concatenate([obs['rgb'], obs['depth'], obs['goal_rgb'], obs['goal_depth']], axis=-1)
            goal_mask = obs['goal_mask']
        elif self.config.input_obs == 'rgbd':
            obs_ = np.concatenate([obs['rgb'], obs['depth']], axis=-1)
        else:
            obs_ = info['observation'][self.config.input_obs]

        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)

        
        to_trans_dict = {
            self.config.input_obs: np.expand_dims(obs_, axis=(0)),
            'mask': np.expand_dims(mask, axis=(0))
        }
        if self.config.input_obs in ['gc-depth', 'gc-rgb', 'gc-rgbd']:
            if len(goal_mask.shape) == 2:
                goal_mask = np.expand_dims(goal_mask, axis=0)
            to_trans_dict['goal-mask'] = np.expand_dims(goal_mask, axis=(0))
            

        image = self.transform(
            to_trans_dict, 
            train=False)[self.config.input_obs].to(self.config.device)
       
        
        image = symlog(image, self.symlog)
        if len(image.shape) == 4:
            image = image.unsqueeze(0)

        

        self.cur_state[arena_id] = {
            'deter': torch.zeros(
                1, self.config.deterministic_latent_dim, 
                device=self.config.device),
            'stoch': {
                'sample': torch.zeros(
                    1, self.config.stochastic_latent_dim, 
                    device=self.config.device)
            },
            'input_obs': image # batch*horizon*C*H*W
        }
        
    
        action = np_to_ts(action.flatten(), self.config.device).unsqueeze(0).unsqueeze(0)


        self.cur_state[arena_id] = self.unroll_state_action(self.cur_state[arena_id] , action)


        self.internal_states[arena_id] = {
            'raw_input_obs': obs_.copy(),
            'input_obs': image.squeeze(0).squeeze(0) \
                .cpu().detach().numpy().transpose(1, 2, 0),
            'deter_state': self.cur_state[arena_id]['deter']\
                .squeeze(1).cpu().detach().numpy(),
            'stoch_state': self.cur_state[arena_id]['stoch']['sample']\
                .squeeze(1).cpu().detach().numpy()
        }

    def init(self, infos):
        if not self.loaded:
            self.load()
        for info in infos:
            self._update_helper(info,  np.asarray(self.no_op))
             
    def flatten_action(self, action):
        if 'norm-pixel-pick-and-place' in self.config.action_output:
            action = action['norm-pixel-pick-and-place']
        return np.stack([action['pick_0'], action['place_0']]).flatten()
        
    def update(self, infos, actions):
        if self.config.refresh_init_state:
            self.init(infos)
            return
        
        for info, action in zip(infos, actions):
            self._update_helper(info, self.flatten_action(action))
           
    def cost_fn(self, trajectory, goal=None):
        if self.config.cost_fn == 'trajectory_return':
            return trajectory_return(trajectory, self)
        elif self.config.cost_fn == 'last_step_z_divergence_goal':
            return last_step_z_divergence_goal(trajectory, goal, self)
        elif self.config.cost_fn == 'last_step_z_divergence_goal_reverse':
            return last_step_z_divergence_goal(trajectory, goal, self, revserse=True)
        elif self.config.cost_fn == 'last_step_z_distance_goal_stoch':
            return last_step_z_distance_goal_stoch(trajectory, goal, self)
        elif self.config.cost_fn == 'last_step_z_distance_goal_deter':
            return last_step_z_distance_goal_deter(trajectory, goal, self)
        elif self.config.cost_fn == 'last_step_z_distance_goal_both':
            return last_step_z_distance_goal_both(trajectory, goal, self)
        elif self.config.cost_fn == 'last_step_reward_with_uncertainty':
            return last_step_reward_with_uncertainty(trajectory, self)
        else:
            raise NotImplementedError
        
    def unroll_action_from_cur_state(self, action, state_):

        to_unroll = {}
        candidates, horizons = action.shape[:2]
        action = action.reshape(candidates, horizons, -1)
        state= self.cur_state[state_['arena_id']]
        # #state = self.cur_state
        to_unroll['deter'] = state['deter']\
                .squeeze(dim=1).expand(1, candidates, self.config.deterministic_latent_dim)\
                .reshape(-1, self.config.deterministic_latent_dim)
        
        to_unroll['stoch'] = {
            'sample': state['stoch']['sample']\
                .squeeze(dim=1).expand(1, candidates, self.config.stochastic_latent_dim)\
                .reshape(-1, self.config.stochastic_latent_dim)
        }

        action = np_to_ts(action, self.config.device).permute(1, 0, 2) ## horizon*candidates*actions

        return self.unroll_action(to_unroll, action)
    
    def visual_reconstruct(self, state):

        images = bottle(self.model['observation_model'], 
                        (state['deter'], 
                         state['stoch']['sample']))
        
        images = ((ts_to_np(images).transpose(1, 0, 3, 4, 2) + 0.5)*255.0)\
            .clip(0, 255).astype(np.uint8)

        return images

    def reward_pred(self):
        return lambda a : symexp(self.model['reward_model'](a), self.symlog)
    
    def set_eval(self):
        for v in self.model.values():
            v.eval()

    def set_train(self):
        for v in self.model.values():
            v.train()

    def save(self, path=None):
        
        model_dict = {
            'transition_model': self.model['transition_model'].state_dict(),
            'observation_model': self.model['observation_model'].state_dict(),
            'reward_model': self.model['reward_model'].state_dict(),
            'encoder': self.model['encoder'].state_dict(),
            'optimiser': self.optimiser.state_dict()
        }
        
        if path is None:
            path = self.save_dir
        
        os.makedirs(os.path.join(path, 'checkpoints'), exist_ok=True)
    
        torch.save(
            model_dict, 
            os.path.join(path, 'checkpoints', f'model_{self.update_step}.pth')
        )

        torch.save(
            self.metrics,
            os.path.join(path, 'checkpoints', f'metrics_{self.update_step}.pth')
        )

        if self.config.checkpoint_experience:
            dst = os.path.join(path, 'checkpoints', 'experience.pkl')
            self.memory.save(dst)

    def _load_from_model_dir(self, model_dir):
        checkpoint = torch.load(model_dir)

        self.model['transition_model'].load_state_dict(checkpoint['transition_model'])
        self.model['observation_model'].load_state_dict(checkpoint['observation_model'])
        self.model['reward_model'].load_state_dict(checkpoint['reward_model'])
        self.model['encoder'].load_state_dict(checkpoint['encoder'])
        self.optimiser.load_state_dict(checkpoint['optimiser'])

        self.loaded = True
             
    def load(self, path=None):

        if path is None:
            path = self.save_dir
        
        checkpoint_dir =os.path.join(path, 'checkpoints')

        ## find the latest checkpoint
        if not os.path.exists(checkpoint_dir):
            print('No checkpoint found in directory {}'.format(checkpoint_dir))
            return 0
        
        checkpoints = os.listdir(checkpoint_dir)
        checkpoints = [int(c.split('_')[1].split('.')[0]) for c in checkpoints]
        checkpoints.sort()
        checkpoint = checkpoints[-1]
        model_dir = os.path.join(checkpoint_dir, f'model_{checkpoint}.pth')

        
        if not os.path.exists(model_dir):
            print('No model found for loading in directory {}'.format(model_dir))
            return 0
        
        self._load_from_model_dir(model_dir)
        print('Loaded checkpoint {}'.format(checkpoint))
        self.loaded = True
        return checkpoint
        
    def load_checkpoint(self, checkpoint: int) -> bool:
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        model_dir = os.path.join(checkpoint_dir, f'model_{checkpoint}.pth')
        if not os.path.exists(model_dir):
            print('No model found for loading in directory {}'.format(model_dir))
            return False
        self._load_from_model_dir(model_dir)
        print('Loaded checkpoint {}'.format(checkpoint))
        return True
    
    def _load_metrics(self, path=None):

        if path is None:
            path = self.save_dir
        
        if not os.path.exists(os.path.join(path, 'checkpoint', 'metrics.pth')):
            return {}
        
        return torch.load(os.path.join(path, 'checkpoint', 'metrics.pth'))

    def _preprocess(self, data, train=False, single=False, apply_transform=True):

        if self.config.datasets[0].name == 'default':
            if 'observation' in data:
                obs_data = data['observation']
                act_data = data['action']['default']
            else:
                obs_data = data
                act_data = data['action']
            
            if single:
                obs_data['reward'] = obs_data['reward'][1:].squeeze(-1)
                obs_data['terminal'] = obs_data['terminal'][1:].squeeze(-1)
            else:
                obs_data['reward'] = obs_data['reward'][:, 1:].squeeze(-1)
                obs_data['terminal'] = obs_data['terminal'][:, 1:].squeeze(-1)
           
            data = {
                'action': act_data,
            }
            data.update(obs_data)
        

        
        # Apply transformations
        if apply_transform:
            if single:
                for k, v in data.items():
                    data[k] = np.expand_dims(v, 0)
            data = self.transform(data, train=train)
            for k, v in data.items():
                data[k] = v.to(self.config.device)

        # Swap axes for all items in data
        for k in data:
            data[k] = torch.swapaxes(data[k], 0, 1)

        # Precompute shapes if needed
        if self.config.input_obs == 'rgbd':
            T, B, C, H, W = data['rgb'].shape
            # Concatenate and apply symlog transformation
            rgbd = torch.cat([data['rgb'], data['depth']], dim=2)
            data['input_obs'] = symlog(rgbd, self.symlog)
        elif self.config.input_obs == 'gc-depth':
            gc_depth = torch.cat([data['depth'], data['goal-depth']], dim=2)
            data['input_obs'] = symlog(gc_depth, self.symlog)
        elif self.config.input_obs == 'gc-rgb':
            gc_rgb = torch.cat([data['rgb'], data['goal-rgb']], dim=2)
            data['input_obs'] = symlog(gc_rgb, self.symlog)
        elif self.config.input_obs == 'gc-rgbd':
            gc_rgbd = torch.cat([data['rgb'], data['depth'], data['goal-rgb'], data['goal-depth']], dim=2)
            data['input_obs'] = symlog(gc_rgbd, self.symlog)
        else:
            data['input_obs'] = symlog(data[self.config.input_obs], self.symlog)

        # Determine output observation based on configuration
        if self.config.output_obs == 'input_obs':
            data['output_obs'] = data['input_obs']
        elif self.config.output_obs == 'rgbm':
            rgbm = torch.cat([data['rgb'], data['mask']], dim=2)
            data['output_obs'] = symlog(rgbm, self.symlog)
        elif self.config.output_obs == 'gc-mask':
            gc_mask = torch.cat([data['mask'], data['goal-mask']], dim=2)
            data['output_obs'] = symlog(gc_mask, self.symlog)
        else:
            data['output_obs'] = symlog(data[self.config.output_obs], self.symlog)

        # Apply symlog to reward
        if self.config.get('train_reward_clip', False):
            clip_range = self.config.get('train_reward_clip')
            data['reward'] = torch.clamp(data['reward'], clip_range[0], clip_range[1])
        
        data['reward'] = symlog(data['reward'], self.symlog)


        return data
    
    def train_online(self, env, explore_policy):
        start_update_step = self.load()
        metrics = self._load_metrics()
        action_space = env.get_action_space()
        
        if metrics == {}:
            metrics = {
                'update_step': [],
                # 'train_episodes': [],
                # 'interactive_steps': [],
                'update_step_at_train_episode': [],
                'train_episodes_reward_mean': [],
                'train_episodes_reward_std': [],
                'update_step_at_test_episode': [],
                'test_episodes_reward_mean': [],
                'test_episodes_reward_std': []
            }
        
        

        self.memory = ExperienceReplay(
            self.config.memory_size, 
            self.config.symbolic_env, 
            self.config.input_obs_dim, 
            self.config.action_dim,  
            self.config.device)

        experience_dir = os.path.join(self.save_dir, 'model/experience.pkl')
        if self.config.checkpoint_experience and os.path.exists(experience_dir):
            self.memory.load(experience_dir)
    
        
        
        ## Initial data collection
        # if start_update_step == 0 and start_update_step < self.config.total_update_steps:
        env.set_train()
        with torch.no_grad():
            update = start_update_step
            total_rewards = []
            #s = train_episodes
            for _ in tqdm(range(self.memory.episodes, self.config.intial_train_episodes), 
                            desc="Collecting Initial Training Epsiodes"):
                #train_episodes += 1
                information, total_reward = env.reset(), 0
                obs = information['observation']['rgb']
                

                while not information['done']:
                    action = env.sample_random_action()
                    information = env.step(action)
                    # *self.config.input_obs_dim[1:]
                    obs = cv2.resize(obs, (64, 64) , interpolation=cv2.INTER_LINEAR)
                    mpimg.imsave(
                        os.path.join(self.save_dir, 'train_online.png'), obs)
                    self.memory.append(
                        obs.transpose(2, 0, 1), 
                        action, 
                        information['reward'], 
                        information['done'])
                    obs = information['observation']['rgb']
                    
                    total_reward += information['reward']
                #interactive_steps += env.get_max_interactive_steps()
                
                total_rewards.append(total_reward)

            metrics['update_step_at_train_episode'].append(update)
            metrics['train_episodes_reward_mean'].append(np.mean(total_rewards))
            metrics['train_episodes_reward_std'].append(np.std(total_rewards))
            

            print('Average running reward {} at update step {}/{}'\
                .format(np.mean(total_rewards), update, self.config.total_update_steps))
        
        self.set_train()
        for update in tqdm(range(start_update_step+1, self.config.total_update_steps), desc='Updateing RSSM'):

            # Test Policy in the Env
            if update%self.config.test_interval == 0:
                self.metrics = metrics
                self.save()
                self.set_eval()
                total_rewards = []
                # TODO: change explore policy to test policy
                env.set_eval()
                for e in tqdm(range(self.config.test_episodes), desc="Testing Epsiodes"):
                    information, total_reward = env.reset(episode_config={'eid': e, 'save_video': False}), 0
                    explore_policy.init_state(information)

                    while not information['done']:
                        mpimg.imsave(
                            os.path.join(self.save_dir, 'test_online.png'),
                            information['observation']['rgb'])
                        
                        action = explore_policy.act(information, env)
                        information = env.step(action)
                        total_reward += information['reward']

                        explore_policy.update(information, action)

                    total_rewards.append(total_reward)
                    
                    
                metrics['update_step_at_test_episode'].append(update)
                metrics['test_episodes_reward_mean'].append(np.mean(total_rewards))
                metrics['test_episodes_reward_std'].append(np.std(total_rewards))

                print('Test average reward {} at update step {}/{}'\
                      .format(np.mean(total_rewards), update, self.config.total_update_steps))

                
                self.set_train()
            

            # Train RSSM
            data = self.memory.sample(self.config.batch_size, self.config.sequence_size)
            #print('sample rgb shape', data['rgb'].shape)
            for k, v in data.items():
                if k != 'rgb':
                    data[k] = v[:-1]
                data[k] = np.transpose(v, (1, 0, *range(2, v.ndim)))

            data = self._preprocess(data, train=True)

            self.optimiser.zero_grad()

            losses = self.compute_losses(data, update)
            

            losses['total_loss'].backward()
            nn.utils.clip_grad_norm_(self.param_list, self.config.grad_clip_norm, norm_type=2)
            self.optimiser.step()

            # Collect Losses
            for kk, vv in losses.items():
                if kk in metrics.keys():
                    metrics[kk].append(vv.detach().cpu().item())
                else:
                    metrics[kk] = [vv.detach().cpu().item()]
            metrics['update_step'].append(update)

            # Collect episode data
            if update%self.config.collect_interval == self.config.collect_interval-1:
                # self.save_checkpoint(
                #     metrics,                
                #     self.config.models_dir)
                env.set_train()
                with torch.no_grad():
                    total_rewards = []
                    print('Total loss {} at update step {}'.\
                          format(metrics['total_loss'][-1], metrics['update_step'][-1]))

                    for _ in tqdm(range(self.config.train_episodes), desc="Collecting Training Epsiodes"):

                        information, total_reward = env.reset(), 0
                        obs = information['observation']['rgb']
                        explore_policy.init(information)
                        

                        while not information['done']:
                            
                            action = explore_policy.act(information, env)
                            action += np.random.normal(size=action.shape)*self.config.action_noise
                            action = action.clip(action_space.low, action_space.high)
                            information = env.step(action)
                            explore_policy.update_state(information, action)


                            obs = cv2.resize(obs, (64, 64) , interpolation=cv2.INTER_LINEAR)
                            mpimg.imsave(
                                os.path.join(self.save_dir, 'train_online.png'), obs)
                            self.memory.append(
                                obs.transpose(2, 0, 1), 
                                action, 
                                information['reward'], 
                                information['done'])
                        
                            obs = information['observation']['rgb']
                            total_reward += information['reward']                           
                        
                        total_rewards.append(total_reward)

                    metrics['update_step_at_train_episode'].append(update)
                    metrics['train_episodes_reward_mean'].append(np.mean(total_rewards))
                    metrics['train_episodes_reward_std'].append(np.std(total_rewards))

                    print('Average running reward {} at update step {}/{}'\
                        .format(np.mean(total_rewards), update, self.config.total_update_steps))
      
    def _train_offline(self, datasets, update_steps=-1):
        
        train_dataset = datasets['train']
        test_dataset = datasets['test']
        # self.transform=test_dataset.transform
        
        losses_dict = {}
        updates = []
        start_step = self.load()
        metrics = self._load_metrics()
        if metrics == {}:
            metrics = {
                'update_step': []
            }


        self.set_train()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.get('dataloader_workers', 0),
            shuffle=True)

        
        

        end_update_steps = self.config.total_update_steps if update_steps == -1 \
            else min(start_step+update_steps+1, self.config.total_update_steps)

        for u in tqdm(range(start_step, end_update_steps )):
            self.update_step = u
        
            data = next(iter(train_dataloader))

            data = {k: v.to(self.config.device, non_blocking=True) for k,v in data.items()}

            data = self._preprocess(data, train=True, apply_transform=(not self.apply_transform_in_dataset))
            data['input_obs'] = data['input_obs'][:-1]
            data['output_obs'] = data['output_obs'][:-1]

            

            self.optimiser.zero_grad()

            losses = self.compute_losses(data, u)
            

            losses['total_loss'].backward()
            nn.utils.clip_grad_norm_(self.param_list, self.config.grad_clip_norm, norm_type=2)
            self.optimiser.step()

            # Collect Losses
            for kk, vv in losses.items():
                if kk in losses_dict.keys():
                    losses_dict[kk].append(vv.detach().cpu().item())
                else:
                    losses_dict[kk] = [vv.detach().cpu().item()]
                
                self.writer.add_scalar(kk, vv.detach().cpu().item(), u)

            updates.append(u)
            
            if u%self.config.test_interval == 0:
                self.set_eval()

                # Save Losses
                losses_dict.update({'update_step': updates})
                loss_logger(losses_dict, self.save_dir)
                losses_dict = {}
                updates = []

                # Evaluate & Save
                test_results = self.evaluate(test_dataset)
                train_results = self.evaluate(train_dataset)
                results = {'test_{}'.format(k): v for k, v in test_results.items()}
                results.update({'train_{}'.format(k): v for k, v in train_results.items()})

                for k, v in results.items():
                    self.writer.add_scalar(k, v, u)

                results['update_step'] = [u]

                eval_logger(results, self.save_dir)
                #break # Change here
                
                
                # Save Model
                self.metrics = {'update_step': [u]}
                self.save()
            

                self.set_train()
        
        # Visualised, Evaluate & Save
        if self.config.visusalise:
            self.visualise(datasets)
        
        if self.config.end_training_evaluate:
            test_results = self.evaluate(test_dataset)
            train_results = self.evaluate(train_dataset)
            results = {'test_{}'.format(k): v for k, v in test_results.items()}
            results.update({'train_{}'.format(k): v for k, v in train_results.items()})
            results['update_step'] = [self.config.total_update_steps-1]
            eval_logger(results, self.config)
        
    def evaluate(self, dataset, train=False):

        reward_rmses = {h:[] for h in self.config.test_horizons}
        observation_mses = {h:[] for h in self.config.test_horizons}
        kls_post_to_prior = {h:[] for h in self.config.test_horizons}
        kls_prior_to_post = {h:[] for h in self.config.test_horizons}
        prior_entropies = {h:[] for h in self.config.test_horizons}
        posterior_reward_rmses = []
        posterior_recon_mses = []
        posterior_entropies = []
        eval_action_horizon = self.config.eval_action_horizon
        
        for i in tqdm(range(self.config.eval_episodes)):
            episode = dataset.get_trajectory(i)

            episode = self._preprocess(episode, train=False, single=True, apply_transform=True)
            for k, v in episode.items():
                episode[k] = torch.swapaxes(v, 0, 1).squeeze(0)

            init_belief = torch.zeros(1, self.config.deterministic_latent_dim).to(self.config.device)
            init_state = torch.zeros(1, self.config.stochastic_latent_dim).to(self.config.device)

            no_op_ts = np_to_ts(self.no_op, self.config.device).unsqueeze(0)
            actions = episode['action'][:eval_action_horizon]

            actions = torch.cat([no_op_ts, actions])
            
            input_obs = episode['input_obs'][:eval_action_horizon+1]
            output_obs = episode['output_obs'][:eval_action_horizon+1]

            rewards = episode['reward'][:eval_action_horizon]

            beliefs, posteriors, priors, obs_embedding = self._unroll_state_action(
                input_obs.unsqueeze(1), actions.unsqueeze(1), init_belief, init_state, 
                None, project_obs=True)
            
            if ('eval_save_latent' in self.config) and self.config.eval_save_latent and (not train):
                data = {
                    'belief': ts_to_np(beliefs),
                    'posterior': ts_to_np(posteriors['mean']),
                    'one-step prior': ts_to_np(priors['mean'])
                }
                latent_dir = os.path.join(self.save_dir, 'latent_space')
                os.makedirs(latent_dir, exist_ok=True)
                torch.save(
                    data,
                    os.path.join(latent_dir, 'evaldata_episode_{}.pth'.format(i))
                )

            if ('eval_save_obs_embedding' in self.config) and self.config.eval_save_obs_embedding and (not train):
                data = {}
                for k, v in obs_embedding.items():
                    data[k] = ts_to_np(v)
                
                latent_dir = os.path.join(self.save_dir, 'emb_space')
                os.makedirs(latent_dir, exist_ok=True)
                torch.save(
                    data,
                    os.path.join(latent_dir, 'evaldata_episode_{}.pth'.format(i))
                )
            

            posterior_reward =  bottle(self.model['reward_model'], 
                                       (beliefs[1:], posteriors['sample'][1:]))\
                                        .transpose(0, 1).squeeze(0)
            
            posterior_observation = bottle(self.model['observation_model'], 
                                           (beliefs[1:], posteriors['sample'][1:]))\
                                            .transpose(0, 1).squeeze(0)

            post_dist = ContDist(td.independent.Independent(
                td.normal.Normal(posteriors['mean'][1:], posteriors['std'][1:]), 1))
            
        


            posterior_entropies.extend(post_dist.entropy().mean(dim=-1).flatten().detach().cpu().tolist())

            posterior_reward_rmses.extend((F.mse_loss(
                        symexp(posterior_reward, self.symlog), 
                        symexp(rewards, self.symlog), 
                        reduction='none')**0.5).flatten().detach().cpu().tolist())
            
            posterior_recon_mses.extend(F.mse_loss(
                        symexp(posterior_observation, self.symlog),
                        symexp(output_obs[1:], self.symlog),
                        reduction='none').mean((1, 2, 3)).flatten().detach().cpu().tolist())

            
            # T*30
            for horizon in self.config.test_horizons:
                horizon_actions = [actions[j + 2: j+horizon+2] for j in range(eval_action_horizon-horizon-1)]
                horizon_actions = torch.swapaxes(torch.stack(horizon_actions), 0, 1)
                
                B = horizon_actions.shape[1]
                init_post = posteriors['sample'][1:B+1].squeeze(1)

                imagin_beliefs, imagin_priors = self._unroll_action(
                    horizon_actions, 
                    beliefs[1:B+1].squeeze(1), init_post) # horizon*B

                
                imagin_reward =  bottle(self.model['reward_model'], 
                                        (imagin_beliefs, imagin_priors['sample'])).transpose(0, 1)
                imagin_observation = bottle(self.model['observation_model'], 
                                            (imagin_beliefs, imagin_priors['sample'])).transpose(0, 1)
                
                true_reward = torch.stack(
                    [episode['reward'][j+1: j+horizon+1] for j in range(eval_action_horizon-horizon-1)])\
                        .reshape(-1, horizon)
                true_image = torch.stack(
                    [output_obs[j+2: j+horizon+2] for j in range(eval_action_horizon-horizon-1)])


                reward_rmses[horizon].extend((F.mse_loss(
                        symexp(imagin_reward, self.symlog), 
                        symexp(true_reward, self.symlog),
                        reduction='none')**0.5).flatten().detach().cpu().tolist())
                
                observation_mses[horizon].extend(F.mse_loss(
                        symexp(imagin_observation, self.symlog) , 
                        symexp(true_image, self.symlog),
                        reduction='none').mean((2, 3, 4)).flatten().detach().cpu().tolist())
                
                imagin_post = {k: torch.stack([posteriors[k][j+2:j+2+horizon, 0] \
                                               for j in range(eval_action_horizon-horizon-1)]) 
                                               for k in posteriors.keys()}


                imagin_post_dist = ContDist(td.independent.Independent(
                td.normal.Normal(imagin_post['mean'].transpose(0, 1), imagin_post['std'].transpose(0, 1)), 1))._dist
                
                
                
                imagin_prior_dist = ContDist(td.independent.Independent(
                td.normal.Normal(imagin_priors['mean'], imagin_priors['std']), 1))._dist
                
                

                kls_post_to_prior[horizon].extend(
                    td.kl.kl_divergence(imagin_post_dist, imagin_prior_dist)\
                        .flatten().detach().cpu().tolist())
                kls_prior_to_post[horizon].extend(
                    td.kl.kl_divergence(imagin_prior_dist, imagin_post_dist)\
                        .flatten().detach().cpu().tolist())

                

                prior_entropies[horizon].extend(
                    imagin_prior_dist.entropy().mean(dim=-1)\
                        .flatten().detach().cpu().tolist())
                
        
        results = {
            'img_prior_reward_rmse': {h:reward_rmses[h] for h in self.config.test_horizons},
            'img_prior_img_observation_mse': {h:observation_mses[h] for h in self.config.test_horizons},
            'kl_divergence_between_posterior_and_img_prior': {h:kls_post_to_prior[h] for h in self.config.test_horizons},
            'img_prior_entropy':  {h:prior_entropies[h] for h in self.config.test_horizons}
        }

        res = {
            'posterior_img_observation_mse_mean': np.mean(posterior_recon_mses),
            'posterior_img_observation_mse_std': np.std(posterior_recon_mses),
            'posterior_reward_rmse_mean': np.mean(posterior_reward_rmses),
            'posterior_reward_rmse_std': np.std(posterior_reward_rmses),
            'posterior_entropy_mean': np.mean(posterior_entropies),
            'posterior_entropy_std': np.std(posterior_entropies)
        }
        
        for k, v in results.items():
            for h in self.config.test_horizons:
                res['{}_horizon_{}_mean'.format(k, h)] = np.mean(v[h])
                res['{}_horizon_{}_std'.format(k, h)] = np.std(v[h])
        

        return res
    
    def visualise(self, datasets):
        self._visualise(datasets['train'], train=True)
        self._visualise(datasets['test'], train=False)

    def _visualise(self, dataset, train=False):
        train_str = 'Train' if train else 'Eval'
        
        for e in range(5):
            org_gt = dataset.get_episode(e)
            input_obs = self.config.input_obs
            if self.config.input_obs == 'rgbd':
                input_obs = 'rgb'
            # org_gt = dataset.transform.post_transform(data)

            plot_pick_and_place_trajectory(
                org_gt[input_obs][6:16].transpose(0, 2 ,3, 1),
                org_gt['action'][6:16],
                title='{} Ground Truth Episode {}'.format(train_str, e),
                # rewards=data['reward'][5:15], 
                save_png = True, 
                save_path=os.path.join(self.save_dir, 'visualisations'))
            
            data = {}
            for k, v in org_gt.items():
                data[k] = np.expand_dims(v, 0)
            data = self._preprocess(data, train=False, apply_transform=True)
            for k, v in data.items():
                data[k] = torch.swapaxes(v, 0, 1).squeeze(0)

            recon_image = []

            init_belief = torch.zeros(1, self.config.deterministic_latent_dim).to(self.config.device)
            init_state = torch.zeros(1, self.config.stochastic_latent_dim).to(self.config.device)


            no_op_ts = np_to_ts(self.no_op, self.config.device).unsqueeze(0)
            actions = np_to_ts(data['action'], self.config.device)
            actions = torch.cat([no_op_ts, actions])

            observations = np_to_ts(data['input_obs'], self.config.device)
            rewards = np_to_ts(data['reward'], self.config.device)

            beliefs, posteriors, priors, _ = self._unroll_state_action(
                observations.unsqueeze(1), actions.unsqueeze(1), 
                init_belief, init_state, None)

            posterior_observations = bottle(self.model['observation_model'], (beliefs, posteriors['sample'])).squeeze(1)
            posterior_observations = symexp(posterior_observations, self.symlog)
            if self.config.output_obs == 'input_obs':
                post_process_obs = self.transform.post_transform({self.config.input_obs: posterior_observations})[self.config.input_obs]
            else:
                # post_process_obs = posterior_observations.detach().cpu().numpy()
                post_process_obs = self.transform.post_transform({self.config.output_obs: posterior_observations})[self.config.output_obs]

            
            
            posterior_rewards = bottle(self.model['reward_model'], (beliefs, posteriors['sample'])).squeeze(1)
            posterior_rewards = symexp(posterior_rewards, self.symlog)
            posterior_rewards = posterior_rewards.detach().cpu().numpy()

           

            plot_pick_and_place_trajectory(
                post_process_obs[6:16].transpose(0, 2 ,3, 1),
                # rewards=posterior_rewards[6:16], 
                title='{} Posterior Trajectory Episode {}'.format(train_str, e), 
                save_png = True,
                save_path=os.path.join(self.save_dir, 'visualisations'))
            
            recon_image.append(post_process_obs[6:11].transpose(0, 2 ,3, 1))

            
            # T*30
            horizon = 5
            horizon_actions = [actions[j + 1: j+horizon+1] for j in range(dataset.eval_action_horizon-horizon)]
            horizon_actions = torch.swapaxes(torch.stack(horizon_actions), 0, 1) # 4*64*1 

            B = horizon_actions.shape[1]

            imagin_beliefs, imagin_priors = self._unroll_action(
                horizon_actions, 
                beliefs[:B].squeeze(1), posteriors['sample'][:B].squeeze(1)) # horizon*B
            
            prior_observations = bottle(self.model['observation_model'], (imagin_beliefs, imagin_priors['sample']))
            prior_observations = symexp(prior_observations, self.symlog)

            prior_rewards = bottle(self.model['reward_model'], (imagin_beliefs, imagin_priors['sample'])) 
            prior_rewards = symexp(prior_rewards, self.symlog)
            prior_rewards = prior_rewards.detach().cpu().numpy()


            for i in range(horizon):
                if self.config.output_obs == 'input_obs':
                    post_process_img_obs = self.transform.post_transform({self.config.input_obs: prior_observations[i]})[self.config.input_obs]
                else:
                    post_process_img_obs = self.transform.post_transform({self.config.output_obs: prior_observations[i]})[self.config.output_obs]

                plot_pick_and_place_trajectory(
                    post_process_img_obs[5-i:15-i].transpose(0, 2 ,3, 1),
                    # rewards=prior_rewards[i][5-i:15-i], 
                    title='{}-Step {} Prior Trajectory Episode {}'.format(i, train_str, e), 
                    save_png = True,
                    save_path=os.path.join(self.save_dir, 'visualisations'))
                recon_image.append(post_process_img_obs[5+5:6+5].transpose(0, 2 ,3, 1))

            recon_image = np.concatenate(recon_image, axis=0)
            plot_pick_and_place_trajectory(
                    recon_image,
                    # rewards=posterior_rewards[6:16], 
                    title='{} Recon Trajectory Episode {}'.format(train_str, e), 
                    save_png = True,
                    save_path=os.path.join(self.save_dir, 'visualisations'))
    

    def _unroll_state_action(self, obs, acts, blf, lst, 
                             non_terminals, project_obs=False):
        
        obs_emb = {}
        obs_emb['emb'] = self.model['encoder'](obs)

        if self.config.encoder_mode == 'contrastive' and project_obs:
            obs_proj = self.model['encoder'].project(obs)
            obs_emb['proj'] = obs_proj

        blfs, prior_states_, prior_means_, prior_std_devs_, posterior_states_, posterior_means_, posterior_std_devs_ = \
            self.model['transition_model'](
                lst, 
                acts, 
                blf, 
                obs_emb['emb'], 
                non_terminals)

        posteriors_ = {
            'sample': posterior_states_,
            'mean': posterior_means_,
            'std': posterior_std_devs_
        }

        priors_ = {
            'sample': prior_states_,
            'mean': prior_means_,
            'std': prior_std_devs_
        }
        
        return blfs, posteriors_, priors_, obs_emb
    
    def get_writer(self):
        return self.writer

    def unroll_state_action(self, state, action):

        blfs, prior_states_, prior_means_, prior_std_devs_, posterior_states_, posterior_means_, posterior_std_devs_ = \
            self.model['transition_model'](
                
                state['stoch']['sample'],
                action, 
                state['deter'], 
                self.model['encoder'](state['input_obs']), 
                None)

        posteriors_ = {
            'sample': posterior_states_,
            'mean': posterior_means_,
            'std': posterior_std_devs_
        }

        return {
            'deter': blfs,
            'stoch': posteriors_
        }
    
    
    

    def _unroll_action(self, actions, belief_, latent_state_):


        img_beliefs_, prior_states_, prior_means_, prior_std_devs_  = \
            self.model['transition_model'](
                latent_state_, 
                actions, 
                belief_, 
                None,
                None)

        priors_ = {
            'sample': prior_states_,
            'mean': prior_means_,
            'std': prior_std_devs_
        }
        
        return img_beliefs_, priors_

    def unroll_action(self, init_state, actions):
        img_beliefs_, prior_states_, prior_means_, prior_std_devs_  = \
            self.model['transition_model'](
                init_state['stoch']['sample'], 
                actions, 
                init_state['deter'], 
                None,
                    None)
        
        return {
            'deter': img_beliefs_,
            'stoch': {
                'sample': prior_states_,
                'mean': prior_means_,
                'std': prior_std_devs_
            }
        }

    def unscaled_overshooting_losses(self, experience, beliefs, posteriors):
        if self.config.kl_overshooting_scale == 0:
            return torch.tensor(0).to(self.config.device), torch.tensor(0).to(self.config.device)

        actions = experience['action']
        non_terminals = 1 - experience['terminal']
        rewards = experience['reward']

        
        overshooting_vars = [] 
        for t in range(1, self.config.sequence_size - 1):
            d = min(t + self.config.overshooting_distance, self.config.sequence_size - 1)  # Overshooting distance
            t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
            seq_pad = (0, 0, 0, 0, 0, t - d + self.config.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch

            # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) posterior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
            overshooting_vars.append((
                F.pad(actions[t:d], seq_pad), 
                F.pad(non_terminals[t:d].unsqueeze(2), seq_pad), 
                F.pad(rewards[t:d], seq_pad[2:]), 
                beliefs[t_], 
                posteriors['sample'][t_].detach(), 
                F.pad(posteriors['mean'][t_ + 1:d_ + 1].detach(), seq_pad), 
                F.pad(posteriors['std'][t_ + 1:d_ + 1].detach(), seq_pad, value=1), 
                F.pad(torch.ones(d - t, self.config.batch_size, self.config.stochastic_latent_dim, device=self.config.device), seq_pad)
            ))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
        

        overshooting_vars = tuple(zip(*overshooting_vars))
        

        # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
        beliefs, prior_states, prior_means, prior_std_devs = self.model['transition_model'](
            torch.cat(overshooting_vars[4], dim=0), 
            torch.cat(overshooting_vars[0], dim=1), 
            torch.cat(overshooting_vars[3], dim=0), 
            None, 
            torch.cat(overshooting_vars[1], dim=1))

        reward_seq_mask = torch.cat(overshooting_vars[7], dim=1)
        
        

        # Calculate overshooting KL loss with sequence mask

        posteriors = {
            'mean': torch.cat(overshooting_vars[5], dim=1), 
            'std': torch.cat(overshooting_vars[6], dim=1)}
        
        priors = {
            'mean': prior_means, 
            'std': prior_std_devs}

        kl_overshooting_loss  = self.compute_kl_loss(
            posteriors, priors, 
            self.config.kl_overshooting_balance, 
            free=self.config.free_nats)


        if self.config.reward_overshooting_scale != 0:
           
            if self.config.reward_gradient_stop:
                reward_overshooting_loss = F.mse_loss(
                    bottle(self.model['reward_model'],
                    (beliefs.detach(), prior_states.detach())) * reward_seq_mask[:, :, 0], 
                    torch.cat(overshooting_vars[2], dim=1), reduction='none').mean()
            else:
                reward_overshooting_loss = F.mse_loss(
                        bottle(self.model['reward_model'], 
                        (beliefs, prior_states)) * reward_seq_mask[:, :, 0], 
                        torch.cat(overshooting_vars[2], dim=1), 
                        reduction='none').mean()

        else:
            reward_overshooting_loss = torch.tensor(0).to(self.config.device)

        
        
        return kl_overshooting_loss, reward_overshooting_loss

    def compute_kl_loss(self, post, prior, balance=0.8, forward=False, free=1.0):

        if self.config.kl_balancing:
            kld = td.kl.kl_divergence
            sg = lambda x: {k: v.detach() for k, v in x.items()}
            lhs, rhs = (prior, post) if forward else (post, prior)
            sg_lhs, sg_rhs = sg(lhs), sg(rhs)
            
            lhs = ContDist(td.independent.Independent(
                    td.normal.Normal(lhs['mean'],lhs['std']), 1))
            sg_lhs = ContDist(td.independent.Independent(
                    td.normal.Normal(sg_lhs['mean'], sg_lhs['std']), 1))
            rhs = ContDist(td.independent.Independent(
                    td.normal.Normal(rhs['mean'],rhs['std']), 1))
            sg_rhs = ContDist(td.independent.Independent(
                    td.normal.Normal(sg_rhs['mean'], sg_rhs['std']), 1))

            mix = balance if forward else (1 - balance)
            value_lhs = kld(lhs._dist, sg_rhs._dist)
            value_rhs = kld(sg_lhs._dist, rhs._dist)
            
            loss_lhs = torch.maximum(torch.mean(value_lhs), torch.Tensor([free])[0])
            loss_rhs = torch.maximum(torch.mean(value_rhs), torch.Tensor([free])[0])
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        else:
            
            free_nats = torch.full((1, ), free, dtype=torch.float32, device=self.config.device)
            
            loss = torch.max(
                kl_divergence(
                    Normal(post['mean'], post['std']), 
                    Normal(prior['mean'], prior['std'])).sum(dim=2), 
                free_nats).mean(dim=(0, 1))
            

        return loss
    
    def compute_losses(self, data, steps):
        
        # Create initial belief and state for time t = 0
        init_belief = torch.zeros(
            self.config.batch_size, 
            self.config.deterministic_latent_dim).to(self.config.device)

        init_state = torch.zeros(
            self.config.batch_size, 
            self.config.stochastic_latent_dim).to(self.config.device)

        

        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)

        actions = data['action']
        non_terminals = 1 - data['terminal']
        rewards = data['reward']
        input_obs = data['input_obs']
        output_obs = data['output_obs']


        beliefs, posteriors, priors, _ = self._unroll_state_action(
            input_obs[1:], actions[:-1], 
            init_belief, init_state, non_terminals[:-1].unsqueeze(-1))
        

        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); 
        # sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)

        observation_loss = F.mse_loss(
            bottle(self.model['observation_model'], (beliefs, posteriors['sample'])), 
            output_obs[1:],
            reduction='none')

        observation_loss = observation_loss[:, :, :, :, :].sum(dim=(2, 3, 4)).mean(dim=(0, 1))

        pred_rewards = bottle(self.model['reward_model'], (beliefs, posteriors['sample']))

        if self.config.reward_gradient_stop:
            pred_rewards = bottle(self.model['reward_model'], (beliefs.detach(), posteriors['sample'].detach()))
        else:
            pred_rewards = bottle(self.model['reward_model'], (beliefs, posteriors['sample']))

        reward_loss = F.mse_loss(
            pred_rewards, 
            rewards[:-1],
            reduction='none').mean(dim=(0, 1))

        kl_loss = self.compute_kl_loss(
            posteriors, priors, 
            self.config.kl_balance, free=self.config.free_nats)

        posterior_entropy = td.normal.Normal(posteriors['mean'], posteriors['std']).entropy().mean().detach().cpu()
        prior_entropy =  td.normal.Normal(priors['mean'], priors['std']).entropy().mean().detach().cpu()

        # Overshooting
        kl_overshooting_loss, reward_overshooting_loss = \
                self.unscaled_overshooting_losses(data, beliefs, posteriors)

        if self.config.kl_overshooting_warmup:
            kl_overshooting_scale_ = 1.0*steps/self.config.total_update_steps*self.config.kl_overshooting_scale
        else:
            kl_overshooting_scale_= self.config.kl_overshooting_scale

        if self.config.reward_overshooting_warmup:
            reward_overshooting_scale_ = 1.0*steps/self.config.total_update_steps*self.config.reward_overshooting_scale
        else:
            reward_overshooting_scale_= self.config.reward_overshooting_scale

        total_loss = self.config.observation_scale*observation_loss + \
            self.config.reward_scale*reward_loss + \
            self.config.kl_scale * kl_loss + \
            kl_overshooting_scale_ * kl_overshooting_loss + \
            reward_overshooting_scale_ * reward_overshooting_loss
        
        res = {
            'obs_loss': observation_loss,
            'reward_loss': reward_loss,
            'kl_loss': kl_loss,
            "posterior_entropy": posterior_entropy,
            "prior_entropy": prior_entropy,
            "kl_overshooting_loss": kl_overshooting_loss,
            "reward_overshooting_loss": reward_overshooting_loss
        }
        
        if self.config.encoder_mode == 'contrastive':
            
            contrastive_loss = self.model['encoder'].compute_loss(
                data['anchors'],
                data['positives'])
            
            total_loss += self.config.contrastive_scale * contrastive_loss
            res['contrastive_loss'] = contrastive_loss

            if steps % self.config.update_contrastive_target_interval == 0:
                self.model['encoder'].update_target()

        res['total_loss'] =  total_loss

        return res 