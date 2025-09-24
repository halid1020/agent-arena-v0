# The code is adopted from https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing#scrollTo=VrX4VTl5pYNq

import os
from pathlib import Path
from typing import Any
import ruamel.yaml as yaml
from tqdm import tqdm
import torch
import logging
import numpy as np
from collections import deque
import torch
import cv2
import torch.nn as nn
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from agent_arena.agent.oracle.builder import OracleBuilder
from agent_arena import TrainableAgent
from agent_arena.utilities.networks.utils import np_to_ts, ts_to_np
from agent_arena.utilities.logger.logger_interface import Logger
from matplotlib import pyplot as plt
from agent_arena.utilities.visual_utils \
    import draw_pick_and_place

from .utils \
    import get_resnet, replace_bn_with_gn
from .networks import ConditionalUnet1D
from .dataset import DiffusionDataset, normalize_data, unnormalize_data

def dict_to_action_vector(dict_action, action_output_template):
    """
    Convert dictionary form of action back into flat vector form.

    dict_action: dict filled with actual action values
    action_output_template: same structure as config.action_output (lists of indices)
    """
    # length of flat action = max index + 1
    max_index = max(_max_index_in_dict(action_output_template))
    action = np.zeros(max_index + 1, dtype=float)

    def fill_action(d_action, template):
        for k, v in template.items():
            if isinstance(v, dict):
                fill_action(d_action[k], v)
            elif isinstance(v, list):
                # v = list of indices
                values = d_action[k]
                action[np.array(v)] = values

    fill_action(dict_action, action_output_template)
    return action

def _max_index_in_dict(d):
    """Helper to find all indices used in the template dict."""
    for v in d.values():
        if isinstance(v, dict):
            yield from _max_index_in_dict(v)
        elif isinstance(v, list):
            yield max(v)

class DiffusionTransform():

    def __init__(self, config, stats):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stats = stats
        #self.save_dir = config.save_dir

    def __call__(self, data, train=True):

        ret_data = {}
        #self.transform = DiffusionTransform(self.config)
        
        if not train:
            ret_data[self.config.input_obs] = data[self.config.input_obs].astype(np.float32)/255.0#
            # print('input obs shape', ret_data[self.config.input_obs].shape)
            
            if len(ret_data[self.config.input_obs].shape) == 3:
                ret_data[self.config.input_obs] = np.expand_dims(ret_data[self.config.input_obs], axis=0)
                ret_data[self.config.input_obs] = np.expand_dims(ret_data[self.config.input_obs], axis=0)

            ret_data[self.config.input_obs] = ret_data[self.config.input_obs].transpose(0, 1, 4, 2, 3)
            
            ret_data[self.config.input_obs] = np_to_ts(ret_data[self.config.input_obs], self.device)
            ret_data['vector_state'] = \
                normalize_data(data['vector_state'], 
                               self.stats[self.config.data_state])#
            ret_data['vector_state'] = np_to_ts(ret_data['vector_state'], self.device)
            if len(ret_data['vector_state'].shape) == 1:
                ret_data['vector_state'] = ret_data['vector_state'].unsqueeze(0)

            

        else:
            
            ret_data[self.config.input_obs] = data[self.config.data_obs]
            ret_data[self.config.input_obs] = np_to_ts(ret_data[self.config.input_obs][:, :self.config.obs_horizon], self.device)
            ret_data['vector_state'] = np_to_ts(data[self.config.data_state][:, :self.config.obs_horizon], self.device)
            ret_data['action'] = np_to_ts(data['action'], self.device)

        return ret_data

    def postprocess(self, data):
        # print('data keys', data.keys())
        ret_data = {}
        if 'action' in data.keys():
            data['action'] = unnormalize_data(data['action'], self.stats[self.config.data_action])
            ret_data['action'] = data['action']
        if self.config.input_obs in data.keys():
            #data[self.config.input_obs] = unnormalize_data(data[self.config.input_obs], self.stats[self.config.data_obs])
            ret_data[self.config.input_obs] = (ts_to_np(data[self.config.input_obs])*255.0).clip(0, 255).astype(np.uint8)
        return ret_data

class DiffusionAdapter(TrainableAgent):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'diffusion'
        self.config = config
        #self.save_dir = config.save_dir
        self.logger = Logger()
        self.internal_states = {}
        self.buffer_actions = {}
        self.last_actions = {}
        self.obs_deque = {}

        self._init_networks()

        

        self._init_optimizer()
        self.loaded = False
        #self.writer = SummaryWriter(self.save_dir)

        from .action_sampler import ActionSampler
        self.eval_action_sampler = ActionSampler[self.config.eval_action_sampler]()

        if self.config.dataset_mode != 'diffusion':
            transform_config = self.config.transform
            from agent_arena.api import build_transform
            self.transform =  build_transform(transform_config.name, transform_config.params)
        
        self.update_step = -1
        self.dataset_inited = False
        
        
    def _init_dataset(self):

        if self.config.dataset_mode == 'diffusion':
            dataset = DiffusionDataset(
                dataset_path=self.config.dataset_path,
                pred_horizon=self.config.pred_horizon,
                obs_horizon=self.config.obs_horizon,
                action_horizon=self.config.action_horizon
            )
            self.stats = dataset.stats
            self.transform = DiffusionTransform(self.config, self.stats)
        elif self.config.dataset_mode == 'general':
            from agent_arena.utilities.trajectory_dataset import TrajectoryDataset
            # convert dotmap to dict
            config = self.config.dataset_config.toDict()
            #print('config', config)
            dataset = TrajectoryDataset(**config)
            

            
        else:
            raise ValueError('Invalid dataset mode')

       
       
        torch.backends.cudnn.benchmark = True
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size, #64,
            #num_workers=2,
            shuffle=True,
            # accelerate cpu-gpu transfer
            #pin_memory=True,
            # don't kill worker process afte each epoch
            #persistent_workers=True
        )
        self.dataset_inited = True
        #self.dataloader = None
    
    def _init_demo_policy_dataset(self, arenas):
        arena = arenas[0] # assume only one arena
        from agent_arena.utilities.trajectory_dataset import TrajectoryDataset
            # convert dotmap to dict
        config = self.config.dataset_config.toDict()
        config['io_mode'] = 'a'
        #print('config', config)
        dataset = TrajectoryDataset(**config)

        import agent_arena as ag_ar
        policy = ag_ar.build_agent(self.config.demo_policy)

        qbar = tqdm(total=self.config.num_demos, 
                    desc='Collecting data from policy ...')

        qbar.update(dataset.num_trajectories())
        qbar.refresh()
        
        episode_id = dataset.num_trajectories()
        while dataset.num_trajectories() < self.config.num_demos:
            observations = {obs_type: [] for obs_type in dataset.obs_types}
            actions = {act_type: [] for act_type in dataset.action_types}

            policy.reset([arena.id])
            info = arena.reset({'eid': episode_id})
            policy.init(info)
            info['reward'] = 0
            done = info['done']
            while not done:
                action = policy.single_act(info)
                #print('action', action)

                if action is None:
                    break
                
                for k, v in info['observation'].items():
                    #print('k', k)
                    if k in observations.keys():
                        if k in ['rgb', 'depth']:
                            v_ = cv2.resize(v, (dataset.obs_config[k]['shape'][0], dataset.obs_config[k]['shape'][1]))
                            observations[k].append(v_)
                        elif k == 'mask':
                            v_ = cv2.resize(v_.astype(np.float32), (dataset.obs_config[k]['shape'][0], dataset.obs_config[k]['shape'][1]))
                            v_ = v_ > 0.9
                            observations[k].append(v_)
                        else:
                            observations[k].append(v_)
                    
                for k, v in action.items():
                    if k in actions.keys():
                        #print('action', v)
                        v_ = dict_to_action_vector(v, self.config.action_save.get(k))
                        #print('v_', v_)
                        actions[k].append(v_)

                info = arena.step(action)
                policy.update(info, action)
                info['reward'] = 0
                done = info['done']
                if info['success'] or policy.terminate()[arena.id]:
                    break
                
            for k, v in info['observation'].items():
                if k in observations.keys():
                    if k in ['rgb', 'depth']:
                        v_ = cv2.resize(v, (dataset.obs_config[k]['shape'][0], dataset.obs_config[k]['shape'][1]))
                        observations[k].append(v_)
                    elif k == 'mask':
                        v_ = cv2.resize(v_.astype(np.float32), (dataset.obs_config[k]['shape'][0], dataset.obs_config[k]['shape'][1]))
                        v_ = v_ > 0.9
                        observations[k].append(v_)
                    else:
                        observations[k].append(v_)
            
            if info['success'] or self.config.add_all_demos:
                #print('add to trajectory')
                for k, v in observations.items():
                    observations[k] = np.stack(v)
                for k, v in actions.items():
                    print('k', k)
                    print('action', v)
                    actions[k] = np.stack(v)
                dataset.add_trajectory(observations, actions)
                qbar.update(1)
            
            episode_id += 1
            episode_id %= arena.get_num_episodes()

        torch.backends.cudnn.benchmark = True
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size, #64,
            #num_workers=2,
            shuffle=True,
            # accelerate cpu-gpu transfer
            #pin_memory=True,
            # don't kill worker process afte each epoch
            #persistent_workers=True
        )
        self.dataset_inited = True

    def _init_optimizer(self):
        self.ema = EMAModel(
            parameters=self.nets.parameters(),
            power=0.75)
        
        self.optimizer = torch.optim.AdamW(
            params=self.nets.parameters(),
            lr=1e-4, weight_decay=1e-6)#

        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=self.config.total_update_steps ## make it manual
        )
    
    def _init_networks(self):
        self.input_channel = 3
        if self.config.input_obs == 'rgbd':
            self.input_channel = 4
        elif self.config.input_obs == 'depth':
            self.input_channel = 1
        self.vision_encoder = get_resnet('resnet18', input_channel=self.input_channel)
        self.vision_encoder = replace_bn_with_gn(self.vision_encoder)


        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.config.action_dim,
            global_cond_dim=self.config.obs_dim*self.config.obs_horizon,
            diable_updown=(self.config.disable_updown if 'disable_updown' in self.config else False),
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config.num_diffusion_iters, # default value 100
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

        self.nets = nn.ModuleDict({
            'vision_encoder': self.vision_encoder,
            'noise_pred_net': self.noise_pred_net
        })

        self._test_network()

        self.device = torch.device('cuda' \
            if torch.cuda.is_available() else 'cpu')
        self.nets.to(self.device)
        
    def _test_network(self):

        with torch.no_grad():
            # example inputs
            image = torch.zeros(
                (1, self.config.obs_horizon,
                 self.input_channel,96,96))
             # vision encoder
            image_features = self.nets['vision_encoder'](
                image.flatten(end_dim=1)
            )
            # (2,512)
            obs = image_features.reshape(*image.shape[:2],-1)

            if self.config.include_state:
                vector_state = torch.zeros(
                    (1, self.config.obs_horizon, 
                    self.config.state_dim))
                obs = torch.cat([obs, vector_state],dim=-1)
            
            # print('obs', obs.shape)
            
            noised_action = torch.randn(
                (1, self.config.pred_horizon, self.config.action_dim))
            diffusion_iter = torch.zeros((1,))
            # print('noised action', noised_action.shape)

            # the noise prediction network
            # takes noisy action, diffusion iteration and observation as input
            # predicts the noise added to action
            noise = self.nets['noise_pred_net'](
                sample=noised_action,
                timestep=diffusion_iter,
                global_cond=obs.flatten(start_dim=1))

            # illustration of removing noise
            # the actual noise removal is performed by NoiseScheduler
            # and is dependent on the diffusion noise schedule
            # denoised_action = noised_action - noise


    def train(self, update_steps, arenas):
        if not self.dataset_inited:
            if self.config.train_mode == 'from_dataset':
                self._init_dataset()
            elif self.config.train_mode == 'from_policy':
                self._init_demo_policy_dataset(arenas)
            else:
                raise ValueError('Invalid train mode')
        
        update_steps = min(#
            self.config.total_update_steps - self.update_step-1,
            update_steps)
        #print('train update steps', update_steps)
        pbar = tqdm(range(update_steps), desc="Training")

        for i in pbar:


            # get a batch from dataloader
            nbatch = next(iter(self.dataloader))
            # print('nbatch keys', nbatch.keys())

            # if True:
            #     rgb = ts_to_np(nbatch['observation']['rgb'][0][0])
            #     H, W = rgb.shape[:2]
            #     #print('rgb shape', rgb.shape)
            #     act = ts_to_np(nbatch['action']['default'][0][0]).reshape(2, 2)
            #     px_act = ((act + 1)/2 * np.array([H, W])).astype(np.int32)
            #     start = px_act[0]
            #     end = px_act[1]
            #     pnp_rgb = draw_pick_and_place(
            #         rgb, start, end, get_ready=True, swap=True
            #     )
            #     plt.imsave('tmp/pre_pnp_rgb.png', pnp_rgb)

            if self.config.dataset_mode == 'diffusion':
                nbatch = self.transform(nbatch, train=True)
            else:
                obs = nbatch['observation']
                action = nbatch['action']['default']
                print('action', action.shape)
                nbatch = {v: k for v, k in obs.items()}
                nbatch['action'] = action.reshape(*action.shape[:2], -1)
                print('action after shape', nbatch['action'] .shape)
                nbatch = self.transform(nbatch, train=True)

            # print('nbatch rgb shape', nbatch['rgb'].shape)

            # if True:
            #     rgb = (ts_to_np(nbatch['rgb'][0][0])*255).clip(0, 255)\
            #         .astype(np.uint8).transpose(1, 2, 0)
            #     H, W = rgb.shape[:2]
            #     act = nbatch['action'][0][0].cpu().numpy().reshape(2, 2)
            #     px_act = ((act + 1)/2 * np.array([H, W])).astype(np.int32)
            #     start = px_act[0]
            #     end = px_act[1]
            #     pnp_rgb = draw_pick_and_place(
            #         rgb, start, end, get_ready=True, swap=True
            #     )
            #     plt.imsave('tmp/post_pnp_rgb.png', pnp_rgb)
            
            if self.config.input_obs == 'rgbd':
                # concatenate rgb and depth
                nbatch['rgbd'] = torch.cat([
                    nbatch['rgb'], nbatch['depth']], dim=2)

                # from matplotlib import pyplot as plt
                # plt.imshow(nbatch['rgbd'][0, 0, 3].squeeze(0).cpu().numpy())
                # plt.savefig('process-depth.png')
                # plt.imshow(nbatch['rgbd'][0, 0, :3].squeeze(0).cpu().numpy().transpose(1, 2, 0))
                # plt.savefig('process-rgb.png')
                            
                #print('rgbd shape', nbatch['rgbd'].shape)
            
            # nimage = nbatch['image'][:,:self.config.obs_horizon].to(self.device)
            
            # nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
            # naction = nbatch['action'].to(device)
            B = nbatch[self.config.input_obs].shape[0]
            input_obs = nbatch[self.config.input_obs][:, :self.config.obs_horizon]\
                .flatten(end_dim=1)

            ## check if the input_obs is in the correct shape

            # if input_obs.shape[-1] > 4:
            #     print('input obs shape', input_obs.shape)
            #     input_obs = input_obs.permute(0, 3, 1, 2)

            # encoder vision features
            image_features = self.nets['vision_encoder'](
                input_obs)
            obs_features = image_features.reshape(
                B, self.config.obs_horizon, -1)
            # (B,obs_horizon,D)

            # concatenate vision feature and low-dim obs
            if self.config.include_state:
                vector_state = nbatch['vector_state'][:, :self.config.obs_horizon]
                obs_features = torch.cat([obs_features, vector_state], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)
            # (B, obs_horizon * obs_dim)

            # sample noise to add to actions
            noise = torch.randn(nbatch['action'].shape, device=self.device)
            print('noise shape', noise.shape)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=self.device
            ).long()

            # add noise to the clean images according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            print('action before adding noise',  nbatch['action'].shape)
            noisy_actions = self.noise_scheduler.add_noise(
                nbatch['action'], noise, timesteps)

            print('noisy actino shape', noisy_actions.shape)

            # predict the noise residual
            noise_pred = self.noise_pred_net(
                noisy_actions, timesteps, global_cond=obs_cond)

            # L2 loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            # optimize
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # step lr scheduler every batch
            # this is different from standard pytorch behavior
            self.lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            self.ema.step(self.nets.parameters())

            ## write loss value to tqdm progress bar
            pbar.set_description(f"Training (loss: {loss.item():.4f})")
            self.train_writer.add_scalar('train/loss', loss.item(), self.update_step)
            self.update_step += 1

    def set_log_dir(self, logdir):
        super().set_log_dir(logdir)
        self.save_dir = logdir

        
    def save(self):
        
        ## save to the path self.save_dir/'checkpoints'/net_{update_step}.pt
        ckpt_path = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(ckpt_path, exist_ok=True)
        ckpt_path = os.path.join(ckpt_path, f'net_{self.update_step}.pt')
        torch.save(self.nets.state_dict(), ckpt_path)
    
    def load_checkpoint(self, checkpoint):
        #print('loading checkpoint', checkpoint)
        ckpt_path = os.path.join(self.save_dir, 'checkpoints', f'net_{checkpoint}.pt')
        #print('ckpt path', ckpt_path)
        self.nets.load_state_dict(torch.load(ckpt_path))
        print(f'Loaded checkpoint: {checkpoint}')
        self.loaded = True


    def load(self):
        
        #print('loading checkpoint')
        ## find the latest checkpoint
        ckpt_path = os.path.join(self.save_dir, 'checkpoints')
        #print('ckpt path', ckpt_path)
        os.makedirs(ckpt_path, exist_ok=True)
        ckpt_files = os.listdir(ckpt_path)
        ckpt_files = [ckpt for ckpt in ckpt_files if ckpt.endswith('.pt')]
        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        if len(ckpt_files) == 0:
            print('No checkpoint found')
            return -1
        ckpt_file = ckpt_files[-1]
        ckpt_path = os.path.join(ckpt_path, ckpt_file)
        self.nets.load_state_dict(torch.load(ckpt_path))

        print(f'Loaded checkpoint: {ckpt_file}')
        self.loaded = True
        self.update_step = int(ckpt_file.split('_')[1].split('.')[0])
        return self.update_step

    def single_act(self, info, update):

        if update == True:
            last_action = self.last_actions[info['arena_id']]
            
            if last_action is not None:
                self.update(info, last_action)
            else:
                self.init(info)

        if len(self.buffer_actions[info['arena_id']]) == 0:
            image = torch.stack([x[self.config.input_obs] \
                                    for x in self.obs_deque[info['arena_id']]])
            sample_state = {'image': image}
            # from matplotlib import pyplot as plt
            # plt.imsave('tmp/input_obs.png', image[-1, 0].cpu().numpy())
            if self.config.use_mask:
                mask = torch.stack([x['mask'] for x in self.obs_deque[info['arena_id']]])
                sample_state['mask'] = mask

            obs_features = self.nets['vision_encoder'](image)
            # print('obs features shape', obs_features.shape)

            if self.config.include_state:
                vector_state = torch.stack([x['vector_state'] \
                                            for x in self.obs_deque[info['arena_id']]])
                # print('vector state shape', vector_state.shape)
                
                obs_features = torch.cat([obs_features, vector_state], dim=-1)
            
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
            #print('obs cond', obs_cond.shape)
            naction = self.eval_action_sampler.sample(
                state=sample_state, 
                horizon=self.config.pred_horizon, 
                action_dim=self.config.action_dim
            ).to(self.device)

            start = self.config.obs_horizon - 1
            end = start + self.config.action_horizon
            
            #torch.randn((1, self.config.pred_horizon, self.config.action_dim)).to(self.device)

            self.noise_scheduler.set_timesteps(self.config.num_diffusion_iters)
            noise_actions = [ts_to_np(naction[:, start:end])]
            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

                noise_actions.append(ts_to_np(naction[:, start:end]))
            
            

            action_pred = self.transform.postprocess(
                {'action': ts_to_np(naction)})['action'][0]
            
            self.buffer_actions[info['arena_id']] = deque(
                action_pred[start:end,:], 
                maxlen=self.config.action_horizon)

        action = self.buffer_actions[info['arena_id']]\
            .popleft().reshape(action_shape)

        # if 'readjust_pick' in self.config and self.config.readjust_pick:
        #     #print('readjusting pick')
                
        #     mask = mask[-1][0].cpu().numpy()
        #     #print('mask', mask.shape)
        #     from agent_arena.utilities.utils import adjust_points
        #     H, W = mask.shape
        #     pixel_action = ((action[0, :2] + 1)/2 * np.array([H, W])).astype(np.int32)
        #     pixel_action = np.clip(pixel_action, 0, [H-1, W-1])
        #     place = ((action[0, 2:] + 1)/2 * np.array([H, W])).astype(np.int32)
        #     points = [(pixel_action[0], pixel_action[1])]
        #     if 'swap_pick_and_place' in self.config and self.config.swap_pick_and_place:
        #         #print('swap pick and place (pre)', points)
        #         points = [(pixel_action[1], pixel_action[0])]
        #         place = (place[1], place[0])
        #         #print('swap pick and place (post)', points)
            
        #     # if True:
        #     #     rgb = infos[-1]['observation']['rgb']
        #     #     # resize to H, W
        #     #     import cv2
        #     #     rgb = cv2.resize(rgb, (W, H))
                
        #     #     pre_adjust_rgb = draw_pick_and_place(   
        #     #         rgb, points[0], place, get_ready=True)
        #     #     plt.imsave('tmp/pre_adjust_rgb.png', pre_adjust_rgb)
                
        #     #print('pre adjust pick', points)
        #     adjusted_pick, _ = adjust_points(
        #         points, mask, min_distance=1)
            
        #     # if True:
        #     #     import cv2
        #     #     rgb = cv2.resize(rgb, (W, H))
        #     #     post_adjust_rgb = draw_pick_and_place(
        #     #         rgb, adjusted_pick[0], place, get_ready=True)
        #     #     plt.imsave('tmp/post_adjust_rgb.png', post_adjust_rgb)


        #     if 'swap_pick_and_place' in self.config and self.config.swap_pick_and_place:
        #         #print('swap pick and place back (pre)', adjusted_pick)
        #         adjusted_pick = [(adjusted_pick[0][1], adjusted_pick[0][0])]
        #         place = (place[1], place[0])
        #         #print('swap pick and place back (post)', adjusted_pick)
            
        #     #print('adjusted pick', adjusted_pick)
        #     action[:, 0] = adjusted_pick[0][0]/H*2 - 1
        #     action[:, 1] = adjusted_pick[0][1]/W*2 - 1

        #     noise_actions.append(action.reshape(noise_actions[-1].shape))
        

        #     self.internal_states[info['arena_id']]['noise_actions'] = np.stack(noise_actions).reshape(-1, self.config.action_dim)
            
        
        # print('input obs shape', self.state['input_obs'].shape)
        # print('noise actions shape', self.state['noise_actions'].shape)
        # if 'norm-pixel-pick-and-place' in self.config.action_output:
        #     from agent_arena.utilities.visual_utils import draw_pick_and_place_noise_actions
        #     self.internal_states[info['arena_id']]['denoise_action_input_obs_rgb'] = draw_pick_and_place_noise_actions(
        #         self.internal_states[info['arena_id']]['input_obs'][:, :, :3], 
        #         self.internal_states[info['arena_id']]['noise_actions'],
        #         filename='noise_actions_input_obs.png',
        #         directory='./tmp')

        #     self.internal_states[info['arena_id']]['denoise_action_rgb'] = draw_pick_and_place_noise_actions(
        #         info['observation']['rgb'],
        #         self.internal_states[info['arena_id']]['noise_actions'],
        #         filename='noise_actions_rgb.png',
        #         directory='./tmp')
            
        #     self.internal_states[info['arena_id']]['action_obs_rgb'] = draw_pick_and_place_noise_actions(
        #         self.internal_states[info['arena_id']]['input_obs'][:, :, :3],
        #         self.internal_states[info['arena_id']]['noise_actions'],
        #         filename='action_obs_rgb.png',
        #         directory='./tmp',
        #         draw_noise = False)

        
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

        self.last_actions[info['arena_id']] = ret_action

        return action

    def act(self, infos, updates):
        action_shape = self.config.action_shape
        ret_actions = []

        for info, upd in zip(infos, updates):
            
            #if upd:
            self.single_act(info, upd)

            
            
            ret_actions.append(ret_action)
        
        #print('ret actions', ret_actions)
        return ret_actions
    
    def reset(self, arena_ids):
        
        if not self.loaded:
            self.load()
            
        for arena_id in arena_ids:
            self.internal_states[arena_id] = {}
            self.buffer_actions[arena_id] = deque(maxlen=self.config.action_horizon)
            self.last_actions[arena_id] = None


    def get_state(self):
        return self.internal_states

    def _process_info(self, info):

        if 'depth' in info['observation'].keys():
            depth = info['observation']['depth']
            if len(depth.shape) == 2:
                    depth = np.expand_dims(depth, axis=-1)
                    info['observation']['depth'] = depth

        # if True:
        #    # save input rgb, depth and mask
        #     rgb = info['observation']['rgb']
        #     depth = info['observation']['depth']
        #     mask = info['observation']['mask']

        #     plt.imshow(rgb)
        #     plt.savefig('tmp/input_rgb.png')
        #     plt.imshow(depth)
        #     plt.savefig('tmp/input_depth.png')
        #     plt.imshow(mask)
        #     plt.savefig('tmp/input_mask.png')

        if self.config.input_obs == 'rgbd':
            info['observation']['rgbd'] = np.concatenate(
                [info['observation']['rgb'].astype(np.float32), depth], axis=-1)
            #print('rgbd shape', info['observation']['rgbd'].shape)

        input_data = {
            self.config.input_obs: info['observation'][self.config.input_obs]\
                .reshape(1, 1, *info['observation'][self.config.input_obs].shape),
            
        }
        if 'use_mask' in self.config and self.config.use_mask:
            input_data['mask'] = info['observation']['mask']\
                .reshape(1, 1, *info['observation']['mask'].shape, 1)
            
        if self.config.include_state:
            input_data['vector_state'] = info['observation']['vector_state']\
                .reshape(1, 1, *info['observation']['vector_state'].shape)
        
        input_data = self.transform(input_data, train=False) 
                                    #sim2real=info['sim2real'] if 'sim2real' in info else False)
        
        vis = input_data[self.config.input_obs].squeeze(0).squeeze(0)
        #print('vis shape', vis.shape)
        obs = {
            self.config.input_obs: vis,  
        }

        vis_to_save = vis.cpu().numpy().transpose(1, 2, 0).repeat(3, axis=-1)

        #plt.imsave('tmp/input_obs.png', vis_to_save)

        if 'use_mask' in self.config and self.config.use_mask:
            mask = input_data['mask'].squeeze(0).squeeze(0)
            obs['mask'] = mask

        if self.config.include_state:
            vector_state = input_data['vector_state'].squeeze(0).squeeze(0)
            obs['vector_state'] = vector_state

        input_obs = self.transform.postprocess(obs)[self.config.input_obs]
        self.internal_states[info['arena_id']].update(
            {'input_obs': input_obs.transpose(1,2,0),
             'input_type': self.config.input_obs}
        )
        
        return obs

    def init(self, infos):
        #print('info keys', info.keys())
        for info in infos:
            obs = self._process_info(info)
            # for k, v in obs.items():
            #     print('k', k)
            #     print('v shape', v.shape)
            self.obs_deque[info['arena_id']] = deque([obs]*self.config.obs_horizon, 
                                maxlen=self.config.obs_horizon)

    def update(self, infos, actions):
        for info, action in zip(infos, actions):
            obs = self._process_info(info)
            self.obs_deque[info['arena_id']].append(obs)
    
    def set_eval(self):
        pass
    
    def set_train(self):
        pass