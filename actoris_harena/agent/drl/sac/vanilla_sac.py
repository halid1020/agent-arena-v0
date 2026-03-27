from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from collections import deque
from tqdm import tqdm

from actoris_harena import TrainableAgent
import zarr
from .replay_buffer import ReplayBuffer
from .replay_buffer_zarr import ReplayBufferZarr

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.q1_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_3 = nn.Linear(hidden_dim, 1)
       
        self.q2_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_3 = nn.Linear(hidden_dim, 1)


    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        # Q1
        q1 = F.relu(self.q1_1(x))
        q1 = F.relu(self.q1_2(q1))
        q1 = self.q1_3(q1)
        # Q2
        q2 = F.relu(self.q2_1(x))
        q2 = F.relu(self.q2_2(q2))
        q2 = self.q2_3(q2)
        return q1, q2

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)


    # tanh is not applied on mean
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mean, std

    # tan h is applied on mean'
    def sample(self, obs):
        #print('obs shape', obs.shape)
        mean, std = self(obs)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class VanillaSAC(TrainableAgent):
    """A vanilla Soft Actor-Critic for image observations.

    Assumptions:
    - Replay buffer stores contextual image observations as a stacked channel: (C * N, H, W)
    - The provided `ConvEncoder`, `MLPActor`, and `Critic` have interfaces compatible with this code:
        - encoder(input) -> z (batch x enc_dim)
        - actor.sample(z) -> (action, logp, _)
        - actor(z) -> (mean, log_std) optionally
        - critic(concat(z, action)) -> q
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = config.name
        self.config = config
        self.device = torch.device(config.device)
        self.context_horizon = config.context_horizon
        # self.each_image_shape = config.each_image_shape
       
       
        self._make_actor_critic(config)

        
        # replay
        #self._init_reply_buffer(config)
        self.replay_device = config.get('replay_device', 'RAM')
        self.save_replay = config.get('save_replay', False)
        self.init_reply = False
        # bookkeeping
        self.update_steps = 0
        self.loaded = False
        #self.logger = WandbLogger(project="vanilla-sac", name=config.exp_name, config=dict(config))
        
        


        self.obs_keys = config.obs_keys
        self.reward_key = config.reward_key
        self.last_done = True
        self.episode_return = 0.0
        self.episode_length = 0
        self.act_steps = 0
        self.sim_steps = 0
        self.last_sim_steps = 0
        self.initial_act_steps = config.initial_act_steps
        #self.act_steps_per_update = config.act_steps_per_update
        self.total_update_steps = config.total_update_steps
        self.info = None
        self.save_success = config.get('save_success', False)
        #self.data_augmenter = build_data_augmenter(self.config.data_augmenter)

    def _make_actor_critic(self, config):
        # actor and critics (two critics for twin-Q)
        self.critic_grad_clip_value = config.get('critic_grad_clip_value', float('inf'))
        self.auto_alpha_learning = config.get('auto_alpha_learning', True)

        self.network_action_dim = int(config.action_dim)
        self.actor = Actor(config.state_dim, config.action_dim, config.hidden_dim).to(config.device)

        self.critic = Critic(config.state_dim,  config.action_dim).to(config.device)
       

        self.critic_target = Critic(config.state_dim,  config.action_dim).to(config.device)
       
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.init_alpha = self.config.get("init_alpha", 1.0)
        
        # entropy temperature
        if self.auto_alpha_learning:
            self.log_alpha = torch.nn.Parameter(
                torch.tensor([math.log(self.init_alpha)], device=self.device, requires_grad=True)
            )
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
            self.target_entropy = -float(self.network_action_dim)

        self.replay_action_dim = self.network_action_dim


    def _init_reply_buffer(self, config):
        if not self.init_reply:
            if self.replay_device == 'RAM':
                self.replay = ReplayBuffer(config.replay_capacity, (config.state_dim, ), self.replay_action_dim, self.device)
            elif self.replay_device == 'Disk':
                self.replay = ReplayBufferZarr(
                    config.replay_capacity, (config.state_dim, ), self.replay_action_dim, 
                    self.device, os.path.join(self.save_dir, 'replay_buffer.zarr'))
            self.init_reply = True


    # ---------------------- utils ----------------------
    def _process_obs_for_input(self, state) -> np.ndarray:
        return np.concatenate(state).flatten()

    def _process_context_for_input(self, context):
        context = np.stack(context, axis=0)
        
        context = torch.as_tensor(context, dtype=torch.float32, device=self.device)

        ## This is for integrating all garments.
        if context.shape[-1] < self.config.state_dim:
            base = torch.zeros((context.shape[0], self.config.state_dim), dtype=torch.float32, device=self.device)
            base[:, :context.shape[-1]] = context
            context = base
            
        return context


    def _select_action(self, info: dict, stochastic: bool = False):
        #print('info observation keys', info['observation'].keys())
        obs = [info['observation'][k] for k in self.obs_keys]
        
        obs = self._process_obs_for_input(obs)
        aid = info['arena_id']
        # maintain obs queue per arena
        if aid not in self.internal_states:
            self.reset([aid])
        self.internal_states[aid]['obs_que'].append(obs)
        while len(self.internal_states[aid]['obs_que']) < self.context_horizon:
            self.internal_states[aid]['obs_que'].append(obs)
        obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]
       

        # encoder expects either (B, C*N, H, W) or (B, C, N, H, W) depending on your ConvEncoder
        # follow the same pattern as the framework: reshape to (B, C, N, H, W) then pass
        # B = obs_t.shape[0]
        # C = self.each_image_shape[0]
        # N = self.context_horizon
        # H, W = self.each_image_shape[1:]
        # z = self.encoder(obs_t.reshape(B, C, N, H, W))

        if stochastic:
            #print('obs_t', obs)
            a, logp = self.actor.sample(self._process_context_for_input(obs_list))
            a = torch.clip(a, -self.config.action_range, self.config.action_range)
            a = a.detach().cpu().numpy().squeeze(0)
            return a, a #logp.detach().cpu().numpy().squeeze(0)
        else:
            mean, _ = self.actor(self._process_context_for_input(obs_list))
            action = torch.tanh(mean)
            action = torch.clip(action, -self.config.action_range, self.config.action_range)
            a = action.detach().cpu().numpy().squeeze(0)
            return a, a

    def act(self, info_list, updates=None):
        self.set_eval()
        with torch.no_grad():
            return [self._select_action(info, stochastic=False)[0] for info in info_list]

    # def explore_act(self, info_list):
    #     self.set_eval()
    #     with torch.no_grad():
    #         return [self._select_action(info, stochastic=True)[0] for info in info_list]

    def single_act(self, info, update=False):
        return self._select_action(info)[0]

    def set_eval(self):
        if getattr(self, 'mode', None) == 'eval':
            return
       
        self.actor.eval()
        self.critic.eval()
        self.mode = 'eval'

    def set_train(self):
        if getattr(self, 'mode', None) == 'train':
            return

        self.actor.train()
        self.critic.train()
        self.mode = 'train'

    def _soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        for p_src, p_tgt in zip(source.parameters(), target.parameters()):
            p_tgt.data.copy_(tau * p_src.data + (1 - tau) * p_tgt.data)

    # ---------------------- learning ----------------------
    def _update_networks(self, batch: dict):
        config = self.config
        context, action, reward, next_context, done = batch.values()
        B = context.shape[0]
        # print('sampled action', action)
        # C = self.each_image_shape[0]
        # N = self.context_horizon
        # H, W = self.each_image_shape[1:]

        

        # alpha loss
        if self.auto_alpha_learning:
            pi, log_pi = self.actor.sample(context)
            alpha = self.log_alpha.exp().detach().item()
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
        else:
            alpha = self.init_alpha


        # compute target Q
        with torch.no_grad():
            a_next, logp_next = self.actor.sample(next_context)
            q1_next, q2_next = self.critic_target(next_context, a_next)
            q_next = torch.min(q1_next, q2_next)
            target_q = reward + (1 - done) * config.gamma * (q_next - alpha * logp_next)

        # current Q estimates
        a_curr = action.view(B, -1).to(self.device)
        q1_pred, q2_pred = self.critic(context, a_curr)

        critic_loss = 0.5*(F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q))

        # optimize critics
        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_grad_clip_value)
        self.critic_optim.step()

        # actor loss
        pi, log_pi = self.actor.sample(context)
        q1_pi, q2_pi = self.critic(context, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * log_pi - min_q_pi).mean()
        #print('\nactor loss', actor_loss.item(), 'alpha', alpha.item())

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        
        # soft updates
        if self.update_steps % self.config.target_update_interval == 0:
            self._soft_update(self.critic, self.critic_target, config.tau)

        
        with torch.no_grad():
            q_stats = {
                'q_mean': min_q_pi.mean().item(),
                'q_max': min_q_pi.max().item(),
                'q_min': min_q_pi.min().item(),
                'logp_mean': log_pi.mean().item(),
                'logp_max': log_pi.max().item(),
                'alpha': alpha,
                'logp_min': log_pi.min().item(),
                'critic_grad_norm': critic_grad_norm.item(),
            }
            self.logger.log({f"diag/{k}": v for k,v in q_stats.items()}, step=self.act_steps)

        # logging
        self.logger.log({
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
        }, step=self.act_steps)

        if self.auto_alpha_learning:
            self.logger.log({
                'alpha_loss': alpha_loss.item()
            }, step=self.act_steps)

    # ---------------------- environment interaction ----------------------
    def _collect_from_arena(self, arena):
        if self.last_done:
            if self.info is not None:
                evaluation = self.info['evaluation']
                success = int(self.info['success'])

                for k, v in evaluation.items():
                    self.logger.log({
                        f"train/eps_lst_step_eval_{k}": v,
                    }, step=self.act_steps) 
                
                self.logger.log({
                    "train/episode_return": self.episode_return,
                    "train/episode_length": self.episode_length,
                    'train/episode_success': success,
                    'train/episode_sim_steps': self.sim_steps - self.last_sim_steps,
                    'train/total_sim_steps': self.sim_steps 
                }, step=self.act_steps)

                if success and self.save_success:
                    print('About to Save Success')
                    self.logger.log_frames(arena.get_frames(), key='success episodes', step=self.act_steps)

            self.info = arena.reset()
            if self.save_success:
                # print('set arena save videos')
                arena.save_video = True
            self.set_train()
            self.reset([arena.id])
            self.episode_return = 0.0
            self.episode_length = 0
            self.last_done = False
            self.last_sim_steps = self.sim_steps

        # sample stochastic action for exploration
        a, a_add = self._select_action(self.info, stochastic=True)
        #print(f'[vanilla sac] a {a}, a_add {a_add}',)
        # clip to action range
        #a = np.clip(a, -self.config.action_range, self.config.action_range)
        #dict_action = {'continuous': a}  # user should adapt to their arena's expected action format
        next_info = arena.step(a)
        if next_info.get("fail_step", False):
            self.last_done = True
            return

        
        
        next_obs_for_process =  self._get_next_obs_for_process(next_info)

        reward = next_info.get('reward', 0.0)[self.reward_key] if isinstance(next_info.get('reward', 0.0), dict) else next_info.get('reward', 0.0)
        self.logger.log(
            {'train/step_reward': reward}, step=self.act_steps
        )
        evaluation = next_info.get('evaluation', {})
        for k, v in evaluation.items():
            self.logger.log({
                f"train/{k}": v,
            }, step=self.act_steps)
        done = next_info.get('done', False)
        term = next_info.get('terminated', False)
        self.info = next_info
        #a = next_info['applied_action']
        self.last_done = done

        aid = arena.id
        obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]
        obs_for_replay = self._process_context_for_replay(obs_list)
        # print('obs stack', obs_stack)
        # print('next_obs', next_obs)
        # append next
        #print(next_obs_for_process)
        obs_list.append(self._process_obs_for_input(next_obs_for_process))
        next_obs_for_replay = self._process_context_for_replay(obs_list[-self.context_horizon:])
        #next_obs_stack = np.stack(obs_list)[-self.context_horizon:].flatten() #TODO: .reshape(self.context_horizon * self.each_image_shape[0], *self.each_image_shape[1:])

        #print('\napplied action vecotr', a, type(a))
        self._add_transition_replay(obs_for_replay, self._post_process_action_to_replay(a_add), reward, next_obs_for_replay, term)
        
        
        self.act_steps += 1
        self.sim_steps += next_info['sim_steps']
        self.episode_return += reward
        self.episode_length += 1
    
    def _add_transition_replay(self, obs_for_replay, a, reward, next_obs_for_replay, done):
        #print('[vanilla sac] action before add', a)
        self.replay.add(obs_for_replay, a, reward, next_obs_for_replay,  done)

    def _get_next_obs_for_process(self, next_info):
        next_obs = [next_info['observation'][k] for k in self.obs_keys]
        return next_obs


    def _post_process_action_to_replay(self, action):
        return action.astype(np.float32)

    def _process_context_for_replay(self, context):
        context = np.stack(context).flatten()

        ## This is for integrating all garments.
        if context.shape[0] < self.config.state_dim:
            base = np.zeros((self.config.state_dim), dtype=np.float32)
            base[:context.shape[-1]] = context
            context = base
            
        return context



    def train(self, update_steps, arenas) -> bool:
        if not self.init_reply:
            self._init_reply_buffer(self.config)

        if arenas is None or len(arenas) == 0:
            raise ValueError("SAC.train requires at least one Arena.")
        arena = arenas[0]
        print('arena mode', arena.get_mode())
        self.set_train()
        self.last_done = True
        #print('here update!!')
        with tqdm(total=update_steps, desc=f"{self.name} Training", initial=0) as pbar:
            while self.replay.size < self.initial_act_steps:
                self._collect_from_arena(arena)
                pbar.set_postfix(env_step=self.act_steps, updates=self.update_steps)

            for _ in range(update_steps):
                #print('here update')
                batch = self.replay.sample(self.config.batch_size)
                # optional data augmentation hook
                # if hasattr(self, 'data_augmenter') and callable(self.data_augmenter):
                #     #print('aguemtn!')
                #     self.data_augmenter(batch)
                self._update_networks(batch)
                self.update_steps += 1

                if self.update_steps % self.config.gradient_steps == 0:
                    for _ in range(self.config.train_freq):
                        self._collect_from_arena(arena)
                        pbar.set_postfix(
                            phase="training",
                            env_step=self.act_steps,
                            total_updates=self.update_steps,
                        )

                pbar.update(1)
                pbar.set_postfix(env_step=self.act_steps, updates=self.update_steps)

        return True

    # ---------------------- save/load ----------------------
    def save(self, path: Optional[str] = None, checkpoint_id: Optional[int] = None) -> bool:
        path = path or self.save_dir
        path = os.path.join(path, 'checkpoints')
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, 'last_model.pt')
        self._save_model(model_path)
        
        if checkpoint_id is not None:
            model_path = os.path.join(path, f'checkpoint_{checkpoint_id}.pt')
            self._save_model(model_path)

        replay_path = os.path.join(path, 'last_replay_buffer.pt')

        if self.save_replay:
            self._save_replay_buffer(replay_path)
        return True

    def _save_model(self, model_path):
        #os.makedirs(model_path, exist_ok=True)

        state = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'update_steps': self.update_steps,
            'act_steps': self.act_steps,
            'sim_steps': self.sim_steps,
           
        }

        if self.auto_alpha_learning:
            state['log_alpha'] =  self.log_alpha.detach().cpu()
            state['alpha_optim'] = self.alpha_optim.state_dict()

        
        torch.save(state, model_path)
    
    def save_best(self, path: Optional[str] = None) -> bool:
        path = path or self.save_dir
        path = os.path.join(path, 'checkpoints')
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, 'best_model.pt')
        self._save_model(model_path)
        # save replay
        replay_path = os.path.join(path, 'best_replay_buffer.pt')
        if self.save_replay:
            self._save_replay_buffer(replay_path)
        

        return True
    
    def _save_replay_buffer(self, replay_path):
        """Save the replay buffer to disk, handling both RAM and Zarr cases."""
        print('[VanillaSAC] Saving replay buffer ...')
        if self.replay_device == 'RAM':
            # Standard in-memory version
            torch.save({
                'ptr': self.replay.ptr,
                'size': self.replay.size,
                'capacity': self.replay.capacity,
                'observation': torch.from_numpy(self.replay.observation),
                'actions': torch.from_numpy(self.replay.actions),
                'rewards': torch.from_numpy(self.replay.rewards),
                'next_observation': torch.from_numpy(self.replay.next_observation),
                'dones': torch.from_numpy(self.replay.dones),
            }, replay_path)

        elif self.replay_device == 'Disk':
            # For Zarr version, we only save small metadata
            meta = {
                'ptr': self.replay.ptr,
                'size': self.replay.size,
                'capacity': self.replay.capacity,
                'zarr_path': str(self.replay.zarr_path)
            }
            torch.save(meta, replay_path)
            # Zarr arrays are already persisted automatically

        else:
            raise ValueError(f"Unknown replay device type: {self.replay_device}")
        print('[VanillaSAC] finished saving replay buffer!')
    
    def _load_model(self, model_path, resume=False):
        state = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])

        self.actor_optim.load_state_dict(state['actor_optim'])
        self.critic_optim.load_state_dict(state['critic_optim'])
        
        if self.auto_alpha_learning:
            # restore log_alpha safely
            log_alpha_value = state['log_alpha'].detach().clone().to(self.device)
            if log_alpha_value.ndim == 0:
                log_alpha_value = log_alpha_value.unsqueeze(0)
            self.log_alpha = torch.nn.Parameter(log_alpha_value.requires_grad_(True))
            print('loaded log alpha', self.log_alpha)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
            if 'alpha_optim' in state:
                self.alpha_optim.load_state_dict(state['alpha_optim'])
                for s in self.alpha_optim.state.values():
                    for k, v in s.items():
                        if torch.is_tensor(v) and v.shape == torch.Size([]):
                            s[k] = v.unsqueeze(0)

            print("log_alpha shape:", self.log_alpha.shape)
            for p in self.alpha_optim.param_groups[0]['params']:
                print("alpha_optim param shape:", p.shape)
        
        # Restore counters
        self.update_steps = state.get('update_steps', 0)
        self.act_steps = state.get('act_steps', 0)
        self.sim_steps = state.get('sim_steps', 0)
        self.last_sim_steps = self.sim_steps

        # # ---- W&B resume handling ----
        # run_id = state.get("wandb_run_id", None)
        # try:
        #     if resume and run_id is not None:
        #         print(f"[INFO] Attempting to resume W&B run ID: {run_id}")
        #         self.logger = WandbLogger(
        #             project=self.config.project_name,
        #             name=self.config.exp_name,
        #             config=dict(self.config),
        #             run_id=run_id,
        #             resume=True
        #         )
        #     else:
        #         raise ValueError("No valid run_id for resuming.")
        # except Exception as e:
        #     print(f"[WARN] W&B resume failed ({e}). Starting a new run instead.")
        #     self.logger = WandbLogger(
        #         project=self.config.project_name,
        #         name=self.config.exp_name,
        #         config=dict(self.config),
        #         resume=False
        #     )


    def load(self, path: Optional[str] = None) -> int:
        path = path or self.save_dir
        path = os.path.join(path, 'checkpoints')
        model_file = os.path.join(path, 'last_model.pt')
        replay_file = os.path.join(path, 'last_replay_buffer.pt')

        if not os.path.exists(model_file):
            return 0
        #     print(f"[WARN] Model file not found: {model_file}")

        #     self.logger = WandbLogger(
        #         project=self.config.project_name,
        #         name=self.config.exp_name,
        #         config=dict(self.config),
        #         resume=False
        #     )
            
        #     return 0

        self._load_model(model_file, resume=True)
        
        if self.save_replay:
            self._load_replay_buffer(replay_file)

        self.loaded = True
        return self.update_steps

    def _load_replay_buffer(self, replay_file):
        """Load replay buffer metadata (and data if using RAM)."""
        print('[VanillaSAC] loading replay buffer...')
        if not os.path.exists(replay_file):
            raise FileNotFoundError(f"Replay buffer file not found: {replay_file}")

        self._init_reply_buffer(self.config)
        replay_state = torch.load(replay_file, map_location='cpu')

        if self.replay_device == 'RAM':
            # Restore full in-memory buffer
            self.replay.ptr = replay_state['ptr']
            self.replay.size = replay_state['size']
            self.replay.capacity = replay_state['capacity']
            self.replay.observation = replay_state['observation'].cpu().numpy()
            self.replay.actions = replay_state['actions'].cpu().numpy()
            self.replay.rewards = replay_state['rewards'].cpu().numpy()
            self.replay.next_observation = replay_state['next_observation'].cpu().numpy()
            self.replay.dones = replay_state['dones'].cpu().numpy()

        elif self.replay_device == 'Disk':
            # Just reload Zarr arrays and metadata
            self.replay.ptr = replay_state['ptr']
            self.replay.size = replay_state['size']
            self.replay.capacity = replay_state['capacity']

            # Reopen the Zarr store (in case a new session started)
            zarr_path = replay_state.get('zarr_path', getattr(self.replay, "zarr_path", None))
            if zarr_path is None:
                raise ValueError("Missing zarr_path in replay state for Disk mode.")

            store = zarr.DirectoryStore(zarr_path)
            self.replay.root = zarr.open_group(store=store, mode='a')

            # Rebind dataset references
            self.replay.observation = self.replay.root["observation"]
            self.replay.actions = self.replay.root["actions"]
            self.replay.rewards = self.replay.root["rewards"]
            self.replay.next_observation = self.replay.root["next_observation"]
            self.replay.dones = self.replay.root["dones"]

        else:
            raise ValueError(f"Unknown replay device type: {self.replay_device}")
        print('[VanillaSAC] Finished loading replay buffer!')

    def load_best(self, path: Optional[str] = None) -> int:
        path = path or self.save_dir
        path = os.path.join(path, 'checkpoints')
        model_file = os.path.join(path, 'best_model.pt')
        replay_file = os.path.join(path, 'best_replay_buffer.pt')

        if not os.path.exists(model_file):
            print(f"[WARN] Model file not found: {model_file}")
            return 0

        self._load_model(model_file)
        if self.save_replay:
            self._load_replay_buffer(replay_file)

        self.loaded = True
        return self.update_steps

    def reset(self, arena_ids):
        for aid in arena_ids:
            self.internal_states[aid] = {}
            self.internal_states[aid]['obs_que'] = deque()