from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from collections import deque
from tqdm import tqdm
import math

from actoris_harena import TrainableAgent
from actoris_harena.utilities.logger.logger_interface import Logger
from dotmap import DotMap


from .networks import ConvEncoder, MLPActor, Critic  # expects networks similar to your repo
from .replay_buffer import ReplayBuffer
from .replay_buffer_zarr import ReplayBufferZarr
from .vanilla_sac import VanillaSAC
from .networks import NatureCNNEncoder


class Critic(nn.Module):
    def __init__(self, obs_shape, action_dim, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.encoder = NatureCNNEncoder(obs_shape, feature_dim)
        self.q1_1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.q1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_3 = nn.Linear(hidden_dim, 1)
       
        self.q2_1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.q2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_3 = nn.Linear(hidden_dim, 1)


    def forward(self, obs, action):
        h = self.encoder(obs)
        x = torch.cat([h, action], dim=-1)
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
    def __init__(self, obs_shape, action_dim, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.encoder = NatureCNNEncoder(obs_shape, feature_dim)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)


    def forward(self, obs):
        h = self.encoder(obs)
        x = F.relu(self.fc1(h))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mean, std


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

class VanillaImageSAC(VanillaSAC):
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
        #self.each_image_shape = cfg.each_image_shape
        

    def _make_actor_critic(self, cfg):
        C, H, W = cfg.each_image_shape
        self.input_channel = C * self.context_horizon
        obs_shape = (self.input_channel, H, W) 
        self.network_action_dim = int(self.config.action_dim)
        # actor and critics (two critics for twin-Q)
        self.action_dim = int(cfg.action_dim)
        self.actor = Actor(obs_shape, cfg.action_dim, cfg.feature_dim, cfg.hidden_dim).to(self.device)

        self.critic = Critic(obs_shape, cfg.action_dim, cfg.feature_dim).to(cfg.device)
        self.critic_grad_clip_value = self.config.get('critic_grad_clip_value', float('inf'))

        self.critic_target = Critic(obs_shape, cfg.action_dim, cfg.feature_dim).to(cfg.device)
       
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.init_alpha = self.config.get("init_alpha", 1.0)
        # entropy temperature
        self.auto_alpha_learning = self.config.get('auto_alpha_learning', True)

        if self.auto_alpha_learning:
            self.log_alpha = torch.nn.Parameter(
                torch.tensor([math.log(self.init_alpha)], device=self.device, requires_grad=True)
            )
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
            self.target_entropy = -float(self.network_action_dim)

        self.replay_action_dim = self.network_action_dim

    def _init_reply_buffer(self, cfg):
        
        C, H, W = cfg.each_image_shape
        self.input_channel = C * self.context_horizon
        obs_shape = (self.input_channel, H, W) 

        self.replay = ReplayBuffer(cfg.replay_capacity, obs_shape, self.action_dim, self.device)

    def _init_reply_buffer(self, config):
        C, H, W = config.each_image_shape
        self.input_channel = C * self.context_horizon
        obs_shape = (self.input_channel, H, W)
        
        if not self.init_reply:
            if self.replay_device == 'RAM':
                self.replay = ReplayBuffer(config.replay_capacity, obs_shape, self.replay_action_dim, self.device)
            elif self.replay_device == 'Disk':
                self.replay = ReplayBufferZarr(
                    config.replay_capacity, obs_shape, self.replay_action_dim, 
                    self.device, os.path.join(self.save_dir, 'replay_buffer.zarr'))
            self.init_reply = True

    def _process_obs_for_input(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.concatenate(rgb, axis=-1) 
        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32)
        rgb_resized = cv2.resize(rgb, (self.config.each_image_shape[2], self.config.each_image_shape[1]), interpolation=cv2.INTER_AREA)
        return rgb_resized.transpose(2, 0, 1)

    def _process_context_for_input(self, context):
        
        rgb = torch.as_tensor(context, dtype=torch.float32, device=self.device)
        #state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return rgb
    
    def _process_context_for_replay(self, context):
        return np.stack(context).reshape(
            self.config.context_horizon * self.config.each_image_shape[0], 
            *self.config.each_image_shape[1:])