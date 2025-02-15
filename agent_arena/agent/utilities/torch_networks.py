import os

import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torch.distributions as td

from agent.utilities.torch_utils import *

class ActionHead(nn.Module):

    def __init__(
        self, inp_dim, size, layers, units, act=nn.ELU, dist='trunc_normal',
        init_std=0.0, min_std=0.1, action_disc=5, temp=0.1, outscale=0):
        
        super(ActionHead, self).__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        self._min_std = min_std
        self._init_std = init_std
        self._action_disc = action_disc
        self._temp = temp() if callable(temp) else temp
        self._outscale = outscale

        pre_layers = []
        for index in range(self._layers):
            pre_layers.append(nn.Linear(inp_dim, self._units))
            pre_layers.append(act())
            if index == 0:
                inp_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)

        if self._dist in ['tanh_normal','tanh_normal_5','normal','trunc_normal']:
            self._dist_layer = nn.Linear(self._units, 2 * self._size)
        elif self._dist in ['normal_1','onehot','onehot_gumbel', 'deter', 'clip_deter']:
            self._dist_layer = nn.Linear(self._units, self._size)

    def __call__(self, features, dtype=None):
        x = features
        x = self._pre_layers(x)
        if self._dist == 'deter':
            dist = self._dist_layer(x)
        elif self._dist == 'clip_deter':
            dist= torch.clamp(self._dist_layer(x), -1, 1)
        elif self._dist == 'tanh_normal':
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = td.normal.Normal(mean, std)
            dist = td.transformed_distribution.TransformedDistribution(
                dist, TanhBijector())
            dist = td.independent.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self._dist == 'tanh_normal_5':
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = td.normal.Normal(mean, std)
            dist = td.transformed_distribution.TransformedDistribution(
                dist, TanhBijector())
            dist = td.independent.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self._dist == 'normal':
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = td.normal.Normal(mean, std)
            dist = ContDist(td.independent.Independent(dist, 1))
        elif self._dist == 'normal_1':
            x = self._dist_layer(x)
            dist = td.normal.Normal(mean, 1)
            dist = ContDist(td.independent.Independent(dist, 1))
        elif self._dist == 'trunc_normal':
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size]*2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = SafeTruncatedNormal(mean, std, -1, 1)
            dist = ContDist(td.independent.Independent(dist, 1))
        elif self._dist == 'onehot':
            x = self._dist_layer(x)
            dist = OneHotDist(x)
        elif self._dist == 'onehot_gumble':
            x = self._dist_layer(x)
            temp = self._temp
            dist = ContDist(td.gumbel.Gumbel(x, 1/temp))
        else:
            raise NotImplementedError(self._dist)
        return dist




class GRUCell(nn.Module):

    def __init__(self, inp_size,
                size, norm=False, act=torch.tanh, update_bias=-1):
      super(GRUCell, self).__init__()
      self._inp_size = inp_size
      self._size = size
      self._act = act
      self._norm = norm
      self._update_bias = update_bias
      self._layer = nn.Linear(inp_size+size, 3*size,
                              bias=norm is not None)
      if norm:
          self._norm = nn.LayerNorm(3*size)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        #state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size]*3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output #, [output]

class ConvEncoder(nn.Module):

    def __init__(
        self, depth=32, act=nn.ReLU, 
        kernels=(4, 4, 4, 4), in_channel=3):
        super(ConvEncoder, self).__init__()
        self._act = act
        self._depth = depth
        self._kernels = kernels

        layers = []
        for i, kernel in enumerate(self._kernels):
            if i == 0:
                inp_dim = in_channel
            else:
                inp_dim = 2 ** (i-1) * self._depth
            depth = 2 ** i * self._depth
            layers.append(nn.Conv2d(inp_dim, depth, kernel, 2))
            layers.append(act())
        self.layers = nn.Sequential(*layers)

    def __call__(self, obs):
        x = obs['input_image'].reshape((-1,) + tuple(obs['input_image'].shape[-3:]))
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        shape = list(obs['input_image'].shape[:-3]) + [x.shape[-1]]
        return x.reshape(shape)

class ConvDecoder(nn.Module):

    def __init__(
          self, inp_depth,
          depth=32, act=nn.ReLU, shape=(3, 64, 64), kernels=(5, 5, 6, 6),
          thin=True, output_mode='stochastic'):
        super(ConvDecoder, self).__init__()
        self._inp_depth = inp_depth
        self._act = act
        self._depth = depth
        self._shape = shape
        self._kernels = kernels
        self._thin = thin
        self._output_mode = output_mode

        if self._thin:
            self._linear_layer = nn.Linear(inp_depth, 32 * self._depth)
        else:
            self._linear_layer = nn.Linear(inp_depth, 128 * self._depth)
        inp_dim = 32 * self._depth

        cnnt_layers = []
        for i, kernel in enumerate(self._kernels):
            depth = 2 ** (len(self._kernels) - i - 2) * self._depth
            act = self._act
            if i == len(self._kernels) - 1:
                #depth = self._shape[-1]
                depth = self._shape[0]
                act = None
            if i != 0:
                inp_dim = 2 ** (len(self._kernels) - (i-1) - 2) * self._depth
            cnnt_layers.append(nn.ConvTranspose2d(inp_dim, depth, kernel, 2))
            if act is not None:
                cnnt_layers.append(act())
        self._cnnt_layers = nn.Sequential(*cnnt_layers)

    def __call__(self, features, dtype=None):
        if self._thin:
            x = self._linear_layer(features)
            x = x.reshape([-1, 1, 1, 32 * self._depth])
            x = x.permute(0,3,1,2)
        else:
            x = self._linear_layer(features)
            x = x.reshape([-1, 2, 2, 32 * self._depth])
            x = x.permute(0,3,1,2)
        x = self._cnnt_layers(x)
        mean = x.reshape(features.shape[:-1] + self._shape)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._output_mode == 'deterministic':
            return mean
        elif self._output_mode == 'stochastic':
            return ContDist(td.independent.Independent(
            td.normal.Normal(mean, 1), len(self._shape)))
        else:
            raise NotImplementedError


class DenseHead(nn.Module):

    def __init__(
        self, inp_dim,
        shape, layers, units, act=nn.ELU, 
        dist='normal', std=1.0, output_mode='stochastic'):
        super(DenseHead, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if len(self._shape) == 0:
            self._shape = (1,)
        self._layers = layers
        self._units = units
        self._act = act
        self._dist = dist
        self._std = std
        self._output_mode = output_mode

        mean_layers = []
        for index in range(self._layers):
            mean_layers.append(nn.Linear(inp_dim, self._units))
            mean_layers.append(act())
            if index == 0:
                inp_dim = self._units
        mean_layers.append(nn.Linear(inp_dim, np.prod(self._shape)))
        self._mean_layers = nn.Sequential(*mean_layers)

        if self._std == 'learned':
            self._std_layer = nn.Linear(self._units, np.prod(self._shape))

    def __call__(self, features, dtype=None):
        x = features
        mean = self._mean_layers(x)
        if self._output_mode == 'deterministic':
            return mean

        if self._std == 'learned':
            std = self._std_layer(x)
            std = torch.softplus(std) + 0.01
        else:
            std = self._std
        if self._dist == 'normal':
            return ContDist(td.independent.Independent(
              td.normal.Normal(mean, std), len(self._shape)))
        if self._dist == 'huber':
            return ContDist(td.independent.Independent(
                UnnormalizedHuber(mean, std, 1.0), len(self._shape)))
        if self._dist == 'binary':
            return Bernoulli(td.independent.Independent(
              td.bernoulli.Bernoulli(logits=mean), len(self._shape)))
        raise NotImplementedError(self._dist)