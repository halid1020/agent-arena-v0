#code from https://github.com/Xingyu-Lin/softagent/blob/master/planet/models.py

import os


import numpy as np
import math
import random
import cv2
from tqdm import tqdm
from typing import Optional, List
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torch.distributions as td
import torchvision

from agent.utilities.torch_utils import *
# from logger.visualisation_utils import plot_trajectory
from torch.autograd import Variable

def symlog(x, flag):
    if flag:
        return torch.sign(x) * torch.log(1 + torch.abs(x))
    return x

def symexp(x, flag):
    if flag:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    return x

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, 
                 hidden_dim, kernel_size, bias):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channel of input tensor.
        :param hidden_dim: int
            Number of channel of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """

        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


# ## TODO: We can play with the number of layers and hidden units
# class ActionEncoder(nn.Module):
#     """Encode Action to a feacture map"""

#     def __init__(self, input_channel, output_channel, hidden_channel,
#                  feature_map_dim, activation_function='relu'):
#         super(ActionEncoder, self).__init__()

#         ## from B*C_in*1*1 to B*C_hid*H*W
#         self.act_fn = getattr(F, activation_function)
#         self.encoder = nn.Sequential(
#             nn.ConvTranspose2d(input_channel, hidden_channel, 5, stride=2),
#             ACTIVATIONS[activation_function](),
#             nn.ConvTranspose2d(hidden_channel, output_channel, 5, stride=2),
#             # self.act_fn,
#             # nn.ConvTranspose2d(64, 32, 6, stride=2),
#             # self.act_fn,
#             # nn.ConvTranspose2d(32, output_channel, 6, stride=2)
#         )
#         self.image_dim = feature_map_dim

#     def forward(self, x):

#         ## Input x is B*T*d
#         ## Reshape action to (B*T)*d*1*1

#         before_shape = x.shape[:-1]
#         d = x.shape[-1]
#         x = x.reshape(-1, d, 1, 1)

#         x = self.encoder(x)
#         x = F.interpolate(x, size=(self.image_dim[0], self.image_dim[1]), mode='bilinear', align_corners=False)

#         x = x.view(*before_shape, x.shape[-3], x.shape[-2], x.shape[-1])
#         ## Output x is B*T*C*H*W
#         return x
    
class ActionEncoder(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_channels,
                 feature_map_dim, activation_function='relu',
                 num_layers=2, kernel_size=[5, 5], stride=[2, 2], 
                 padding=[2, 2]):
        super(ActionEncoder, self).__init__()

        self.act_fn = getattr(F, activation_function)
        self.num_layers = num_layers

        layers = []
        in_channels = input_channel
        out_channels = hidden_channels

        for i in range(num_layers):
            layers.append(nn.ConvTranspose2d(in_channels, out_channels[i], 
                kernel_size=kernel_size[i], stride=stride[i], padding=padding[i]))
            layers.append(ACTIVATIONS[activation_function]())
            in_channels = out_channels[i]

        layers.append(nn.ConvTranspose2d(in_channels, output_channel, 
            kernel_size=kernel_size[-1], stride=stride[-1], padding =padding[-1]))
        
        self.encoder = nn.Sequential(*layers)
        self.image_dim = feature_map_dim

    def forward(self, x):
        before_shape = x.shape[:-1]
        d = x.shape[-1]
        x = x.reshape(-1, d, 1, 1)

        x = self.encoder(x)
        x = F.interpolate(x, size=(self.image_dim[0], self.image_dim[1]), mode='bilinear', align_corners=False)

        x = x.view(*before_shape, x.shape[-3], x.shape[-2], x.shape[-1])

        return x


    
class ImageEncoder(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_channels,
                 feature_map_dim, activation_function='relu',
                 num_conv_layers=3,
                 kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=[1, 1, 1]):
        super().__init__()
        self.image_dim = feature_map_dim
        self.act_fn = getattr(F, activation_function)
        self.num_layers = num_conv_layers

        layers = []
        in_channels = input_channel
        out_channels = hidden_channels

        for i in range(self.num_layers):
            layers.append(
                nn.Conv2d(in_channels, out_channels[i], 
                    kernel_size=kernel_size[i], stride=stride[i], padding=padding[i]))
            layers.append(ACTIVATIONS[activation_function]())
            in_channels = out_channels[i]

        layers.append(nn.Conv2d(in_channels, output_channel, 
            kernel_size=kernel_size[-1], 
            stride=stride[-1], 
            padding=padding[-1]))
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        before_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])

        x = self.encoder(x)

        x = F.interpolate(x, size=(self.image_dim[0], self.image_dim[1]), mode='bilinear', align_corners=False)

        x = x.view(*before_shape, x.shape[-3], x.shape[-2], x.shape[-1])

        return x
    
class ImageDecoder(nn.Module):
    def __init__(self, belief_channel, state_channel, hidden_channels,
                 image_dim, activation_function='relu', output_mode=None,
                 num_conv_layers=3,
                 kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=[1, 1, 1]):
        super().__init__()
        self.image_dim = image_dim
        self.act_fn = getattr(F, activation_function)
        self.output_mode = output_mode

        input_channel = belief_channel + state_channel

        layers = []
        in_channels = input_channel
        out_channels = hidden_channels

        for i in range(num_conv_layers):
            layers.append(nn.ConvTranspose2d(in_channels, out_channels[i], 
                kernel_size=kernel_size[i], stride=stride[i], padding=padding[i]))
            layers.append(ACTIVATIONS[activation_function]())
            in_channels = out_channels[i]

        layers.append(nn.ConvTranspose2d(in_channels, image_dim[0], 
            kernel_size=kernel_size[-1], stride=stride[-1], padding=padding[-1]))

        self.decoder = nn.Sequential(*layers)

    def forward(self, belief, latent_state):
        x = torch.cat([belief, latent_state], dim=1)

        x = self.decoder(x)

        x = F.interpolate(x, size=(self.image_dim[1], self.image_dim[2]), mode='bilinear', align_corners=False)

        shape = x.shape[1:]
        if self.output_mode == 'normal':
            x = td.Independent(td.Normal(x, 1), len(shape))

        return x
    
class RewardModel(nn.Module):
    def __init__(self, belief_channel, state_channel, hidden_channels, feature_map_dim,
                 activation_function='relu', output_mode=None,
                 num_conv_layers=2, conv_kernel_sizes=[3, 3], conv_strides=[1, 2], conv_paddings=[1, 1],
                 num_linear_layers=1, linear_hidden_dims=[],
                 linear_input_dim=0):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.num_conv_layers = num_conv_layers
        self.num_linear_layers = num_linear_layers

        # Convolutional Layers
        conv_layers = []
        in_channels = belief_channel + state_channel
        out_channels = hidden_channels


        for i in range(num_conv_layers):
            conv_layers.append(
                nn.Conv2d(in_channels, out_channels[i], 
                          kernel_size=conv_kernel_sizes[i], 
                          stride=conv_strides[i], padding=conv_paddings[i]))
            conv_layers.append(ACTIVATIONS[activation_function]())
            in_channels = out_channels[i]

        self.conv_layers = nn.Sequential(*conv_layers)

        # Linear Layers
        linear_layers = []
        linear_input_dim = hidden_channels[-1] * feature_map_dim[0] * feature_map_dim[1] \
              if linear_input_dim == 0 else linear_input_dim
        prev_layer_dim = linear_input_dim

        for i in range(num_linear_layers):
            if i < len(linear_hidden_dims):
                linear_layers.append(nn.Linear(prev_layer_dim, linear_hidden_dims[i]))
                prev_layer_dim = linear_hidden_dims[i]
                linear_layers.append(ACTIVATIONS[activation_function]())

        linear_layers.append(nn.Linear(prev_layer_dim, 1))
        self.linear_layers = nn.Sequential(*linear_layers)

        self.output_mode = output_mode
        self.feature_map_dim = feature_map_dim

    def forward(self, beliefs, states):
        before_shape = beliefs.shape[:-3]
        beliefs = beliefs.reshape(-1, *beliefs.shape[-3:])
        states = states.reshape(-1, *states.shape[-3:])
        
        x = torch.cat([beliefs, states], dim=1)
        B = x.shape[0]

        hidden = self.conv_layers(x)
        hidden = hidden.reshape(B, -1)

        x = self.linear_layers(hidden)

        x = x.view(*before_shape)

        if self.output_mode == 'normal':
            shape = x.shape
            reward = td.Independent(td.Normal(x, 1), len(shape))
            return reward

        return x

    

# ## TODO: can play with the layers of convoltion and lnear layers. 
# class RewardModel(nn.Module):
#     def __init__(self, belief_channel, state_channel, 
#                  hidden_channel, feature_map_dim, activation_function='relu', output_mode=None):
#         super().__init__()
#         self.act_fn = getattr(F, activation_function)
#         self.conv1 = nn.Conv2d(belief_channel + state_channel, hidden_channel, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(hidden_channel, 1, kernel_size=3, padding=1)
#         self.linear = nn.Linear(feature_map_dim[0]*feature_map_dim[1], 1)
#         self.output_mode = output_mode
#         self.feature_map_dim = feature_map_dim

#     def forward(self, beliefs, states):
#         before_shape = beliefs.shape[:-3]
#         beliefs = beliefs.reshape(-1, *beliefs.shape[-3:])
#         states = states.reshape(-1, *states.shape[-3:])
        
#         x = torch.cat([beliefs, states], dim=1)
#         B = x.shape[0]

#         hidden = self.act_fn(self.conv1(x))
#         hidden = self.act_fn(self.conv2(hidden))
#         # print('hidden shape', hidden.shape)
       
#         hidden = hidden.reshape(B, -1)

#         # print('hidden shape', hidden.shape)


#         x = self.linear(hidden)

#         x = x.view(*before_shape)

#         if self.output_mode == 'normal':
#             shape = x.shape
#             reward = td.Independent(td.Normal(x, 1), len(shape))
#             return reward

#         return x
    
    
    

class TransitionModel(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(
            self, belief_channel, state_channel, action_embedding_channel, 
            hidden_channel, obs_embedding_channel, feature_map_dim, 
            activation_function='relu', min_std_dev=0.1, embedding_layers=1):
        
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.rnn = ConvGRUCell(feature_map_dim, belief_channel, 
                               belief_channel, kernel_size=(3, 3), bias=False) # nn.GRUCell(belief_size, belief_size)
        
        self.state_channel = state_channel
        
        self.fc_embed_state_action = self.make_layers(state_channel+action_embedding_channel, belief_channel, hidden_channel, embedding_layers)
        self.fc_embed_belief_prior = self.make_layers(belief_channel, hidden_channel, hidden_channel, embedding_layers)
        self.fc_state_prior = self.make_layers(hidden_channel, 2 * state_channel, hidden_channel, embedding_layers)
        self.fc_embed_belief_posterior = self.make_layers(belief_channel + obs_embedding_channel, hidden_channel, hidden_channel, embedding_layers)
        self.fc_state_posterior = self.make_layers(hidden_channel, 2 * state_channel, hidden_channel, embedding_layers)

    def make_layers(self, input_channel, output_channel, hidden_channel, num_layers):
        if num_layers == 1:
            return nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)

        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Conv2d(input_channel, hidden_channel, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            input_channel = hidden_channel

        layers.append(nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1))

        return nn.Sequential(*layers)

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    def forward(self, prev_state:torch.Tensor, actions:torch.Tensor, prev_belief:torch.Tensor, observations:Optional[torch.Tensor]=None,
                nonterminals:Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = \
             [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
        
        # print('action dim', actions.shape)
        # print('posterior_states dim', posterior_states[0].shape)
        # print('prior_states dim', prior_states[0].shape)
        # Loop over time sequence
        for t in range(T - 1):
            # print('t', t)
            # if nonterminals is not None:
            #     print('nonterminals', nonterminals[t].shape)

            _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
            #print('_state dim', _state.shape)
            _state = _state if nonterminals is None else torch.mul(_state, nonterminals[t].unsqueeze(2).unsqueeze(3)) # Mask if previous transition was terminal

            # Compute belief (deterministic hidden state)
            #print('state dim', _state.shape)

            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.act_fn(self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
        # Return new hidden states
        hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
        return hidden


class ConvRSSM():

    def __init__(self, config):
        self.config = config
        self.config.models_dir = os.path.join(self.config.save_dir, 'model')
        if not os.path.exists(config.models_dir):
            os.makedirs(config.models_dir)
        
        self.model = dict()
        self.model['transition_model'] = TransitionModel(
            belief_channel=self.config.belief_channel,
            state_channel=self.config.state_channel,
            action_embedding_channel = self.config.act_embedding_channel,
            hidden_channel=self.config.hidden_channel,
            obs_embedding_channel=self.config.obs_embedding_channel,
            activation_function=self.config.activation,
            feature_map_dim = self.config.feature_map_dim,
            min_std_dev=self.config.min_std_dev,
            embedding_layers=self.config.trans_layers
        ).to(self.config.device)

        self.model['observation_model'] = ImageDecoder(
            belief_channel=config.belief_channel,
            state_channel=config.state_channel,
            hidden_channels=config.image_decoder_hidden_channel,
            image_dim = config.observation_dim,
            num_conv_layers=config.image_decoder_num_conv_layers,
            kernel_size=config.image_decoder_kernel_size,
            stride=config.image_decoder_stride,
            padding=config.image_decoder_padding,
            activation_function=config.activation,
        ).to(config.device)

        self.model['reward_model'] = RewardModel(
            belief_channel=self.config.belief_channel,
            state_channel=self.config.state_channel,
            num_conv_layers=self.config.reward_num_conv_layers,
            hidden_channels=self.config.reward_hidden_channels,
            conv_kernel_sizes=self.config.reward_conv_kernel_sizes,
            conv_strides=self.config.reward_conv_strides,
            conv_paddings=self.config.reward_conv_paddings,
            feature_map_dim=self.config.feature_map_dim,
            num_linear_layers=self.config.reward_num_linear_layers,
            linear_hidden_dims=self.config.reward_linear_hidden_dims,
            activation_function = self.config.activation
        ).to(self.config.device)

        self.model['encoder'] = ImageEncoder(
            input_channel = self.config.observation_dim[0],
            output_channel= self.config.obs_embedding_channel,
            hidden_channels = self.config.image_encoder_hidden_channel,
            feature_map_dim = self.config.feature_map_dim,
            num_conv_layers=self.config.image_encoder_num_conv_layers,
            kernel_size=self.config.image_encoder_kernel_size,
            stride=self.config.image_encoder_stride,
            padding=self.config.image_encoder_padding,
            activation_function=self.config.activation,
        ).to(config.device)

        self.model['action_encoder'] = ActionEncoder(
            input_channel = self.config.action_dim,
            output_channel= self.config.act_embedding_channel,
            num_layers=self.config.action_encoder_num_layers,
            kernel_size=self.config.action_encoder_kernel_size,
            stride=self.config.action_encoder_stride,
            padding=self.config.action_encoder_padding,
            hidden_channels = self.config.action_encoder_hidden_channel,
            feature_map_dim = self.config.feature_map_dim,
            activation_function=self.config.activation,
        ).to(config.device)

        params = [list(m.parameters()) for m in self.model.values()]
        self.param_list = []
        for p in params:
            self.param_list.extend(p)

        optimiser_params = self.config.optimiser_params.copy()
        optimiser_params['params'] = self.param_list
        self.optimiser = OPTIMISER_CLASSES[self.config.optimiser_class](**optimiser_params)
        self.symlog = self.config.symlog
        

    def init(self, state):
        obs = state['observation']
        self.load_models(os.path.join(self.config.save_dir, 'model/model.pth'))
        # image = self.transform(
        #     {'observation': np.expand_dims(cv2.resize(obs['image'], (64, 64)), axis=0).transpose(0, 3, 1, 2)}, 
        #     train=False)['observation']
        # TODO: add a preprocess procedure, so it does not have to use transform.
        
        
        image = np_to_ts(
            np.expand_dims(cv2.resize(obs['image'], (64, 64)), axis=0).transpose(0, 3, 1, 2)/255.0 - 0.5,
            self.config.device)
        image = symlog(image, self.symlog)
        self.cur_state = {
            'deterministic_latent_state': torch.zeros(
                1, self.config.belief_channel, *self.config.feature_map_dim).to(self.config.device),
            'stochastic_latent_state': {
                'sample': torch.zeros(
                    1, self.config.state_channel, *self.config.feature_map_dim).to(self.config.device)
            },
            'image': image.unsqueeze(0) # batch*horizon*C*H*W
        }
    
        action = np_to_ts(np.asarray(self.config.no_op), self.config.device).unsqueeze(0).unsqueeze(0) # B*H*action_dim
        self.cur_state = self.unroll_state_action_(self.cur_state , action)

        return self.cur_state

    def update_state(self, state, action):
        # image = self.transform(
        #     {'observation': np.expand_dims(cv2.resize(obs['image'], (64, 64)), axis=0).transpose(0, 3, 1, 2)}, 
        #     train=False)['observation']
        obs = state['observation']
        image = np_to_ts(
            np.expand_dims(cv2.resize(obs['image'], (64, 64)), axis=0).transpose(0, 3, 1, 2)/255.0 - 0.5,
            self.config.device)
        image = symlog(image, self.symlog)

        self.cur_state['stochastic_latent_state']['sample'] = self.cur_state['stochastic_latent_state']['sample'].squeeze(dim=1)
        self.cur_state['deterministic_latent_state'] = self.cur_state['deterministic_latent_state'].squeeze(dim=1)
        self.cur_state['image'] = image.unsqueeze(0) # batch*horizon*C*H*W
        
        action = np_to_ts(action.flatten(), self.config.device).unsqueeze(0).unsqueeze(0)
        self.cur_state  = self.unroll_state_action_(self.cur_state , action)

        return self.cur_state

    def cost_fn(self, trajectory):

        #reward_pred = self.reward_pred
        planning_horizon = trajectory['deterministic_latent_state'].shape[0]
        returns = self.model['reward_model'](
            trajectory['deterministic_latent_state'].view(-1, self.config.belief_channel, *self.config.feature_map_dim), 
            trajectory['stochastic_latent_state']['sample'].view(-1, self.config.state_channel, *self.config.feature_map_dim))\
                .view(planning_horizon, -1).sum(dim=0)

        return -returns.detach().cpu().numpy()

    def unroll_action_from_cur_state(self, action):
        to_unroll = {}
        candidates = action.shape[0]
        state = self.cur_state
        to_unroll['deterministic_latent_state'] = state['deterministic_latent_state']\
                .squeeze(dim=1).expand(1, candidates, self.config.belief_channel, *self.config.feature_map_dim)\
                .reshape(-1, self.config.belief_channel, *self.config.feature_map_dim)
        
        to_unroll['stochastic_latent_state'] = {
            'sample': state['stochastic_latent_state']['sample']\
                .squeeze(dim=1).expand(1, candidates, self.config.state_channel, *self.config.feature_map_dim)\
                .reshape(-1, self.config.state_channel, *self.config.feature_map_dim)
        }
        
        action = np_to_ts(action, self.config.device).squeeze(2).permute(1, 0, 2) ## horizon*candidates*actions

        return self.unroll_action_(to_unroll, action)
    
    def visual_reconstruct(self, state):

        images = bottle(self.model['observation_model'], 
                        (state['deterministic_latent_state'], state['stochastic_latent_state']['sample']))
        
        images = ((ts_to_np(images).transpose(1, 0, 3, 4, 2) + 0.5)*255.0).clip(0, 255).astype(np.uint8)

        return images

    def reward_pred(self):
        return lambda a : symexp(self.model['reward_model'](a), self.symlog)
    
    
    def set_eval(self):
        for v in self.model.values():
            v.eval()

    def set_train(self):
        for v in self.model.values():
            v.train()

    def save_checkpoint(self, update_step, name):
        model_dict = {'transition_model': self.model['transition_model'].state_dict(),
                'observation_model': self.model['observation_model'].state_dict(),
                'reward_model': self.model['reward_model'].state_dict(),
                'encoder': self.model['encoder'].state_dict(),
                'update_step': update_step,
                'optimiser': self.optimiser.state_dict()}
    
        torch.save(
            model_dict, 
            os.path.join('{}.pth'.format(name))
        )
    
    def load_models(self, model_dir=None):

        if model_dir is None:
            model_dir = os.path.join(self.config.save_dir, 'model/model.pth')

        if not os.path.exists(model_dir):
            return 0
        
        checkpoint = torch.load(model_dir)

        self.model['transition_model'].load_state_dict(checkpoint['transition_model'])
        self.model['observation_model'].load_state_dict(checkpoint['observation_model'])
        self.model['reward_model'].load_state_dict(checkpoint['reward_model'])
        self.model['encoder'].load_state_dict(checkpoint['encoder'])
        self.optimiser.load_state_dict(checkpoint['optimiser'])
        
        return  checkpoint['update_step'] + 1
    
    def preprocess(self, data):      
        for k, v in data.items():
            data[k] = torch.swapaxes(v, 0, 1)

        
        T, B, C, H, W = data['observation'].shape
        data['observation'] = F.interpolate(
            data['observation'].reshape(T*B, C, H, W),
            size=(64, 64), mode='bilinear', align_corners=False)\
                .reshape(T, B, C, 64, 64)
        
        data['observation'] = symlog(data['observation'], self.symlog)
        data['reward'] = symlog(data['reward'], self.symlog)
        
        return data

    def train(self, datasets, env, loss_logger, eval_logger):
        
        train_dataset = datasets['train']
        test_dataset = datasets['test']
        self.transform=test_dataset.transform
        
        #torch.multiprocessing.set_start_method('spawn', force=True)
        losses_dict = {}
        updates = []
        start_step = self.load_models(os.path.join(self.config.save_dir, 'model/model.pth'))
        #print('start_step', start_step)
        self.set_train()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            # prefetch_factor=10,
            # num_workers=2,
            # multiprocessing_context='spawn',
            shuffle=True)
    
        for u in tqdm(range(start_step, self.config.update_steps)):
        
            data = next(iter(train_dataloader))
            data = self.preprocess(data)
            

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
            updates.append(u)
            
            if u%self.config.test_interval == 0:
                self.set_eval()

                # Save Losses
                losses_dict.update({'update_step': updates})
                loss_logger(losses_dict, self.config)
                losses_dict = {}
                updates = []

                # Evaluate & Save
                test_results = self.evaluate(test_dataset)
                train_results = self.evaluate(train_dataset)
                results = {'test_{}'.format(k): v for k, v in test_results.items()}
                results.update({'train_{}'.format(k): v for k, v in train_results.items()})
                results['update_step'] = [u]

                eval_logger(results, self.config)
                #break # Change here
                
                
                # Save Model
                self.save_checkpoint(u, os.path.join(self.config.models_dir, 'model'))
            

                self.set_train()

    def evaluate(self, dataset):

        reward_rmses = {h:[] for h in self.config.test_horizons}
        observation_mses = {h:[] for h in self.config.test_horizons}
        kls_post_to_prior = {h:[] for h in self.config.test_horizons}
        kls_prior_to_post = {h:[] for h in self.config.test_horizons}
        prior_entropies = {h:[] for h in self.config.test_horizons}
        posterior_reward_rmses = []
        posterior_recon_mses = []
        posterior_entropies = []
        eval_action_horizon = dataset.eval_action_horizon
        
        for i in tqdm(range(self.config.eval_episodes)):
            episode = dataset.get_episode(i, transform=True, train=False)
            
            for k, v in episode.items():
                episode[k] = v.unsqueeze(0)
            episode = self.preprocess(episode)
            for k, v in episode.items():
                episode[k] = torch.swapaxes(v, 0, 1).squeeze(0)

            init_belief = torch.zeros(
                1, 
                self.config.belief_channel, *self.config.feature_map_dim).to(self.config.device)

            init_state = torch.zeros(
                1, 
                self.config.state_channel, *self.config.feature_map_dim).to(self.config.device)

            no_op_ts = np_to_ts(self.config.no_op, self.config.device).unsqueeze(0)
            actions = episode['action'][:eval_action_horizon]

            actions = torch.cat([no_op_ts, actions])
            
            observations = episode['observation'][:eval_action_horizon+1]
            rewards = episode['reward'][:eval_action_horizon]

            beliefs, posteriors, priors = self._unroll_state_action(
                observations.unsqueeze(1), actions.unsqueeze(1), init_belief, init_state, None)
            

            posterior_reward =  bottle(self.model['reward_model'], (beliefs[1:], posteriors['sample'][1:])).transpose(0, 1).squeeze(0)
            posterior_observation = bottle(self.model['observation_model'], (beliefs[1:], posteriors['sample'][1:])).transpose(0, 1).squeeze(0)

            post_dist = ContDist(td.independent.Independent(
                td.normal.Normal(posteriors['mean'][1:], posteriors['std'][1:]), 1))
            
        


            posterior_entropies.extend(post_dist.entropy().mean(dim=-1).flatten().detach().cpu().tolist())

            posterior_reward_rmses.extend((F.mse_loss(
                        symexp(posterior_reward, self.symlog), 
                        symexp(rewards, self.symlog),
                        reduction='none')**0.5).flatten().detach().cpu().tolist())
            
            posterior_recon_mses.extend(F.mse_loss(
                        symexp(posterior_observation, self.symlog),
                        symexp(observations[1:], self.symlog),
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

                
                imagin_reward =  bottle(self.model['reward_model'], (imagin_beliefs, imagin_priors['sample'])).transpose(0, 1)
                imagin_observation = bottle(self.model['observation_model'], (imagin_beliefs, imagin_priors['sample'])).transpose(0, 1)
                
                true_reward = torch.stack([episode['reward'][j+1: j+horizon+1] for j in range(eval_action_horizon-horizon-1)]).reshape(-1, horizon)
                true_image = torch.stack([episode['observation'][j+2: j+horizon+2] for j in range(eval_action_horizon-horizon-1)])


                reward_rmses[horizon].extend((F.mse_loss(
                        symexp(imagin_reward, self.symlog), 
                        symexp(true_reward, self.symlog),
                        reduction='none')**0.5).flatten().detach().cpu().tolist())
                
                observation_mses[horizon].extend(F.mse_loss(
                        symexp(imagin_observation, self.symlog) , 
                        symexp(true_image, self.symlog),
                        reduction='none').mean((2, 3, 4)).flatten().detach().cpu().tolist())
                
                imagin_post = {k: torch.stack([posteriors[k][j+2:j+2+horizon, 0] for j in range(eval_action_horizon-horizon-1)]) for k in posteriors.keys()}


                imagin_post_dist = ContDist(td.independent.Independent(
                td.normal.Normal(imagin_post['mean'].transpose(0, 1), imagin_post['std'].transpose(0, 1)), 1))._dist
                
                
                
                imagin_prior_dist = ContDist(td.independent.Independent(
                td.normal.Normal(imagin_priors['mean'], imagin_priors['std']), 1))._dist
                
                

                kls_post_to_prior[horizon].extend(td.kl.kl_divergence(imagin_post_dist, imagin_prior_dist).flatten().detach().cpu().tolist())
                kls_prior_to_post[horizon].extend(td.kl.kl_divergence(imagin_prior_dist, imagin_post_dist).flatten().detach().cpu().tolist())

                

                prior_entropies[horizon].extend(imagin_prior_dist.entropy().mean(dim=-1).flatten().detach().cpu().tolist())
                
        
        results = {
            'img_prior_reward_rmse': {h:reward_rmses[h] for h in self.config.test_horizons},
            'img_prior_rgb_observation_mse': {h:observation_mses[h] for h in self.config.test_horizons},
            'kl_divergence_between_posterior_and_img_prior': {h:kls_post_to_prior[h] for h in self.config.test_horizons},
            'img_prior_entropy':  {h:prior_entropies[h] for h in self.config.test_horizons}
        }

        res = {
            'posterior_rgb_observation_mse_mean': np.mean(posterior_recon_mses),
            'posterior_rgb_observation_mse_std': np.std(posterior_recon_mses),
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
        
        for e in range(3):
            data = dataset.get_episode(e, transform=True, train=False)
            org_gt = dataset.transform.post_transform(data)

            for k, v in data.items():
                data[k] = v.unsqueeze(0)
            data = self.preprocess(data)
            for k, v in data.items():
                data[k] = torch.swapaxes(v, 0, 1).squeeze(0)

            recon_image = []

        
            plot_trajectory(
                org_gt['observation'][6:16].transpose(0, 2 ,3, 1),
                org_gt['action'][6:16],
                title='{} Ground Truth Episode {}'.format(train_str, e),
                # rewards=data['reward'][5:15], 
                save_png = True, 
                save_path=os.path.join(self.config.save_dir, 'visualisations'))
            

            init_belief = torch.zeros(
                1, 
                self.config.belief_channel, *self.config.feature_map_dim).to(self.config.device)

            init_state = torch.zeros(
                1, 
                self.config.state_channel, *self.config.feature_map_dim).to(self.config.device)

            no_op_ts = np_to_ts(self.config.no_op, self.config.device).unsqueeze(0)
            actions = np_to_ts(data['action'], self.config.device)
            actions = torch.cat([no_op_ts, actions])

            observations = np_to_ts(data['observation'], self.config.device)
            rewards = np_to_ts(data['reward'], self.config.device)

            beliefs, posteriors, priors = self._unroll_state_action(
                observations.unsqueeze(1), actions.unsqueeze(1), init_belief, init_state, None)

            posterior_observations = bottle(self.model['observation_model'], (beliefs, posteriors['sample'])).squeeze(1)
            posterior_observations = symexp(posterior_observations, self.symlog)
            post_process_obs = dataset.transform.post_transform({'observation': posterior_observations})['observation']
            
            posterior_rewards = bottle(self.model['reward_model'], (beliefs, posteriors['sample'])).squeeze(1)
            posterior_rewards = symexp(posterior_rewards, self.symlog)
            posterior_rewards = posterior_rewards.detach().cpu().numpy()

            

            plot_trajectory(
                post_process_obs[6:16].transpose(0, 2 ,3, 1),
                # rewards=posterior_rewards[6:16], 
                title='{} Posterior Trajectory Episode {}'.format(train_str, e), 
                save_png = True,
                save_path=os.path.join(self.config.save_dir, 'visualisations'))
            
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
                post_process_img_obs = dataset.transform.post_transform({'observation': prior_observations[i]})['observation']
                plot_trajectory(
                    post_process_img_obs[4-i:14-i].transpose(0, 2 ,3, 1),
                    # rewards=prior_rewards[i][5-i:15-i], 
                    title='{}-Step {} Prior Trajectory Episode {}'.format(i, train_str, e), 
                    save_png = True,
                    save_path=os.path.join(self.config.save_dir, 'visualisations'))
                recon_image.append(post_process_img_obs[5+5:6+5].transpose(0, 2 ,3, 1))

            recon_image = np.concatenate(recon_image, axis=0)
            plot_trajectory(
                    recon_image,
                    # rewards=posterior_rewards[6:16], 
                    title='{} Recon Trajectory Episode {}'.format(train_str, e), 
                    save_png = True,
                    save_path=os.path.join(self.config.save_dir, 'visualisations'))
    

    def _unroll_state_action(self, obs, acts, blf, lst, non_terminals):
       
        blfs, prior_states_, prior_means_, prior_std_devs_, posterior_states_, posterior_means_, posterior_std_devs_ = \
            self.model['transition_model'](
                lst, 
                self.model['action_encoder'](acts), 
                blf, 
                self.model['encoder'](obs), 
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
        
        return blfs, posteriors_, priors_

    def unroll_state_action_(self, state, action):

        blfs, prior_states_, prior_means_, prior_std_devs_, posterior_states_, posterior_means_, posterior_std_devs_ = \
            self.model['transition_model'](
                
                state['stochastic_latent_state']['sample'],
                self.model['action_encoder'](action), 
                state['deterministic_latent_state'], 
                self.model['encoder'](state['image']), 
                None)

        posteriors_ = {
            'sample': posterior_states_,
            'mean': posterior_means_,
            'std': posterior_std_devs_
        }

        return {
            'deterministic_latent_state': blfs,
            'stochastic_latent_state': posteriors_
        }
    
    
    

    def _unroll_action(self, actions, belief_, latent_state_):


        img_beliefs_, prior_states_, prior_means_, prior_std_devs_  = \
            self.model['transition_model'](
                latent_state_, 
                self.model['action_encoder'](actions), 
                belief_, 
                None,
                None)

        priors_ = {
            'sample': prior_states_,
            'mean': prior_means_,
            'std': prior_std_devs_
        }
        
        return img_beliefs_, priors_

    def unroll_action_(self, init_state, actions):
        img_beliefs_, prior_states_, prior_means_, prior_std_devs_  = \
            self.model['transition_model'](
                init_state['stochastic_latent_state']['sample'], 
                self.model['action_encoder'](actions), 
                init_state['deterministic_latent_state'], 
                None,
                    None)
        
        return {
            'deterministic_latent_state': img_beliefs_,
            'stochastic_latent_state': {
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
            seq_pad = (0, 0, 0, 0, 0, 0, 0, 0, 0, t - d + self.config.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch

            # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) posterior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
            overshooting_vars.append((
                F.pad(actions[t:d], seq_pad[4:]), 
                F.pad(non_terminals[t:d].unsqueeze(2), seq_pad[4:]), 
                F.pad(rewards[t:d], seq_pad[6:]), 
                beliefs[t_], 
                posteriors['sample'][t_].detach(), 
                F.pad(posteriors['mean'][t_ + 1:d_ + 1].detach(), seq_pad), 
                F.pad(posteriors['std'][t_ + 1:d_ + 1].detach(), seq_pad, value=1), 
                F.pad(torch.ones(d - t, self.config.batch_size, 
                                 self.config.state_channel, *self.config.feature_dim, device=self.config.device), seq_pad[4:])
            ))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences

            #print('posterior tmp mean shape', F.pad(posteriors['mean'][t_ + 1:d_ + 1].detach(), seq_pad).shape)
        

        overshooting_vars = tuple(zip(*overshooting_vars))
        

        # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
        beliefs, prior_states, prior_means, prior_std_devs = self.model['transition_model'](
            torch.cat(overshooting_vars[4], dim=0), 
            self.model['action_encoder'](torch.cat(overshooting_vars[0], dim=1)), 
            torch.cat(overshooting_vars[3], dim=0), 
            None, 
            torch.cat(overshooting_vars[1], dim=1))

        reward_seq_mask = torch.cat(overshooting_vars[7], dim=1)
        
        

        # Calculate overshooting KL loss with sequence mask
        #print('posterior mean shape', posteriors['mean'].shape)
        # for t in overshooting_vars[5]:
        #     print('shape')
        #     print(t.shape)

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
            #print('here')
            kld = td.kl.kl_divergence
            post = ContDist(td.independent.Independent(
                    td.normal.Normal(post['mean'],post['std']), 1))
            
            prior = ContDist(td.independent.Independent(
                    td.normal.Normal(prior['mean'], prior['std']), 1))
            loss =  kld(post._dist, prior._dist)
            loss = torch.maximum(torch.mean(loss), torch.Tensor([free])[0])
            

        return loss


    def compute_losses(self, experience, steps):
        
        # Create initial belief and state for time t = 0
        init_belief = torch.zeros(
            self.config.batch_size, 
            self.config.belief_channel, *self.config.feature_map_dim).to(self.config.device)

        init_state = torch.zeros(
            self.config.batch_size, 
            self.config.state_channel, *self.config.feature_map_dim).to(self.config.device)

        

        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)

        actions = experience['action']
        non_terminals = 1 - experience['terminal']
        rewards = experience['reward']
        observations = experience['observation']
            
        beliefs, posteriors, priors = self._unroll_state_action(
            observations[1:], actions[:-1], 
            init_belief, init_state, non_terminals[:-1].unsqueeze(-1))
        

        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); 
        # sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)

        observation_loss = F.mse_loss(
            bottle(self.model['observation_model'], (beliefs, posteriors['sample'])), 
            observations[1:-1],
            reduction='none')

        observation_loss = observation_loss[:, :, :, :, :].sum(dim=(2, 3, 4)).mean(dim=(0, 1))

        # print('belief shape', beliefs.shape)
        # print('posteriors shape', posteriors['sample'].shape)

        

       

        if self.config.reward_gradient_stop:
            pred_rewards = self.model['reward_model'](beliefs.detach(), posteriors['sample'].detach())
        else:
            pred_rewards = self.model['reward_model'](beliefs, posteriors['sample'])

        # print('pred_rewards', pred_rewards.shape)
        # print('rewards', rewards.shape)

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
                self.unscaled_overshooting_losses(experience, beliefs, posteriors)

        if self.config.kl_overshooting_warmup:
            kl_overshooting_scale_ = 1.0*steps/self.config.update_steps*self.config.kl_overshooting_scale
        else:
            kl_overshooting_scale_= self.config.kl_overshooting_scale

        if self.config.reward_overshooting_warmup:
            reward_overshooting_scale_ = 1.0*steps/self.config.update_steps*self.config.reward_overshooting_scale
        else:
            reward_overshooting_scale_= self.config.reward_overshooting_scale

        total_loss = self.config.rgb_observation_scale*observation_loss + \
            self.config.reward_scale*reward_loss + \
            self.config.kl_scale * kl_loss + \
            kl_overshooting_scale_ * kl_overshooting_loss + \
            reward_overshooting_scale_ * reward_overshooting_loss

        res = {
            'total_loss': total_loss,
            'rgb_observation_loss': observation_loss,
            'reward_loss': reward_loss,
            'kl_loss': kl_loss,
            "posterior_entropy": posterior_entropy,
            "prior_entropy": prior_entropy,
            "kl_overshooting_loss": kl_overshooting_loss,
            "reward_overshooting_loss": reward_overshooting_loss
        }

        return res 