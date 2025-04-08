from typing import Optional, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])


# Wraps the input tuple for a function to process a time x batch x chunk x features sequence in batch x features (assumes one output)
def bottle3(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1] * x[1][2], *x[1][3:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    return y.view(x_sizes[0][0], x_sizes[0][1], x_sizes[0][2], *y_size[1:])


def symlog(x, flag):
    if flag:
        #print('no here')
        return torch.sign(x) * torch.log(1 + torch.abs(x))
    return x

def symexp(x, flag):
    if flag:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    return x

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
        return output 
    

class TransitionModel(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1, embedding_layers=1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.rnn = nn.GRUCell(belief_size, belief_size)
        
        self.fc_embed_state_action = self.make_layers(state_size+action_size, belief_size, hidden_size, embedding_layers)
        self.fc_embed_belief_prior = self.make_layers(belief_size, hidden_size, hidden_size, embedding_layers)
        self.fc_state_prior = self.make_layers(hidden_size, 2 * state_size, hidden_size, embedding_layers)
        self.fc_embed_belief_posterior = self.make_layers(belief_size + embedding_size, hidden_size, hidden_size, embedding_layers)
        self.fc_state_posterior = self.make_layers(hidden_size, 2 * state_size, hidden_size, embedding_layers)

    def make_layers(self, input_dim, output_dim, hidden_dim, num_layers):

        if num_layers == 1:
            return nn.Linear(input_dim, output_dim)

        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, output_dim))

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
        
        
        # Loop over time sequence
        for t in range(T - 1):
            _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
            _state = _state if nonterminals is None else _state * nonterminals[t]  # Mask if previous transition was terminal
  
            # Compute belief (deterministic hidden state)

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


class RewardModel(nn.Module):
    def __init__(self, belief_size, state_size, hidden_size, 
                 activation_function='relu', output_mode=None):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.output_mode = output_mode

    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=-1)
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)

        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        output = self.fc3(hidden)

        shape = output.shape[1:]

        reward = output.reshape((*batch_shape, *shape))

        if self.output_mode == 'normal':
            reward = td.Independent(td.Normal(reward, 1), len(shape))
        return reward.squeeze(-1)
