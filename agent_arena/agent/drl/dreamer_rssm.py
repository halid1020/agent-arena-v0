#code from https://github.com/Xingyu-Lin/softagent/blob/master/planet/models.py
import os

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torch.distributions as td

from agent.utilities.torch_utils import *
from agent.utilities.torch_networks import *
from agent.algorithm.dreamer_rssm_utils import *
# from logger.visualisation_utils import *

# from registration.data_transformer import *


def symlog(x, flag):
    if flag:
        return torch.sign(x) * torch.log(1 + torch.abs(x))
    return x

def symexp(x, flag):
    if flag:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    return x

class RSSM(nn.Module):

    def __init__(
        self, stoch=30, deter=200, hidden=200, layers_input=1, layers_output=1,
        rec_depth=1, shared=False, discrete=False, act=nn.ELU,
        mean_act='none', std_act='softplus', temp_post=True, min_std=0.1,
        cell='gru',
        num_actions=None, embed = None, device=None):

        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._rec_depth = rec_depth
        self._shared = shared
        self._discrete = discrete
        self._act = act
        self._mean_act = mean_act
        self._std_act = std_act
        self._temp_post = temp_post
        self._embed = embed
        self._device = device


        inp_layers = []

        
       
        
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        
        if self._shared:
            inp_dim += self._embed
        
        for i in range(self._layers_input):
            inp_layers.append(nn.Linear(inp_dim, self._hidden))
            inp_layers.append(self._act())
            if i == 0:
                inp_dim = self._hidden
        self._inp_layers = nn.Sequential(*inp_layers)

        if cell == 'gru':
            self._cell = GRUCell(self._hidden, self._deter)
        elif cell == 'gru_layer_norm':
            self._cell = GRUCell(self._hidden, self._deter, norm=True)
        else:
            raise NotImplementedError(cell)

        img_out_layers = []
        inp_dim = self._deter
        for i in range(self._layers_output):
            img_out_layers.append(nn.Linear(inp_dim, self._hidden))
            img_out_layers.append(self._act())
            if i == 0:
                inp_dim = self._hidden
        self._img_out_layers = nn.Sequential(*img_out_layers)

        obs_out_layers = []
        if self._temp_post:
            inp_dim = self._deter + self._embed
        else:
            inp_dim = self._embed
        for i in range(self._layers_output):
            obs_out_layers.append(nn.Linear(inp_dim, self._hidden))
            obs_out_layers.append(self._act())
            if i == 0:
                inp_dim = self._hidden
        self._obs_out_layers = nn.Sequential(*obs_out_layers)

        if self._discrete:
            self._ims_stat_layer = nn.Linear(self._hidden, self._stoch*self._discrete)
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch*self._discrete)
        else:
            self._ims_stat_layer = nn.Linear(self._hidden, 2*self._stoch)
            self._obs_stat_layer = nn.Linear(self._hidden, 2*self._stoch)
 
    def initial(self, batch_size):
      deter = torch.zeros(batch_size, self._deter).to(self._device)
      if self._discrete:
          state = dict(
              logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
              stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
              deter=deter)
      else:
          state = dict(
              mean=torch.zeros([batch_size, self._stoch]).to(self._device),
              std=torch.zeros([batch_size, self._stoch]).to(self._device),
              stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
              deter=deter)
      return state

    def observe(self, embed, action, state=None, terminal=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        
        if state is None:
            state = self.initial(action.shape[0])
        
        if terminal is None:
            terminal = torch.zeros(*action.shape[:2]).to(self._device)

        embed, action, terminal = swap(embed), swap(action), swap(terminal)

        post, prior = static_scan(
            lambda prev_state, prev_act, embed, term: self.obs_step(
                prev_state[0], prev_act, embed, terminal=term),
            (action, embed, terminal), (state, state))
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior


    def get_dist(self, state, dtype=None):
      if self._discrete:
          logit = state['logit']
          dist = td.independent.Independent(OneHotDist(logit), 1)
      else:
          mean, std = state['mean'], state['std']
          dist = ContDist(td.independent.Independent(
              td.normal.Normal(mean, std), 1))
      return dist

    def obs_step(self, prev_state, prev_action, embed, sample=True, terminal=None):
        prior = self.img_step(prev_state, prev_action, None,  sample, terminal)
        if self._shared:
            post = self.img_step(prev_state, prev_action, embed, sample, terminal)
        else:
            if self._temp_post:
                x = torch.cat([prior['deter'], embed], -1)
            else:
                x = embed
            x = self._obs_out_layers(x)
            stats = self._suff_stats_layer('obs', x)
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mode()
            post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, embed=None, sample=True, prev_terminal=None):
      # Mask if previous transition was terminal

      prev_stoch = prev_state['stoch']
      if self._discrete:
          shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
          prev_stoch = prev_stoch.reshape(shape)
      if prev_terminal is not None:
          prev_stoch = prev_stoch * (1-prev_terminal) 

      if self._shared:
          if embed is None:
              shape = list(prev_action.shape[:-1]) + [self._embed]
              embed = torch.zeros(shape)
          x = torch.cat([prev_stoch, prev_action, embed], -1)
      else:
          x = torch.cat([prev_stoch, prev_action], -1)
      x = self._inp_layers(x)
      for _ in range(self._rec_depth): # rec depth is not correctly implemented
          deter = prev_state['deter']
          if prev_terminal is not None: ### Changed here
            deter = deter * (1-prev_terminal) 
          x = self._cell(x, deter)
          deter = x
          #deter = deter[0]  # Keras wraps the state in a list.
      x = self._img_out_layers(x)
      stats = self._suff_stats_layer('ims', x)
      if sample:
          stoch = self.get_dist(stats).sample()
      else:
          stoch = self.get_dist(stats).mode()
      prior = {'stoch': stoch, 'deter': deter, **stats}
      return prior

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == 'ims':
              x = self._ims_stat_layer(x)
            elif name == 'obs':
              x = self._obs_stat_layer(x)
            else:
              raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {'logit': logit}
        
        else:
            if name == 'ims':
                x = self._ims_stat_layer(x)
            elif name == 'obs':
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch]*2, -1)
            mean = {
                'none': lambda: mean,
                'tanh5': lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                'softplus': lambda: F.softplus(std),
                'abs': lambda: torch.abs(std + 1),
                'sigmoid': lambda: torch.sigmoid(std),
                'sigmoid2': lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {'mean': mean, 'std': std}

    def kl_loss(self, post, prior, forward, balance, free, scale):
        kld = td.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)

        value_lhs = value = kld(dist(lhs) if self._discrete else dist(lhs)._dist,
                                dist(sg(rhs)) if self._discrete else dist(sg(rhs))._dist)
        value_rhs = kld(dist(sg(lhs)) if self._discrete else dist(sg(lhs))._dist,
                        dist(rhs) if self._discrete else dist(rhs)._dist)
        loss_lhs = torch.maximum(torch.mean(value_lhs), torch.Tensor([free])[0])
        loss_rhs = torch.maximum(torch.mean(value_rhs), torch.Tensor([free])[0])
        loss = mix * loss_lhs + (1 - mix) * loss_rhs

        
        loss *= scale
        return loss, value



class WorldModel(nn.Module):

    def __init__(self, config):
        super(WorldModel, self).__init__()
        self._step = config.update_steps
        self._use_amp = False
        self.config = config
        self.encoder = ConvEncoder(
            config.cnn_depth, ACTIVATIONS[config.act], 
            config.encoder_kernels, in_channel=config.channels)

        if config.size[0] == 64 and config.size[1] == 64:
            embed_size = 2 ** (len(config.encoder_kernels)-1) * config.cnn_depth
            embed_size *= 2 * 2
        else:
            raise NotImplemented(f"{config.size} is not applicable now")
        self.dynamics = RSSM(
            config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
            config.dyn_input_layers, config.dyn_output_layers,
            config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
            ACTIVATIONS[config.act], config.dyn_mean_act, config.dyn_std_act,
            config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
            config.num_actions, embed_size, config.device)
        self.heads = nn.ModuleDict()
        channels = config.channels
        shape = (channels,) + tuple(config.size)
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads['image'] = ConvDecoder(
            feat_size,  # pytorch version
            config.cnn_depth, ACTIVATIONS[config.act], shape, config.decoder_kernels,
            config.decoder_thin, output_mode=config['decoder_output_mode'])
        self.heads['reward'] = DenseHead(
            feat_size,  # pytorch version
            [], config.reward_layers, config.units, ACTIVATIONS[config.act],
            output_mode=config['reward_output_mode'])
        
        if config.pred_discount:
            self.heads['discount'] = DenseHead(
                feat_size,  # pytorch version
                [], config.discount_layers, config.units, ACTIVATIONS[config.act], dist='binary')

        for name in config.grad_heads:
            assert name in self.heads, name

        self._model_opt = Optimizer(
            'model', self.parameters(), config.model_lr, config.opt_eps, config.grad_clip,
            config.weight_decay, opt=config.opt,
            use_amp=self._use_amp)
        self._scales = dict(
            reward=config.reward_scale, discount=config.discount_scale)


    def _train(self, data ,current_step):

        data = self.preprocess(data, train=True)
    
        with RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed[:, 1:], 
                    data['action'].float(), 
                    state=None, 
                    terminal=data['terminal'].float()) ### Big change here


                kl_balance = schedule(self.config.kl_balance, self._step)
                kl_free = schedule(self.config.kl_free, self._step)
                kl_scale = schedule(self.config.kl_scale, self._step)
                kl_loss, kl_value = self.dynamics.kl_loss(
                    post, prior, self.config.kl_forward, kl_balance, kl_free, kl_scale)

               
                if self.config.overshooting_distance > 0:
                    overshooting_kl_balance = schedule(self.config.overshooting_kl_balance, self._step)
                    overshooting_kl_free = schedule(self.config.overshooting_kl_free, self._step)
                    overshooting_kl_scale = schedule(self.config.overshooting_kl_scale, self._step)

                    if self.config.kl_overshooting_warmup:
                        overshooting_kl_scale = 1.0*current_step/self.config.update_steps*self.config.overshooting_kl_scale
                    
                    horizon_actions = [data['action'][:, j+1: j+self.config.overshooting_distance+1] \
                        for j in range(self.config.sequence_size-self.config.overshooting_distance-1)]
                    horizon_actions = torch.stack(horizon_actions).float()
                    P, B, T, X = horizon_actions.shape
                    
                    horizon_actions = horizon_actions.reshape(P*B, T, X)
                    init_posterior = {k: v[:, 0:self.config.sequence_size-self.config.overshooting_distance-1].detach()\
                        .transpose(0, 1).reshape(P*B, *v.shape[2:]) for k, v in post.items()}
                    
                    imagin_prior = self.dynamics.imagine(horizon_actions, init_posterior)
                    imagin_post = {
                        k: torch.stack([v[:, j:j+self.config.overshooting_distance] \
                            for j in range(self.config.sequence_size-self.config.overshooting_distance-1)]).detach()\
                            .float().reshape(P*B, T, *v.shape[2:])
                        for k, v in post.items()
                    }

                    # print('overshoting kl balance, free, scale')
                    # print( overshooting_kl_balance, 
                    #     overshooting_kl_free, 
                    #     overshooting_kl_scale)
        
                    overshooting_kl_loss, overshooting_kl_value = self.dynamics.kl_loss(
                        imagin_post, imagin_prior, 
                        self.config.overshooting_kl_forward, 
                        overshooting_kl_balance, 
                        overshooting_kl_free, 
                        overshooting_kl_scale)

                
                losses = {}
                likes = {}
                for name, head in self.heads.items():
                    grad_head = (name in self.config.grad_heads)
                    feat = get_feat(post, self.config)
                    feat = feat if grad_head else feat.detach()
                    scale =  self._scales.get(name, 1.0)
                    target_data_name = name
                    if name == 'reward' and self.config.reward_gradient_stop:
                        feat = feat.detach()
                    if name == 'image':
                        target_data_name = 'target_image'

                    
                    pred = head(feat)

                    if name == 'image':
                        if self.config.decoder_loss == 'nll':  
                            like = pred.log_prob(data[target_data_name][:, 1:]) # Big change here
                            likes[name] = like
                            losses[name] = -torch.mean(like) * scale
                        elif self.config.decoder_loss == 'mse':
                            loss = F.mse_loss(pred, data[target_data_name][:, 1:], reduction='none').sum(dim=(2, 3, 4))
                            losses[name] = loss.mean() * scale
                        elif self.config.decoder_loss == 'bce':
                            loss = F.binary_cross_entropy(F.sigmoid(pred), data[target_data_name][:, 1:], reduction='none')
                            losses[name] = loss.mean() * scale
                        else:
                            raise NotImplementedError
                            
                    elif name == 'reward':
                        if self.config.reward_loss == 'nll':
                            like = pred.log_prob(data[target_data_name]) # Big change here
                            likes[name] = like
                            losses[name] = -torch.mean(like) * scale
                        elif self.config.reward_loss == 'mse':
                            loss = F.mse_loss(pred, data[target_data_name], reduction='none')
                            losses[name] = loss.mean() * scale
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                
                if self.config.overshooting_distance > 0 and self.config.overshooting_reward_scale > 0:
                    feat = get_feat(imagin_prior, self.config)
                    if self.config.reward_gradient_stop:
                        feat = feat.detach()
                    
                    imagin_reward = self.heads['reward'](feat)
                    true_reward = torch.stack([data['reward'][:, j+1: j+self.config.overshooting_distance+1] \
                        for j in range(self.config.sequence_size-self.config.overshooting_distance-1)]).float().reshape(P*B, T, 1)
                    
                    overshooting_reward_scale = self.config.overshooting_reward_scale
                    if self.config.reward_overshooting_warmup:
                        overshooting_reward_scale = 1.0*current_step/self.config.update_steps*self.config.overshooting_reward_scale

                    if self.config.reward_loss == 'nll':
                        like = imagin_reward.log_prob(true_reward)
                        likes['overshooting_reward'] = like
                        losses['overshooting_reward'] = -torch.mean(like) * overshooting_reward_scale
                    elif self.config.reward_loss == 'mse':
                        like = F.mse_loss(imagin_reward, true_reward, reduction='none')
                        likes['overshooting_reward'] = loss
                        losses['overshooting_reward'] = torch.mean(like) * overshooting_reward_scale
                    else:
                        raise NotImplementedError

                
                # if self.config.overshooting_distance > 0 and self.config.overshooting_observation_scale > 0:
                #     imagin_image = self.heads['image'](get_feat(imagin_prior, self.config))
                #     true_observation = torch.stack([data['input_image'][:, j+1: j+self.config.overshooting_distance+1] \
                #         for j in range(self.config.sequence_size-self.config.overshooting_distance-1)]).float().reshape(P*B, T, 64, 64, 3) # TODO magic number
                #     like = imagin_image.log_prob(true_observation)
                #     likes['overshooting_observation'] = like
                #     overshooting_observation_scale = self.config.overshooting_observation_scale
                #     if self.config.overshooting_warmup:
                #         overshooting_observation_scale = 1.0*current_step/self.config.update_steps*self.config.overshooting_observation_scale

                #     losses['overshooting_observation'] = -torch.mean(like) * overshooting_observation_scale


                model_loss = sum(losses.values()) + kl_loss + (overshooting_kl_loss if self.config.overshooting_distance > 0 else 0)
            metrics = self._model_opt(model_loss, self.parameters())

        metrics.update({f'{name}_loss': ts_to_np(loss) for name, loss in losses.items()})
        metrics['kl_balance'] = kl_balance
        metrics['kl_free'] = kl_free
        metrics['kl_scale'] = kl_scale
        metrics['kl'] = ts_to_np(torch.mean(kl_value))

        if self.config.overshooting_distance > 0:
            metrics['overshooting_kl_balance'] = overshooting_kl_balance
            metrics['overshooting_kl_free'] = overshooting_kl_free
            metrics['overshooting_kl_scale'] = overshooting_kl_scale
            metrics['overshooting_kl'] = ts_to_np(torch.mean(overshooting_kl_value))


        with torch.cuda.amp.autocast(self._use_amp):
            metrics['prior_ent'] = ts_to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
            metrics['post_ent'] = ts_to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
            context = dict(
                embed=embed, feat=get_feat(post, self.config),
                kl=kl_value, postent=self.dynamics.get_dist(post).entropy())

        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    def preprocess(self, data, train=True):

        data = self.config.transform(data, train=train)

        # for k, v in data.items():
        #     data[k] = torch.swapaxes(v, 0, 1)

        
        if self.config.input_obs == 'rgbd':
            T, B, C, H, W = data['rgb'].shape
            data['input_image'] = symlog(torch.cat([data['rgb'], data['depth']], axis=2), self.config.symlog)\
                .permute(0, 1, 3, 4, 2)
        
        else:
            data['input_image'] = symlog(data[self.config.input_obs], self.config.symlog)\
            .permute(0, 1, 3, 4, 2)
        
        if self.config.output_obs == 'input_obs':
            data['target_image'] = data['input_image']
        else:
                
            data['target_image'] = symlog(data[self.config.output_obs] , self.config.symlog) \
            .permute(0, 1, 3, 4, 2)


        if 'reward' in data.keys():
            if self.config.clip_rewards == 'tanh':
                data['reward'] = torch.tanh(data['reward']).unsqueeze(-1).float()
            elif self.config.clip_rewards == 'identity':
                data['reward'] = data['reward'].unsqueeze(-1) 
            else:
                raise NotImplemented(f' not implemented')
        
        if 'terminal' in data.keys():
            data['terminal'] = data['terminal'].unsqueeze(-1)
            

        return data

class Dreamer():

    def __init__(self, config):
       
        self.config = config
        if self.config.output_obs == 'input_obs':
            self.config.output_obs = self.config.input_obs
        transform_config = self.config.transform
        self.transform = TRANSORMER[transform_config.name](transform_config.params)
        
        self.config.transform = self.transform
        self.model = WorldModel(config)
        self.model = self.model.to(config.device, dtype=torch.float)
        

    def cost_fn(self, trajectory):
        rewards = self.model.heads['reward'](get_feat(trajectory, self.config))
        if self.config.reward_output_mode == 'stochastic':
            rewards = rewards.mode()
        returns = rewards.sum(dim=1).squeeze(1)
        return -returns.detach().cpu().numpy()

    def unroll_action_from_cur_state(self, action):
        state = self.cur_state
        B = state['stoch'].shape[0]
        candidiates = action.shape[0]
        state = {k: v.expand(B, candidiates, *v.shape[2:]).reshape(B*candidiates, *v.shape[2:]) for k, v in state.items()}

        imagin_prior = self.model.dynamics.imagine(np_to_ts(action, self.config.device).squeeze(2), state)
        return imagin_prior

    def init(self, state):
        configs = self.config
        obs = state['observation']['image']

        
        data = {
            self.config.input_obs: np.expand_dims(obs, axis=(0, 1)).transpose(0, 1, 4, 2, 3),
            'action': np.expand_dims(np.asarray(configs.no_op), axis=(0, 1))
        }

        data = self.model.preprocess(data, train=False)
        embed = self.model.encoder(data)
        lst, _ =  self.model.dynamics.observe(embed, data['action'].float())
        self.cur_state = lst

        return lst

    def update_state(self, state, action):
        obs = state['observation']['image']
        
        data = {
            self.config.input_obs: np.expand_dims(obs, axis=(0, 1)).transpose(0, 1, 4, 2, 3),
            'action': np.expand_dims(np.asarray(action), (0, 1))
        }

        data = self.model.preprocess(data, train=False)
        embed =  self.model.encoder(data)
        lst = {k: v.squeeze(1) for k, v in self.cur_state.items()}
        lst, _ =  self.model.dynamics.observe(embed, data['action'], state=lst)
        self.cur_state = lst
        
        return lst

    def save_checkpoint(self, update_step, path, title):
        
        if not os.path.exists(path):
            os.makedirs(path)

        model_dict = {
            'model': self.model.state_dict(),
            'optimser': self.model._model_opt._opt.state_dict(),
            'update_step': update_step
        }

        torch.save(
            model_dict, 
            os.path.join(path, '{}.pth'.format(title))
        )

    def load_models(self, model_dir):
        
        if not os.path.exists(model_dir):
            print('No model found at {}'.format(model_dir)) 
            return 0
        
        checkpoint = torch.load(model_dir)
        self.model.load_state_dict(checkpoint['model'])
        self.model._model_opt._opt.load_state_dict(checkpoint['optimser'])
        return checkpoint['update_step'] + 1
    
    def reward_pred(self):
        return self.model.heads['reward']

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()

    def train(self, datasets, env, loss_logger, eval_logger):
        
        train_dataset = datasets['train']
        test_dataset = datasets['test']

        updates = []
        start_step = self.load_models(os.path.join(self.config.save_dir, 'model/model.pth'))
        losses_dict = {}
        self.set_train()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            prefetch_factor=2,
            shuffle=True)

    
        for u in tqdm(range(start_step, self.config.update_steps)):
            data = next(iter(train_dataloader))  
            _, _, met = self.model._train(data, u)
        
            # # Collect Losses
            for kk, vv in met.items():
                if kk in losses_dict.keys():
                    losses_dict[kk].append(vv)
                else:
                    losses_dict[kk] = [vv]
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
                
                
                # Save Model
                self.save_checkpoint(u, os.path.join(self.config.save_dir, 'model'), 'model')
                self.set_train()

    def visualise(self, datasets):
        self._visualise(datasets['train'], train=True)
        self._visualise(datasets['test'], train=False)


    def _visualise(self, dataset, train=False):

        #print('trans model', self.model.dynamics)
        
        train_str = 'Train' if train else 'Eval'
        eval_action_horizon = dataset.eval_action_horizon
        
        for e in range(5):
            org_gt = dataset.get_episode(e)
            plot_trajectory(
                org_gt[self.config.input_obs][6:16].transpose(0, 2 ,3, 1),
                org_gt['action'][6:16],
                title='{} Ground Truth Episode {}'.format(train_str, e),
                # rewards=data['reward'][5:15], 
                save_png = True, 
                save_path=os.path.join(self.config.save_dir, 'visualisations'))
            

            
            recon_image = []
            
            for k, v in org_gt.items():
                org_gt[k] = np.expand_dims(v, 0)
            data = self.model.preprocess(org_gt, train=False)

            
            embed = self.model.encoder(data)
            #print('embed', embed.shape)
            post, prior = self.model.dynamics.observe(embed[:, 1:], data['action'])
            post_dist = self.model.dynamics.get_dist(post)
            if not self.config.dyn_discrete:
                post_dist = post_dist._dist


            posterior_observations = self.model.heads['image'](get_feat(post, self.config))
            if self.config.decoder_output_mode == 'stochastic':
                posterior_observations = posterior_observations.mode()

            posterior_rewards = self.model.heads['reward'](get_feat(post, self.config))
            if self.config.reward_output_mode == 'stochastic':
                posterior_rewards = posterior_rewards.mode()
            


            # no_op_ts = np_to_ts(self.config.no_op, self.config.device).unsqueeze(0)
            
            # actions = torch.cat([no_op_ts, actions])


            post_process_obs = self.transform.post_transform(
                {self.config.output_obs: posterior_observations.squeeze(0).permute(0, 3, 1, 2)})[self.config.output_obs]

            posterior_rewards = ts_to_np(posterior_rewards.squeeze(0).squeeze(-1))
            plot_trajectory(
                post_process_obs[5:15].transpose(0, 2, 3, 1),
                # rewards=posterior_rewards[5:15], 
                title='{} Posterior Trajectory Episode {}'.format(train_str, e), 
                save_png = True,
                save_path=os.path.join(self.config.save_dir, 'visualisations'))

            recon_image.append(post_process_obs[5:10].transpose(0, 2 ,3, 1))

            
            for horizon in [1, 2, 3, 4, 5]:
                horizon_actions = [data['action'][:, j+1: j+horizon+1] for j in range(eval_action_horizon-horizon)]
                horizon_actions = torch.stack(horizon_actions).squeeze(1)
                
                B = horizon_actions.shape[1]
                init = {k: v[:, 0:eval_action_horizon-horizon].squeeze(0) for k, v in post.items()}

                imagin_prior = self.model.dynamics.imagine(horizon_actions, init)
                imagin_post = {k: torch.stack([post[k][:, j+1:j+1+horizon].squeeze(0) for j in range(eval_action_horizon-horizon)]) for k in post.keys()}
            
                
                imagin_reward = self.model.heads['reward'](get_feat(imagin_prior, self.config)) #.mode() # B*T*1
                if self.config.reward_output_mode == 'stochastic':
                    imagin_reward = imagin_reward.mode()

                imagin_recon = self.model.heads['image'](get_feat(imagin_prior, self.config)) # .mode() # B*T*H*W*C
                if self.config.decoder_output_mode == 'stochastic':
                    imagin_recon = imagin_recon.mode()
                
                imagin_recon =  self.transform.post_transform(
                    {self.config.output_obs: imagin_recon[:, -1, :, :, :].permute(0, 3, 1, 2)}
                )[self.config.output_obs]
                
                imagin_reward = ts_to_np(imagin_reward[:, -1, :].squeeze(-1))
                
                plot_trajectory(
                    imagin_recon[5-horizon:15-horizon].transpose(0, 2, 3, 1),
                    # rewards=imagin_reward[5-horizon:15-horizon], 
                    title='{}-Step {} Prior Trajectory Episode {}'.format(horizon, train_str, e), 
                    save_png = True,
                    save_path=os.path.join(self.config.save_dir, 'visualisations'))
                recon_image.append(imagin_recon[4+5:5+5].transpose(0, 2 ,3, 1))
            
            recon_image = np.concatenate(recon_image, axis=0)
            plot_trajectory(
                    recon_image,
                    # rewards=posterior_rewards[6:16], 
                    title='{} Recon Trajectory Episode {}'.format(train_str, e), 
                    save_png = True,
                    save_path=os.path.join(self.config.save_dir, 'visualisations'))


    def evaluate(self, dataset):
        self._visualise(dataset)
        reward_rmses = {h:[] for h in self.config.test_horizons}
        observation_mses = {h:[] for h in self.config.test_horizons}
        kls_post_to_prior = {h:[] for h in self.config.test_horizons}
        kls_prior_to_post = {h:[] for h in self.config.test_horizons}
        prior_entropies = {h:[] for h in self.config.test_horizons}
        prior_entropies = {h:[] for h in self.config.test_horizons}
        posterior_reward_rmses = []
        posterior_recon_mses = []
        posterior_entropies = []
        eval_action_horizon = dataset.eval_action_horizon

       

        for i in tqdm(range(self.config.eval_episodes), desc='evaluation on going..'):
            data = dataset.get_episode(i)
            for k, v in data.items():
                data[k] =  np.expand_dims(v, 0)
            data = self.model.preprocess(data)
            
            true_image = data['input_image'][:, 1:]
            true_reward = data['reward']

            embed = self.model.encoder(data)
            post, prior = self.model.dynamics.observe(embed[:, 1:], data['action'])
            post_dist = self.model.dynamics.get_dist(post)
            if not self.config.dyn_discrete:
                post_dist = post_dist._dist
            posterior_entropies.extend(post_dist.entropy().mean(dim=-1).flatten().detach().cpu().tolist())


            recon_post = self.model.heads['image'](get_feat(post, self.config))
            if self.config.decoder_output_mode == 'stochastic':
                recon_post = recon_post.mode()

            reward_post = self.model.heads['reward'](get_feat(post, self.config))
            if self.config.reward_output_mode == 'stochastic':
                reward_post = reward_post.mode()

            posterior_reward_rmses.extend((F.mse_loss(
                reward_post, 
                true_reward,
                reduction='none')**0.5).flatten().detach().cpu().tolist())
            
            posterior_recon_mses.extend(F.mse_loss(
                recon_post, 
                true_image,
                reduction='none').mean((2, 3, 4)).flatten().detach().cpu().tolist())

            
            # T*30
            for horizon in self.config.test_horizons:
                horizon_actions = [data['action'][:, j+1: j+horizon+1] for j in range(eval_action_horizon-horizon)]
                horizon_actions = torch.stack(horizon_actions).squeeze(1)
                
                B = horizon_actions.shape[1]
                init = {k: v[:, 0:eval_action_horizon-horizon].squeeze(0) for k, v in post.items()}

                imagin_prior = self.model.dynamics.imagine(horizon_actions, init)
                imagin_post = {k: torch.stack([post[k][:, j+1:j+1+horizon].squeeze(0) for j in range(eval_action_horizon-horizon)]) for k in post.keys()}
            
                
                imagin_reward = self.model.heads['reward'](get_feat(imagin_prior, self.config)) #.mode() # B*T*1
                if self.config.reward_output_mode == 'stochastic':
                    imagin_reward = imagin_reward.mode()

                imagin_recon = self.model.heads['image'](get_feat(imagin_prior, self.config)) # .mode() # B*T*H*W*C
                if self.config.decoder_output_mode == 'stochastic':
                    imagin_recon = imagin_recon.mode()

                true_reward = torch.stack([data['reward'][:, j+1: j+horizon+1] for j in range(eval_action_horizon-horizon)]).squeeze(1)
                true_image = torch.stack([data['input_image'][:, j+2: j+horizon+2] for j in range(eval_action_horizon-horizon)]).squeeze(1)

                reward_rmses[horizon].extend((F.mse_loss(
                        imagin_reward, 
                        true_reward,
                        reduction='none')**0.5).flatten().detach().cpu().tolist())
                
                observation_mses[horizon].extend(F.mse_loss(
                        imagin_recon , 
                        true_image,
                        reduction='none').mean((2, 3, 4)).flatten().detach().cpu().tolist())

                

                imagin_post_dist = self.model.dynamics.get_dist(imagin_post)
                imagin_prior_dist =  self.model.dynamics.get_dist(imagin_prior)

                if not self.config.dyn_discrete:
                    imagin_post_dist = imagin_post_dist._dist
                    imagin_prior_dist = imagin_prior_dist._dist

                kls_post_to_prior[horizon].extend(td.kl.kl_divergence(imagin_post_dist, imagin_prior_dist).flatten().detach().cpu().tolist())
                kls_prior_to_post[horizon].extend(td.kl.kl_divergence(imagin_prior_dist, imagin_post_dist).flatten().detach().cpu().tolist())

                prior_entropies[horizon].extend(imagin_prior_dist.entropy().flatten().detach().cpu().tolist())
        
        results = {
            'img_prior_reward_rmse': {h:reward_rmses[h] for h in  self.config.test_horizons},
            'img_prior_rgb_observation_mse': {h:observation_mses[h] for h in  self.config.test_horizons},
            'kl_divergence_between_posterior_and_img_prior': {h:kls_post_to_prior[h] for h in  self.config.test_horizons},
            'img_prior_entropy':  {h:prior_entropies[h] for h in  self.config.test_horizons}
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