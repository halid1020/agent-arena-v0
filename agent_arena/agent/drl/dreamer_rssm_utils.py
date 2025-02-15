import torch
import numpy as np

from agent.utilities.torch_utils import *


def get_feat(state, config):
    stoch = state['stoch']
    if config.dyn_discrete:
        shape = list(stoch.shape[:-2]) + [config.dyn_stoch * config.dyn_discrete]
        stoch = stoch.reshape(shape)
    return torch.cat([stoch, state['deter']], -1)


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
          outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.unbind(outputs, dim=0)
    return outputs


def lambda_return(
    reward, value, pcont, bootstrap, lambda_, axis):
  # Setting lambda=1 gives a discounted Monte Carlo return.
  # Setting lambda=0 gives a fixed 1-step return.
  #assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    #returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
        (inputs, pcont), bootstrap)
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class Optimizer():

    def __init__(
        self, name, parameters, lr, eps=1e-4, clip=None, wd=None, wd_pattern=r'.*',
        opt='adam', use_amp=False):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            'adam': lambda: torch.optim.Adam(parameters,
                                lr=lr,
                                eps=eps),
            'nadam': lambda: NotImplemented(
                                f'{config.opt} is not implemented'),
            'adamax': lambda: torch.optim.Adamax(parameters,
                                  lr=lr,
                                  eps=eps),
            'sgd': lambda: torch.optim.SGD(parameters,
                              lr=lr),
            'momentum': lambda: torch.optim.SGD(parameters,
                                    lr=lr,
                                    momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=False):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f'{self._name}_loss'] = loss.detach().cpu().numpy()
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self._opt)
        #loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        #self._opt.step()
        self._opt.zero_grad()
        metrics[f'{self._name}_grad_norm'] = norm.item()
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = (self._wd_pattern != r'.*')
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(['False', 'True'].index(x))
        if isinstance(default, int):
            return float(x) if ('e' in x or '.' in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(','))
        return type(default)(x)
    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x
    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {key: value.clone().unsqueeze(0) for key, value in last.items()}
            else:
                outputs = []
                for _last in last:
                  if type(_last) == type({}):
                    outputs.append({key: value.clone().unsqueeze(0) for key, value in _last.items()})
                  else:
                    outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                  outputs[key] = torch.cat([outputs[key], last[key].unsqueeze(0)], dim=0)
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat([outputs[j][key],
                                last[j][key].unsqueeze(0)], dim=0)
                    else:
                      outputs[j] = torch.cat([outputs[j], last[j].unsqueeze(0)], dim=0)
    if type(last) == type({}):
        outputs = [outputs]
    return outputs



class Every:

    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
      if not self._every:
          return False
      if self._last is None:
          self._last = step
          return True
      if step >= self._last + self._every:
          self._last += self._every
          return True
      return False


class Once:

    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:

    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
          return True
        return step < self._until


def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clip(torch.Tensor([step / duration]), 0, 1)[0]
            return (1 - mix) * initial + mix * final
        match = re.match(r'warmup\((.+),(.+)\)', string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = torch.clip(step / warmup, 0, 1)
            return scale * value
        match = re.match(r'exp\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        match = re.match(r'horizon\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clip(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)