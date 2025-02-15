
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torch.distributions as td

OPTIMISER_CLASSES = {
    'adam': optim.Adam
}

ACTIVATIONS = {
    'relu': nn.ReLU,
    'elu': nn.ELU
}

np_to_ts = lambda x, y: torch.tensor(x).float().to(y) # y is device
ts_to_np = lambda x: x.detach().cpu().numpy()

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


class RequiresGrad:

    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)


class SampleDist:

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(td.one_hot_categorical.OneHotCategorical):

    def __init__(self, logits=None, probs=None):
        super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
          raise ValueError('need to check')
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
          probs = probs[None]
        sample += probs - probs.detach()
        return sample

class ContDist:

    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        return self._dist.mean

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:

  def __init__(self, dist=None):
      super().__init__()
      self._dist = dist
      self.mean = dist.mean

  def __getattr__(self, name):
      return getattr(self._dist, name)

  def entropy(self):
      return self._dist.entropy()

  def mode(self):
      _mode = torch.round(self._dist.mean)
      return _mode.detach() +self._dist.mean - self._dist.mean.detach()

  def sample(self, sample_shape=()):
      return self._dist.rsample(sample_shape)

  def log_prob(self, x):
      _logits = self._dist.base_dist.logits
      log_probs0 = -F.softplus(_logits)
      log_probs1 = -F.softplus(-_logits)

      return log_probs0 * (1-x) + log_probs1 * x


class UnnormalizedHuber(td.normal.Normal):

  def __init__(self, loc, scale, threshold=1, **kwargs):
      super().__init__(loc, scale, **kwargs)
      self._threshold = threshold

  def log_prob(self, event):
      return -(torch.sqrt(
          (event - self.mean) ** 2 + self._threshold ** 2) - self._threshold)

  def mode(self):
      return self.mean


class SafeTruncatedNormal(td.normal.Normal):

  def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
      super().__init__(loc, scale)
      self._low = low
      self._high = high
      self._clip = clip
      self._mult = mult

  def sample(self, sample_shape):
      event = super().sample(sample_shape)
      if self._clip:
          clipped = torch.clip(event, self._low + self._clip,
              self._high - self._clip)
          event = event - event.detach() + clipped.detach()
      if self._mult:
          event *= self._mult
      return event


class TanhBijector(td.Transform):

    def __init__(self, validate_args=False, name='tanh'):
        super().__init__()

    def _forward(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997), y)
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))

    
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