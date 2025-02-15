# modified upon https://github.com/Kaixhin/PlaNet/blob/master/memory.py

import numpy as np
import torch
import joblib


class ExperienceReplay():
  def __init__(self, 
               size, symbolic_env, 
               observation_size, action_size, device,
               **kwargs):
    self.device = device
    self.symbolic_env = symbolic_env
    self.size = size
    self.observations = np.empty((size, observation_size) if symbolic_env else (size, *observation_size), dtype=np.float32)
    self.actions = np.empty((size, action_size), dtype=np.float32)
    self.rewards = np.empty((size, ), dtype=np.float32) 
    self.nonterminals = np.empty((size, 1), dtype=np.float32)
    self.idx = 0
    self.full = False  # Tracks if memory has been filled/all slots are valid
    self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total

  def append(self, observation, action, reward, done):
    if self.symbolic_env:
      self.observations[self.idx] = observation.numpy()
    else:
      self.observations[self.idx] = observation
    self.actions[self.idx] = action
    self.rewards[self.idx] = reward
    self.nonterminals[self.idx] = not done
    #print('self.nonterminals[self.idx]', self.nonterminals[self.idx])
    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0
    self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

  # Returns an index for a valid single sequence chunk uniformly sampled from the memory
  def _sample_idx(self, L):
    #print('idx', self.idx)
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - L)
      idxs = np.arange(idx, idx + L) % self.size
      valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
    return idxs

  def _retrieve_batch(self, idxs, n, L):
    vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
    #observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
    observations = self.observations[vec_idxs].astype(np.float32)

    ## TODO: rgb is too specific.
    return {
      'rgb':  observations.reshape(L, n, *observations.shape[1:]),
      'action': self.actions[vec_idxs].reshape(L, n, -1),
      'reward':  self.rewards[vec_idxs].reshape(L, n),
      'terminal': 1 - self.nonterminals[vec_idxs].reshape(L, n)
    }

  # Returns a batch of sequence chunks uniformly sampled from the memory
  def sample(self, n, L):
    return  self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(n)]), n, L)
    # return [torch.as_tensor(item).to(device=self.device) for item in batch]

  
  def save(self, file_path):
    print('Saving experience ...')
    #file_path = file_path + '/data.pkl'
    with open(file_path, 'wb') as f:
        data = {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'nonterminals': self.nonterminals,
            'idx': self.idx,
            'full': self.full,
            'steps': self.steps,
            'episodes': self.episodes
        }
        joblib.dump(data, f)

    print('Finished saving experience')

  def load(self, file_path):
    #file_path = file_path + '/data.pkl'
    print('Loading experience ...')
    with open(file_path, 'rb') as f:
        data = joblib.load(f)
        self.observations = data['observations']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.nonterminals = data['nonterminals']
        self.idx = data['idx']
        self.full = data['full']
        self.steps = data['steps']
        self.episodes = data['episodes']
    print('Finished loading experience')
  