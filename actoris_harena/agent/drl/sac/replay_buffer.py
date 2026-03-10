import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, max_action_dim, device):
        self.capacity = int(capacity)
        self.device = device

        self.ptr = 0
        self.size = 0
        #print('obs shape', obs_shape)

        # Support any observation shape (e.g., images, vectors, etc.)
        self.observation = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, max_action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_observation = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

    def add(self, observation, action, reward, next_observation, done):
        self.observation[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observation[self.ptr] = next_observation
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            observation=torch.tensor(self.observation[idxs]).to(self.device),
            action=torch.tensor(self.actions[idxs]).to(self.device),
            reward=torch.tensor(self.rewards[idxs]).unsqueeze(-1).to(self.device),
            next_observation=torch.tensor(self.next_observation[idxs]).to(self.device),
            done=torch.tensor(self.dones[idxs]).unsqueeze(-1).to(self.device),
        )
        return batch