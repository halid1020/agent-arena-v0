from torch.utils.data import Dataset

class BC_Dataset(Dataset):
    def __init__(self):
        self.num_samples = 0
        self.state = []
        self.action = []

    def __len__(self):
        print('num samples in bc dataset', self.num_samples)
        return self.num_samples

    def __getitem__(self, idx):
        return self.state[idx], self.action[idx]

    def append(self, state, action):
        self.state.append(state)
        self.action.append(action)
        self.num_samples += 1