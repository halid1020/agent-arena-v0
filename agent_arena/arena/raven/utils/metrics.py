import torch


class MeanMetrics:
    def __init__(self):
        self.init_values()

    def init_values(self):
        self.values = []

    def reset(self):
        self.init_values()

    def result(self):
        return torch.mean(self.values)

    def __call__(self, x):
        self.values.append(x)

