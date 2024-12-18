from abc import ABC, abstractmethod

import torch


class AbstractDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.x, self.y = self.generate()

    def ground_truth(self, x):
        raise NotImplementedError

    def generate(self):
        x = torch.linspace(0, 1, self.config.num_points).view(-1, 1)  # add feature dimension
        y = self.ground_truth(x) + torch.randn(self.config.num_points, 1) * self.config.noise
        return x, y

    def __len__(self):
        return self.config.num_points

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
