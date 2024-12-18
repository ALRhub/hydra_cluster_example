import torch

from hydra_cluster_example.dataset.abstract_dataset import AbstractDataset


class SineDataset(AbstractDataset):
    def __init__(self, config):
        super().__init__(config)

    def ground_truth(self, x):
        return torch.sin(2 * torch.pi * x)

