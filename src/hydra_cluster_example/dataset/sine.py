import torch

from hydra_cluster_example.dataset.abstract_in_memory_dataset import AbstractInMemoryDataset


class SineDataset(AbstractInMemoryDataset):
    def __init__(self, config):
        super().__init__(config)

    def ground_truth(self, x):
        return torch.sin(2 * torch.pi * x)
