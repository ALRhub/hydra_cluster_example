import torch

from hydra_cluster_example.dataset.abstract_dataset import AbstractDataset


class LineDataset(AbstractDataset):
    def __init__(self, config):
        super().__init__(config)

    def ground_truth(self, x):
        return x * self.config.slope + self.config.intercept
