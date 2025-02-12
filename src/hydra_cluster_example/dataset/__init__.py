import numpy as np

from hydra_cluster_example.dataset.line import LineDataset
from hydra_cluster_example.dataset.sine import SineDataset


def get_dataset(dataset_config):
    if dataset_config.name == "line":
        train_ds = LineDataset(dataset_config)
        test_ds = LineDataset(dataset_config)
    elif dataset_config.name == "sine":
        train_ds = SineDataset(dataset_config)
        test_ds = SineDataset(dataset_config)
    elif dataset_config.name == "on_disk":
        from hydra_cluster_example.dataset.on_disk import OnDiskDataset
        train_ds = OnDiskDataset(dataset_config, train=True)
        test_ds = OnDiskDataset(dataset_config, train=False)
    else:
        raise ValueError(f"Dataset {dataset_config.name} not supported.")
    return train_ds, test_ds

