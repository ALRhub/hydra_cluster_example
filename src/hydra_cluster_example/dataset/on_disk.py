import os

import h5py
import torch


class OnDiskDataset(torch.utils.data.Dataset):
    def __init__(self, config, train=True):
        super().__init__()
        self.config = config
        print("In OnDiskDataset")
        print("Path to dataset: ", config.path_to_dataset)
        print("joined path: ", os.path.join(config.path_to_dataset, "on_disk_train.hdf5"))
        if train:
            self.file_path = os.path.join(config.path_to_dataset, "on_disk_train.hdf5")
        else:
            self.file_path = os.path.join(config.path_to_dataset, "on_disk_test.hdf5")
        with h5py.File(self.file_path, "r") as hdf:
            self.length = len(hdf["x"])

    def __len__(self):
        return self.length

    def ground_truth(self, x):
        # for visualization only
        return 0.1 * torch.sin(2 * torch.pi * x) + x ** 2

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as hdf:
            x = torch.tensor(hdf["x"][idx])
            y = torch.tensor(hdf["y"][idx])
        return x, y
