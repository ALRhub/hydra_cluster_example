import os

import h5py
import torch


class OnDiskDataset(torch.utils.data.Dataset):
    def __init__(self, config, train=True):
        super().__init__()
        self.config = config
        path_to_dataset = config.path_to_dataset
        if path_to_dataset.startswith("$TMPDIR"):
            # replace env variable with actual path
            path_to_dataset = os.path.expandvars(path_to_dataset)

        if train:
            self.file_path = os.path.join(path_to_dataset, "on_disk_train.hdf5")
        else:
            self.file_path = os.path.join(path_to_dataset, "on_disk_test.hdf5")
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
