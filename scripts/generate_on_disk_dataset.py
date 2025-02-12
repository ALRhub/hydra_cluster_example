import os

import h5py
import torch

def generate_dataset(output_path):
    data_points = 1000
    noise = 0.01
    x = torch.linspace(0, 1, data_points).view(-1, 1)
    y = 0.1 * torch.sin(2 * torch.pi * x) + x ** 2 + noise * torch.randn(data_points, 1)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)


if __name__ == "__main__":
    torch.manual_seed(42)
    # get path of this file
    current_path = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.join(current_path, "../outputs/datasets/on_disk_train.hdf5")
    generate_dataset(output_path)
    output_path = os.path.join(current_path, "../outputs/datasets/on_disk_test.hdf5")
    generate_dataset(output_path)
