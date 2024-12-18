import hydra
import torch.utils.data
from omegaconf import OmegaConf

from hydra_cluster_example.algorithm import get_algorithm
from hydra_cluster_example.dataset import get_dataset

EXP_NAME = "exp_1.yaml"  # your experiment config you want to run


@hydra.main(version_base=None, config_path="configs", config_name=EXP_NAME)
def main(config) -> None:
    print(OmegaConf.to_yaml(config))
    if config.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("Training on device {}".format(device))
    train_ds, test_ds = get_dataset(config.dataset)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.dataset.batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.dataset.batch_size, shuffle=False)
    algorithm = get_algorithm(config.algorithm, device)
    for epoch in range(config.epochs):
        train_loss = algorithm.train_epoch(train_dl)
        test_loss = algorithm.eval(test_dl)
        print(f"Epoch {epoch}: Train Loss: {train_loss}, Test Loss: {test_loss}")
        if epoch % 100 == 0 and config.visualize:
            # visualize
            visualize(algorithm, train_ds, device)


def visualize(algorithm, train_ds, device):
    import matplotlib.pyplot as plt
    import numpy as np
    x = torch.linspace(0, 1, 100).view(-1, 1)
    y = train_ds.ground_truth(x)
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        pred = algorithm.model(x)
    x = x[:, 0].cpu().numpy()
    y = y[:, 0].cpu().numpy()
    pred = pred[:, 0].cpu().numpy()
    plt.plot(x, y, label="Ground Truth")
    plt.plot(x, pred, label="Prediction")
    plt.scatter(train_ds.x, train_ds.y, label="Noisy Train Data", color="red", marker="x", s=10)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
