import argparse
import os

import hydra
import numpy as np
import torch.utils.data
from omegaconf import OmegaConf

from hydra_cluster_example.algorithm import get_algorithm
from hydra_cluster_example.dataset import get_dataset


# use --config-name <config_name> to specify the config file as an argument
@hydra.main(version_base=None, config_path="configs")
def main(config) -> None:
    # set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
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
    if config.wandb:
        import wandb
        # save config as dict for wandb
        # group your runs in the wandb dashboard as ["Group", "Job Type"]
        wandb.init(project="hydra-cluster-example", config=OmegaConf.to_container(config, resolve=True),
                   name=f"{config.name}_seed_{config.seed}",
                   group=config.group_name,
                   job_type=config.name,
                   )
    for epoch in range(config.epochs):
        train_loss = algorithm.train_epoch(train_dl)
        test_loss = algorithm.eval(test_dl)
        print(f"Epoch {epoch}: Train Loss: {train_loss}, Test Loss: {test_loss}")
        if config.wandb:
            wandb.log({"train_loss": train_loss, "test_loss": test_loss, "epoch": epoch}, step=epoch)
        if epoch % 100 == 0 and config.visualize:
            # visualize
            visualize(algorithm, train_ds, device, show=not config.wandb)
            if config.wandb:
                wandb.log({"prediction": wandb.Image("outputs/prediction.png")}, step=epoch)
    if config.wandb:
        wandb.finish()


def visualize(algorithm, train_ds, device, show=True):
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
    if show:
        plt.show()
    else:
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/prediction.png")
        plt.close()




if __name__ == '__main__':
    main()
