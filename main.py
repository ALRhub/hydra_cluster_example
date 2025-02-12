import os
import hydra
import numpy as np
import torch.utils.data
from omegaconf import OmegaConf
import wandb
import matplotlib.pyplot as plt

from hydra_cluster_example.algorithm import get_algorithm
from hydra_cluster_example.dataset import get_dataset


# use --config-name <config_name> to specify the config file as an argument
@hydra.main(version_base=None, config_path="configs")
def main(config) -> None:
    # initialize data, algorithm, and device. Also sets seed
    algorithm, device, test_dl, train_dl, train_ds = initialize(config)

    # training loop
    for epoch in range(config.epochs):
        train_loss = algorithm.train_epoch(train_dl)
        # you may eval only every x epochs for bigger projects
        test_loss = algorithm.eval(test_dl)
        print(f"Epoch {epoch}: Train Loss: {train_loss}, Test Loss: {test_loss}")
        if config.wandb:
            # log current loss
            wandb.log({"train_loss": train_loss, "test_loss": test_loss, "epoch": epoch}, step=epoch)
        if epoch % 100 == 0 and config.visualize:
            # visualize
            vis_path = visualize(algorithm, train_ds, device, epoch, show=not config.wandb)
            if config.wandb:
                # you can either log the visualization as an image, or create a plotly plot and log that.
                wandb.log({"prediction": wandb.Image(vis_path)}, step=epoch)
    if config.wandb:
        # important to finish the wandb run, especially for multiruns. Otherwise a new run will not start.
        wandb.finish()


def initialize(config):
    print("Using the following Config:")
    print(OmegaConf.to_yaml(config))
    # set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    # get device
    if config.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("Training on device {}".format(device))
    # loading data
    train_ds, test_ds = get_dataset(config.dataset)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.dataset.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    # get algorithm
    algorithm = get_algorithm(config.algorithm, device)
    if config.wandb:
        # save config as dict for wandb
        # group your runs in the wandb dashboard as ["Group", "Job Type"]
        wandb.init(project="hydra-cluster-example", config=OmegaConf.to_container(config, resolve=True),
                   name=f"{config.name}_seed_{config.seed}",
                   group=config.group_name,
                   job_type=config.name,
                   )
        # you can do more fancy stuff with wandb init, to set the names, tags, and more..
    return algorithm, device, test_dl, train_dl, train_ds


def visualize(algorithm, train_ds, device, epoch, show=True):
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
    # if you want to plot the train data as well, uncomment the next line
    # plt.scatter(train_ds.x, train_ds.y, label="Noisy Train Data", color="red", marker="x", s=10)
    plt.legend()
    if show:
        plt.show()
        return None
    else:
        # hydra save dir
        recording_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        save_path = os.path.join(recording_dir, f"prediction_epoch_{epoch}.png")
        plt.savefig(save_path)
        plt.close()
        return save_path




if __name__ == '__main__':
    main()
