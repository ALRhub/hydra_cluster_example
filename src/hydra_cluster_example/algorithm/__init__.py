from hydra_cluster_example.algorithm.mlp import MLP


def get_algorithm(config, device):
    if config.name == "mlp":
        return MLP(config, device)
    else:
        raise ValueError(f"Algorithm {config.name} not supported.")
