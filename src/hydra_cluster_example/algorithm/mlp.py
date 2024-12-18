import torch.nn


class MLP:
    def __init__(self, algorithm_config, device):
        self.algorithm_config = algorithm_config
        self.device = device
        if algorithm_config.activation_function == "relu":
            self.activation_function = torch.nn.ReLU()
        elif algorithm_config.activation_function == "tanh":
            self.activation_function = torch.nn.Tanh()
        else:
            raise ValueError(f"Activation function {algorithm_config.activation_function} not supported.")
        self.model = torch.nn.Sequential(
            torch.nn.Linear(algorithm_config.input_size, algorithm_config.hidden_size),
            self.activation_function,
            torch.nn.Linear(algorithm_config.hidden_size, algorithm_config.output_size)
        )
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=algorithm_config.learning_rate)

    def train_epoch(self, train_dl):
        self.model.train()
        train_loss = 0
        for x, y in train_dl:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        return train_loss / len(train_dl)

    def eval(self, test_dl):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_dl:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                test_loss += torch.nn.functional.mse_loss(pred, y).item()
        return test_loss / len(test_dl)

