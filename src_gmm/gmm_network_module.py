import torch
import torch.nn as nn

from omegaconf import DictConfig


class RMLP(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config
        self.activations = self._store_activations()
        assert not (self.config.quad and self.config.sigmoid)

        layers = [
            nn.Linear(self.config.input_dim, self.config.hidden_dim, bias=self.config.bias),
            self._get_activation(self.config.activation)
        ]

        for _ in range(self.config.num_layers - 2):
            layers.append(nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=self.config.bias))

            if self.config.batch_norm:
                layers.append(nn.BatchNorm1d(self.config.hidden_dim))
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))

            layers.append(self._get_activation(self.config.activation))

        layers.append(nn.Linear(self.config.hidden_dim, self.config.output_dim, bias=self.config.bias))

        if self.config.full_activ:
            layers.append(self._get_activation(self.config.final_activ))

        self.network = nn.Sequential(*layers)


    def forward(self, inp):
        out = self.network(inp)

        if self.config.res:
            out = out + inp

        if self.config.quad:
            return out ** 2
        elif self.config.sigmoid:
            return torch.sigmoid(3.0 * out / 20.0) * 20.0
        else:
            return out
        
    
    def _store_activations(self):
        return {
            "celu": nn.CELU,
            "hardshrink": nn.Hardshrink,
            "prelu": nn.PReLU,
            "relu": lambda: nn.ReLU(inplace=True),
            "rrelu": lambda: nn.RReLU(0.5, 0.8),            
            "selu": nn.SELU,
            "sigmoid": nn.Sigmoid,            
            "softsign": nn.Softsign,
            "tanh": nn.Tanh,
            "tanshrink": nn.Tanhshrink
        }
    
    
    def _get_activation(self, name):
        try:
            return self.activations[name.lower()]()
        except KeyError:
            raise ValueError(f"Unknown activation function: {name}")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.create({
        "input_dim": 2,
        "output_dim": 2,
        "hidden_dim": 8,
        "num_layers": 4,
        "activation": "Prelu",
        "final_activ": "Prelu",
        "full_activ": True,
        "bias": True,
        "dropout": 0.04,
        "batch_norm": False,
        "res": True,
        "quad": False,
        "sigmoid": False,
    })

    transport_net = RMLP(config)
    print(transport_net)