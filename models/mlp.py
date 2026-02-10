import torch
import torch.nn as nn

from models.variational_layers import VarLinear


class VarMLP(nn.Module):
    def __init__(self, dims, activation="relu", bias=True):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must include at least [in_dim, out_dim]")

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(VarLinear(dims[i], dims[i + 1], bias=bias))

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x, sample: bool = True):
        for i, layer in enumerate(self.layers):
            x = layer(x, sample=sample)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

    def kl(self):
        total = None
        for layer in self.layers:
            kl_i = layer.kl()
            total = kl_i if total is None else total + kl_i
        if total is None:
            return torch.tensor(0.0)
        return total


def make_var_mlp(dims, activation="relu", bias=True):
    return VarMLP(dims=dims, activation=activation, bias=bias)


if __name__ == "__main__":
    mlp = make_var_mlp([3, 5, 2])
    x = torch.randn(8, 3)
    y = mlp(x, sample=True)
    print(f"shape ok: {tuple(y.shape)} == {(8, 2)}")
    print(f"kl={mlp.kl().item():.8f}")
