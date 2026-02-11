from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ClassifierMLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 1000, n_layers: int = 3, num_classes: int = 10):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(data_name: str, device: torch.device) -> nn.Module:
    path = Path("original_code_pytorch/dgm/classifier/save") / f"{data_name}_weights.pt"
    if not path.exists():
        raise FileNotFoundError(f"Classifier weights not found: {path}")

    model = ClassifierMLP()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded classifier from {path}")
    return model
