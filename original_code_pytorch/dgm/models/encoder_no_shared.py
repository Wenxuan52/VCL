from __future__ import annotations

import torch
from torch import nn

from alg.helper_functions import sample_gaussian


class EncoderNoShared(nn.Module):
    """Task-specific encoder MLP: 784 -> 500 -> 500 -> 500 -> 2*50 by default."""

    def __init__(self, dimX: int = 784, dimH: int = 500, dimZ: int = 50, n_layers: int = 3, name: str | None = None):
        super().__init__()
        self.dimX = dimX
        self.dimH = dimH
        self.dimZ = dimZ
        self.n_layers = n_layers
        self.name = name

        hidden = []
        in_dim = dimX
        for _ in range(n_layers):
            hidden.append(nn.Linear(in_dim, dimH))
            hidden.append(nn.ReLU())
            in_dim = dimH
        self.hidden = nn.Sequential(*hidden)
        self.out = nn.Linear(dimH, dimZ * 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.hidden(x)
        out = self.out(h)
        mu_qz, log_sig_qz = out.chunk(2, dim=1)
        return mu_qz, log_sig_qz

    @staticmethod
    def reparameterize(mu_qz: torch.Tensor, log_sig_qz: torch.Tensor) -> torch.Tensor:
        return sample_gaussian(mu_qz, log_sig_qz)
