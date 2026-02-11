from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, sizes: list[int], final_activation: str = "linear") -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
            elif final_activation == "sigmoid":
                layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, dim_x: int, dim_h: int, dim_z: int, n_layers: int) -> None:
        super().__init__()
        self.backbone = MLP([dim_x] + [dim_h] * n_layers + [2 * dim_z])
        self.dim_z = dim_z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(x)
        return out[:, : self.dim_z], out[:, self.dim_z :]


class DecoderHead(nn.Module):
    def __init__(self, dim_z: int, dim_h: int, n_layers: int) -> None:
        super().__init__()
        self.net = MLP([dim_z] + [dim_h] * n_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DecoderShared(nn.Module):
    def __init__(self, dim_x: int, dim_h: int, n_layers: int, last_activation: str = "sigmoid") -> None:
        super().__init__()
        self.net = MLP([dim_h] * n_layers + [dim_x], final_activation=last_activation)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


@dataclass
class TaskModel(nn.Module):
    encoder: Encoder
    decoder_head: DecoderHead
    decoder_shared: DecoderShared

    def __init__(self, encoder: Encoder, decoder_head: DecoderHead, decoder_shared: DecoderShared) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder_head = decoder_head
        self.decoder_shared = decoder_shared

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_shared(self.decoder_head(z))

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_sig = self.encoder(x)
        if sample:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(log_sig) * eps
        else:
            z = mu
        x_hat = self.decode(z)
        return x_hat, mu, log_sig

    def sample(self, n: int, dim_z: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, dim_z, device=device)
        return self.decode(z)
