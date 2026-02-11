from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _init_linear_mu(out_features: int, in_features: int) -> torch.Tensor:
    # Glorot-uniform to mirror TF init intent
    limit = math.sqrt(6.0 / (in_features + out_features))
    return torch.empty(out_features, in_features).uniform_(-limit, limit)


class BayesianLinear(nn.Module):
    """Mean-field Gaussian Bayesian linear layer with (mu, log_sig) for W and b."""

    def __init__(self, in_features: int, out_features: int, log_sig_init: float = -6.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_W = nn.Parameter(_init_linear_mu(out_features, in_features))
        self.log_sig_W = nn.Parameter(torch.full((out_features, in_features), float(log_sig_init)))
        self.mu_b = nn.Parameter(torch.zeros(out_features))
        self.log_sig_b = nn.Parameter(torch.full((out_features,), float(log_sig_init)))

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            w = self.mu_W + torch.exp(self.log_sig_W) * torch.randn_like(self.mu_W)
            b = self.mu_b + torch.exp(self.log_sig_b) * torch.randn_like(self.mu_b)
        else:
            w = self.mu_W
            b = self.mu_b
        return F.linear(x, w, b)


class GeneratorHeadBayesian(nn.Module):
    """Task-specific head: dimZ -> dimH -> dimH with ReLU activations."""

    def __init__(self, dimZ: int = 50, dimH: int = 500):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BayesianLinear(dimZ, dimH),
                BayesianLinear(dimH, dimH),
            ]
        )

    def forward(self, z: torch.Tensor, sample_W: bool = True) -> torch.Tensor:
        x = z
        for layer in self.layers:
            x = F.relu(layer(x, sample=sample_W))
        return x


class GeneratorSharedBayesian(nn.Module):
    """Shared trunk: dimH -> dimH -> dimX with ReLU then sigmoid output."""

    def __init__(self, dimH: int = 500, dimX: int = 784):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BayesianLinear(dimH, dimH),
                BayesianLinear(dimH, dimX),
            ]
        )

    def forward(self, h: torch.Tensor, sample_W: bool = True) -> torch.Tensor:
        h = F.relu(self.layers[0](h, sample=sample_W))
        logits = self.layers[1](h, sample=sample_W)
        return torch.sigmoid(logits)


class BayesianDecoder(nn.Module):
    """Decoder with shared Bayesian trunk and task-specific Bayesian heads."""

    def __init__(self, dimZ: int = 50, dimH: int = 500, dimX: int = 784):
        super().__init__()
        self.dimZ = dimZ
        self.dimH = dimH
        self.dimX = dimX
        self.shared = GeneratorSharedBayesian(dimH=dimH, dimX=dimX)
        self.heads = nn.ModuleDict()

    def get_head(self, task_id: int) -> GeneratorHeadBayesian:
        key = str(task_id)
        if key not in self.heads:
            head = GeneratorHeadBayesian(dimZ=self.dimZ, dimH=self.dimH)
            # Important: heads can be created *after* decoder.to(device).
            # Ensure newly created head follows decoder parameter device/dtype.
            ref_param = next(self.shared.parameters())
            head = head.to(device=ref_param.device, dtype=ref_param.dtype)
            self.heads[key] = head
        return self.heads[key]

    def forward(self, z: torch.Tensor, task_id: int, sample_W: bool = True) -> torch.Tensor:
        head = self.get_head(task_id)
        h = head(z, sample_W=sample_W)
        mu_x = self.shared(h, sample_W=sample_W)
        return mu_x
