from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
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


class BayesLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_logstd: float,
        prior_mu: float = 0.0,
        prior_logstd: float = 0.0,
    ) -> None:
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_logstd = nn.Parameter(torch.full((out_features, in_features), init_logstd))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_logstd = nn.Parameter(torch.full((out_features,), init_logstd))

        self.register_buffer("prior_weight_mu", torch.full((out_features, in_features), prior_mu))
        self.register_buffer("prior_weight_logstd", torch.full((out_features, in_features), prior_logstd))
        self.register_buffer("prior_bias_mu", torch.full((out_features,), prior_mu))
        self.register_buffer("prior_bias_logstd", torch.full((out_features,), prior_logstd))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5.0))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1.0 / math.sqrt(max(1, fan_in))
        nn.init.uniform_(self.bias_mu, -bound, bound)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            weight = self.weight_mu + torch.exp(self.weight_logstd) * weight_eps
            bias = self.bias_mu + torch.exp(self.bias_logstd) * bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    @staticmethod
    def _diag_gaussian_kl(mu: torch.Tensor, logstd: torch.Tensor, prior_mu: torch.Tensor, prior_logstd: torch.Tensor) -> torch.Tensor:
        sigma2 = torch.exp(2.0 * logstd)
        prior_sigma2 = torch.exp(2.0 * prior_logstd)
        return (prior_logstd - logstd + (sigma2 + (mu - prior_mu) ** 2) / (2.0 * prior_sigma2) - 0.5).sum()

    def kl(self) -> torch.Tensor:
        w_kl = self._diag_gaussian_kl(self.weight_mu, self.weight_logstd, self.prior_weight_mu, self.prior_weight_logstd)
        b_kl = self._diag_gaussian_kl(self.bias_mu, self.bias_logstd, self.prior_bias_mu, self.prior_bias_logstd)
        return w_kl + b_kl

    def reset_posterior_from_prior(self, init_logstd: float) -> None:
        with torch.no_grad():
            self.weight_mu.copy_(self.prior_weight_mu)
            self.bias_mu.copy_(self.prior_bias_mu)
            self.weight_logstd.fill_(init_logstd)
            self.bias_logstd.fill_(init_logstd)

    def reset_logstd(self, init_logstd: float) -> None:
        with torch.no_grad():
            self.weight_logstd.fill_(init_logstd)
            self.bias_logstd.fill_(init_logstd)

    def update_prior_from_posterior(self) -> None:
        with torch.no_grad():
            self.prior_weight_mu.copy_(self.weight_mu.detach())
            self.prior_weight_logstd.copy_(self.weight_logstd.detach())
            self.prior_bias_mu.copy_(self.bias_mu.detach())
            self.prior_bias_logstd.copy_(self.bias_logstd.detach())


class DecoderHead(nn.Module):
    def __init__(self, dim_z: int, dim_h: int, n_layers: int, bayesian: bool = False, init_logstd: float = -6.0) -> None:
        super().__init__()
        self.net = MLP([dim_z] + [dim_h] * n_layers)
        self.bayesian = bayesian
        if bayesian:
            sizes = [dim_z] + [dim_h] * n_layers
            self.bayes_layers = nn.ModuleList(
                [BayesLinear(sizes[i], sizes[i + 1], init_logstd=init_logstd, prior_mu=0.0, prior_logstd=0.0) for i in range(len(sizes) - 1)]
            )
        else:
            self.bayes_layers = nn.ModuleList()

    def forward(self, z: torch.Tensor, sample_w: bool = True) -> torch.Tensor:
        if not self.bayesian:
            return self.net(z)
        h = z
        for i, layer in enumerate(self.bayes_layers):
            h = layer(h, sample=sample_w)
            if i < len(self.bayes_layers) - 1:
                h = F.relu(h)
        return h

    def kl(self) -> torch.Tensor:
        if not self.bayesian:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return sum(layer.kl() for layer in self.bayes_layers)

    def reset_posterior_from_prior(self, init_logstd: float) -> None:
        for layer in self.bayes_layers:
            layer.reset_logstd(init_logstd)

    def update_prior_from_posterior(self) -> None:
        for layer in self.bayes_layers:
            layer.update_prior_from_posterior()


class DecoderShared(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_h: int,
        n_layers: int,
        last_activation: str = "sigmoid",
        bayesian: bool = False,
        init_logstd: float = -6.0,
    ) -> None:
        super().__init__()
        self.net = MLP([dim_h] * n_layers + [dim_x], final_activation=last_activation)
        self.bayesian = bayesian
        self.last_activation = last_activation
        if bayesian:
            sizes = [dim_h] * n_layers + [dim_x]
            self.bayes_layers = nn.ModuleList(
                [BayesLinear(sizes[i], sizes[i + 1], init_logstd=init_logstd, prior_mu=0.0, prior_logstd=0.0) for i in range(len(sizes) - 1)]
            )
        else:
            self.bayes_layers = nn.ModuleList()

    def forward(self, h: torch.Tensor, sample_w: bool = True) -> torch.Tensor:
        if not self.bayesian:
            return self.net(h)
        out = h
        for i, layer in enumerate(self.bayes_layers):
            out = layer(out, sample=sample_w)
            if i < len(self.bayes_layers) - 1:
                out = F.relu(out)
            elif self.last_activation == "sigmoid":
                out = torch.sigmoid(out)
        return out

    def kl(self) -> torch.Tensor:
        if not self.bayesian:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return sum(layer.kl() for layer in self.bayes_layers)

    def reset_posterior_from_prior(self, init_logstd: float) -> None:
        for layer in self.bayes_layers:
            layer.reset_posterior_from_prior(init_logstd)

    def update_prior_from_posterior(self) -> None:
        for layer in self.bayes_layers:
            layer.update_prior_from_posterior()


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

    def decode(self, z: torch.Tensor, sample_w: bool = False) -> torch.Tensor:
        return self.decoder_shared(self.decoder_head(z, sample_w=sample_w), sample_w=sample_w)

    def forward(self, x: torch.Tensor, sample: bool = True, sample_w: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_sig = self.encoder(x)
        if sample:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(log_sig) * eps
        else:
            z = mu
        x_hat = self.decode(z, sample_w=sample_w)
        return x_hat, mu, log_sig

    def sample(self, n: int, dim_z: int, device: torch.device, sample_w: bool = False) -> torch.Tensor:
        z = torch.randn(n, dim_z, device=device)
        return self.decode(z, sample_w=sample_w)
