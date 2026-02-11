from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import torch


def kl_gaussian(mu_p: torch.Tensor, log_sig_p: torch.Tensor, mu_q: float | torch.Tensor = 0.0, log_sig_q: float | torch.Tensor = 0.0) -> torch.Tensor:
    if not torch.is_tensor(mu_q):
        mu_q = torch.tensor(mu_q, device=mu_p.device, dtype=mu_p.dtype)
    if not torch.is_tensor(log_sig_q):
        log_sig_q = torch.tensor(log_sig_q, device=mu_p.device, dtype=mu_p.dtype)
    precision_q = torch.exp(-2.0 * log_sig_q)
    kl = 0.5 * (mu_p - mu_q) ** 2 * precision_q - 0.5
    kl = kl + log_sig_q - log_sig_p + 0.5 * torch.exp(2.0 * log_sig_p - 2.0 * log_sig_q)
    return kl.flatten(1).sum(dim=1)


def reconstruction_log_prob(x: torch.Tensor, x_hat: torch.Tensor, ll: str) -> torch.Tensor:
    if ll == "bernoulli":
        p = torch.clamp(x_hat, 1e-9, 1.0 - 1e-9)
        logp = x * torch.log(p) + (1 - x) * torch.log(1 - p)
    elif ll == "l2":
        logp = -((x - x_hat) ** 2)
    elif ll == "l1":
        logp = -(x - x_hat).abs()
    else:
        raise ValueError(ll)
    return logp.flatten(1).sum(dim=1)


def elbo(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, log_sig: torch.Tensor, ll: str) -> torch.Tensor:
    return reconstruction_log_prob(x, x_hat, ll) - kl_gaussian(mu, log_sig)


def make_fisher_diag(loss: torch.Tensor, params: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
    grads = torch.autograd.grad(loss, list(params), retain_graph=False, create_graph=False)
    out = []
    for g in grads:
        if g is None:
            out.append(torch.tensor(0.0, device=loss.device))
        else:
            out.append(g.detach() ** 2)
    return out


@dataclass
class ContinualRegularizer:
    kind: str
    lbd: float
    old_params: list[torch.Tensor] | None = None
    fisher: list[torch.Tensor] | None = None
    omega: list[torch.Tensor] | None = None
    prior_mean: list[torch.Tensor] | None = None
    prior_log_sigma: float = -2.3

    def penalty(self, shared_params: list[torch.nn.Parameter]) -> torch.Tensor:
        device = shared_params[0].device
        z = torch.tensor(0.0, device=device)
        if self.kind in {"noreg", ""} or self.old_params is None:
            return z

        if self.kind == "ewc":
            assert self.fisher is not None
            return 0.5 * self.lbd * sum((f * (p - p0) ** 2).sum() for p, p0, f in zip(shared_params, self.old_params, self.fisher))
        if self.kind == "laplace":
            assert self.fisher is not None
            return 0.5 * self.lbd * sum((f * (p - p0) ** 2).sum() for p, p0, f in zip(shared_params, self.old_params, self.fisher))
        if self.kind == "si":
            assert self.omega is not None
            return 0.5 * self.lbd * sum((om * (p - p0) ** 2).sum() for p, p0, om in zip(shared_params, self.old_params, self.omega))
        if self.kind == "onlinevi":
            assert self.prior_mean is not None
            sig2 = torch.exp(torch.tensor(2 * self.prior_log_sigma, device=device))
            return 0.5 * sum(((p - pm) ** 2 / sig2).sum() for p, pm in zip(shared_params, self.prior_mean))
        raise ValueError(self.kind)

    def update_after_task(
        self,
        shared_params: list[torch.nn.Parameter],
        fisher_estimate: list[torch.Tensor] | None = None,
        si_new_omega: list[torch.Tensor] | None = None,
    ) -> None:
        self.old_params = [p.detach().clone() for p in shared_params]

        if self.kind == "ewc":
            self.fisher = [f.detach().clone() for f in (fisher_estimate or [])]
        elif self.kind == "laplace":
            if self.fisher is None:
                self.fisher = [torch.zeros_like(p) for p in shared_params]
            if fisher_estimate is not None:
                self.fisher = [f_old + f_new.detach() for f_old, f_new in zip(self.fisher, fisher_estimate)]
        elif self.kind == "si":
            self.omega = si_new_omega
        elif self.kind == "onlinevi":
            self.prior_mean = [p.detach().clone() for p in shared_params]
