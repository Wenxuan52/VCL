from __future__ import annotations

import math
import numpy as np
import torch

from methods import kl_gaussian, reconstruction_log_prob


@torch.no_grad()
def importance_sample_ll(model, x: torch.Tensor, ll: str, k: int = 100, sample_w: bool = False) -> tuple[float, float]:
    del sample_w
    n = x.shape[0]
    x_rep = x.repeat((k, 1))
    mu, log_sig = model.encoder(x_rep)
    z = mu + torch.exp(log_sig) * torch.randn_like(mu)
    x_hat = model.decode(z)

    log_prior = -0.5 * (math.log(2 * math.pi) + z ** 2).sum(dim=1)
    log_q = -0.5 * (math.log(2 * math.pi) + 2 * log_sig + ((z - mu) / torch.exp(log_sig)) ** 2).sum(dim=1)
    logp = reconstruction_log_prob(x_rep, x_hat, ll)
    bound = (logp - (log_q - log_prior)).reshape(k, n)

    bound_max = bound.max(dim=0).values
    centered = bound - bound_max
    ll_i = torch.log(torch.exp(centered).mean(dim=0).clamp_min(1e-9)) + bound_max
    mean = ll_i.mean().item()
    var = ((ll_i - ll_i.mean()) ** 2).mean().item()
    return mean, var
