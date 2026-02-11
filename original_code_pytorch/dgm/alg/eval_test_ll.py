from __future__ import annotations

import math
from typing import Tuple

import torch

from .helper_functions import kl_diag_gaussians, log_bernoulli_prob, sample_gaussian


@torch.no_grad()
def eval_elbo_on_dataset(
    encoder,
    decoder,
    dataloader,
    task_id: int,
    device: torch.device,
    K: int = 50,
    sample_W: bool = False,
) -> float:
    encoder.eval()
    decoder.eval()

    total = 0.0
    n_total = 0
    for (x,) in dataloader:
        x = x.to(device)
        mu_qz, log_sig_qz = encoder(x)
        kl_z = kl_diag_gaussians(mu_qz, log_sig_qz, 0.0, 0.0, reduce_dims=True)

        recon = torch.zeros_like(kl_z)
        for _ in range(K):
            z = sample_gaussian(mu_qz, log_sig_qz)
            mu_x = decoder(z, task_id=task_id, sample_W=sample_W)
            recon = recon + log_bernoulli_prob(x, mu_x) / float(K)

        elbo = recon - kl_z
        total += elbo.sum().item()
        n_total += x.shape[0]

    return total / max(1, n_total)


def _log_gaussian_diag(x: torch.Tensor, mu: torch.Tensor, log_sig: torch.Tensor) -> torch.Tensor:
    # x, mu, log_sig: [..., D], returns [...]
    const = -0.5 * math.log(2.0 * math.pi)
    log_prob = const - log_sig - 0.5 * ((x - mu) / torch.exp(log_sig)) ** 2
    return log_prob.sum(dim=-1)


@torch.no_grad()
def IS_estimate(
    encoder,
    decoder,
    x_data: torch.Tensor,
    task_id: int,
    device: torch.device,
    K: int = 5000,
    sample_W: bool = False,
    batch_size: int = 50,
    chunk_k: int = 100,
) -> Tuple[float, float]:
    """Importance-sampling estimate of test log-likelihood with K samples.

    Returns (mean_ll, ste_ll), where ste_ll = std(sample_ll)/sqrt(N).
    """
    encoder.eval()
    decoder.eval()
    x_data = x_data.to(device)

    ll_chunks = []
    n_total = x_data.shape[0]
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        x = x_data[start:end]  # [B, D]
        bsz = x.shape[0]

        mu_qz, log_sig_qz = encoder(x)  # [B, Z]
        z_dim = mu_qz.shape[1]

        bound_parts = []
        remaining = K
        while remaining > 0:
            cur_k = min(chunk_k, remaining)
            remaining -= cur_k

            eps = torch.randn(cur_k, bsz, z_dim, device=device, dtype=mu_qz.dtype)
            z = mu_qz.unsqueeze(0) + torch.exp(log_sig_qz).unsqueeze(0) * eps  # [cur_k,B,Z]
            z_flat = z.reshape(cur_k * bsz, z_dim)

            mu_x = decoder(z_flat, task_id=task_id, sample_W=sample_W).reshape(cur_k, bsz, -1)
            x_rep = x.unsqueeze(0).expand(cur_k, bsz, -1)

            logp = log_bernoulli_prob(
                x_rep.reshape(cur_k * bsz, -1), mu_x.reshape(cur_k * bsz, -1)
            ).reshape(cur_k, bsz)

            log_prior = _log_gaussian_diag(z, torch.zeros_like(z), torch.zeros_like(z))
            logq = _log_gaussian_diag(
                z,
                mu_qz.unsqueeze(0).expand(cur_k, bsz, z_dim),
                log_sig_qz.unsqueeze(0).expand(cur_k, bsz, z_dim),
            )
            bound_parts.append(logp - (logq - log_prior))

        bounds = torch.cat(bound_parts, dim=0)  # [K,B]
        ll_x = torch.logsumexp(bounds, dim=0) - math.log(K)  # [B]
        ll_chunks.append(ll_x)

    ll_all = torch.cat(ll_chunks, dim=0)  # [N]
    mean_ll = ll_all.mean().item()
    ste_ll = (ll_all.var(unbiased=False) / max(1, ll_all.numel())).sqrt().item()
    return mean_ll, ste_ll
