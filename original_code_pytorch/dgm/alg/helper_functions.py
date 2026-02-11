from __future__ import annotations

import torch


def sample_gaussian(mu: torch.Tensor, log_sig: torch.Tensor) -> torch.Tensor:
    """Reparameterized Gaussian sample. log_sig is log(sigma)."""
    eps = torch.randn_like(mu)
    return mu + torch.exp(log_sig) * eps


def kl_diag_gaussians(
    mu_q: torch.Tensor,
    log_sig_q: torch.Tensor,
    mu_p: torch.Tensor | float,
    log_sig_p: torch.Tensor | float,
    reduce_dims: bool = True,
) -> torch.Tensor:
    """
    KL(q||p) for diagonal Gaussians.

    If reduce_dims=True, returns per-sample KL by summing non-batch dims -> shape [B].
    Otherwise returns elementwise KL with same shape as mu_q.
    """
    if not torch.is_tensor(mu_p):
        mu_p = torch.tensor(mu_p, device=mu_q.device, dtype=mu_q.dtype)
    if not torch.is_tensor(log_sig_p):
        log_sig_p = torch.tensor(log_sig_p, device=mu_q.device, dtype=mu_q.dtype)

    var_ratio = torch.exp(2.0 * log_sig_q - 2.0 * log_sig_p)
    mean_term = (mu_q - mu_p).pow(2) * torch.exp(-2.0 * log_sig_p)
    kl = 0.5 * (var_ratio + mean_term - 1.0 + 2.0 * (log_sig_p - log_sig_q))

    if not reduce_dims:
        return kl

    if kl.dim() <= 1:
        return kl

    dims = tuple(range(1, kl.dim()))
    return kl.sum(dim=dims)


def log_bernoulli_prob(x: torch.Tensor, mu: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Pixel-wise Bernoulli log p(x|mu), summed over non-batch dimensions.

    Returns shape [B] if x has batch dimension.
    """
    mu = mu.clamp(min=eps, max=1.0 - eps)
    logprob = x * torch.log(mu) + (1.0 - x) * torch.log(1.0 - mu)
    if logprob.dim() <= 1:
        return logprob
    dims = tuple(range(1, logprob.dim()))
    return logprob.sum(dim=dims)
