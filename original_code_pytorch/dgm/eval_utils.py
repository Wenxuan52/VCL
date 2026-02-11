from __future__ import annotations

import math
import numpy as np
import torch

from methods import kl_gaussian, reconstruction_log_prob


@torch.no_grad()
def importance_sample_ll(model, x: torch.Tensor, ll: str, k: int = 100, sample_w: bool = False) -> tuple[float, float]:
    del sample_w

    # ---------- guards ----------
    # 确保输入在 [0,1]（MNIST 通常如此，但防御一下）
    x = x.clamp(0.0, 1.0)

    n = x.shape[0]

    # 用 expand/reshape 比 repeat 更省内存（行为等价）
    x_rep = x.unsqueeze(0).expand(k, n, x.shape[1]).reshape(k * n, x.shape[1])

    # ---------- encoder / sampling ----------
    mu, log_sig = model.encoder(x_rep)

    # 防止 exp(log_sig) 溢出 / 下溢
    log_sig = log_sig.clamp(-8.0, 8.0)
    std = torch.exp(log_sig).clamp_min(1e-6)

    z = mu + std * torch.randn_like(mu)

    # ---------- decode ----------
    x_hat = model.decode(z)

    # 关键：Bernoulli log prob 里会用 log(x_hat) / log(1-x_hat)，必须避免 0 或 1
    eps = 1e-6
    x_hat = x_hat.clamp(eps, 1.0 - eps)

    # ---------- log probs ----------
    log2pi = math.log(2.0 * math.pi)

    # log p(z) where p is N(0, I)
    log_prior = -0.5 * (log2pi + z.pow(2)).sum(dim=1)

    # log q(z|x) where q is N(mu, std^2)
    # 注意：这里 log_sig 是 log(std)，所以方差项是 2*log_sig
    normed = (z - mu) / std
    log_q = -0.5 * (log2pi + 2.0 * log_sig + normed.pow(2)).sum(dim=1)

    # log p(x|z)
    logp = reconstruction_log_prob(x_rep, x_hat, ll)

    # log w = log p(x,z) - log q(z|x) = logp + log_prior - log_q
    logw = (logp + log_prior - log_q).reshape(k, n)

    # ---------- stable log-mean-exp ----------
    # log(mean(exp(logw))) = logsumexp(logw) - log(k)
    ll_i = torch.logsumexp(logw, dim=0) - math.log(k)

    # 如果还有极端情况出现非有限值（理论上 clamp 后几乎不会了），这里兜底一下
    ll_i = torch.where(torch.isfinite(ll_i), ll_i, torch.full_like(ll_i, -1e9))

    mean = ll_i.mean().item()
    var = ll_i.var(unbiased=False).item()
    return mean, var

