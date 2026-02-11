from __future__ import annotations

from collections import OrderedDict

import torch

from .helper_functions import kl_diag_gaussians
from models.bayesian_generator import BayesianDecoder, BayesianLinear


def _iter_shared_bayesian_layers(decoder: BayesianDecoder):
    for idx, layer in enumerate(decoder.shared.layers):
        if not isinstance(layer, BayesianLinear):
            continue
        yield idx, layer


def _shared_prior_key_map(idx: int) -> dict[str, str]:
    return {
        "mu_W": f"shared.l{idx}.W.mu",
        "log_sig_W": f"shared.l{idx}.W.log_sig",
        "mu_b": f"shared.l{idx}.b.mu",
        "log_sig_b": f"shared.l{idx}.b.log_sig",
    }


def update_shared_prior(decoder: BayesianDecoder) -> dict[str, torch.Tensor]:
    prior = OrderedDict()
    for idx, layer in _iter_shared_bayesian_layers(decoder):
        key_map = _shared_prior_key_map(idx)
        prior[key_map["mu_W"]] = layer.mu_W.detach().cpu().clone()
        prior[key_map["log_sig_W"]] = layer.log_sig_W.detach().cpu().clone()
        prior[key_map["mu_b"]] = layer.mu_b.detach().cpu().clone()
        prior[key_map["log_sig_b"]] = layer.log_sig_b.detach().cpu().clone()
    return prior


def kl_param_shared(decoder: BayesianDecoder, shared_prior_params: dict[str, torch.Tensor]) -> torch.Tensor:
    device = next(decoder.parameters()).device
    dtype = next(decoder.parameters()).dtype

    total_kl = torch.zeros((), device=device, dtype=dtype)
    for idx, layer in _iter_shared_bayesian_layers(decoder):
        key_map = _shared_prior_key_map(idx)

        mu_p_w = shared_prior_params[key_map["mu_W"]].to(device=device, dtype=dtype)
        log_sig_p_w = shared_prior_params[key_map["log_sig_W"]].to(device=device, dtype=dtype)
        mu_p_b = shared_prior_params[key_map["mu_b"]].to(device=device, dtype=dtype)
        log_sig_p_b = shared_prior_params[key_map["log_sig_b"]].to(device=device, dtype=dtype)

        total_kl = total_kl + kl_diag_gaussians(
            layer.mu_W, layer.log_sig_W, mu_p_w, log_sig_p_w, reduce_dims=False
        ).sum()
        total_kl = total_kl + kl_diag_gaussians(
            layer.mu_b, layer.log_sig_b, mu_p_b, log_sig_p_b, reduce_dims=False
        ).sum()

    return total_kl


def reset_shared_logsig(decoder: BayesianDecoder, value: float = -6.0) -> None:
    with torch.no_grad():
        for _, layer in _iter_shared_bayesian_layers(decoder):
            layer.log_sig_W.fill_(float(value))
            layer.log_sig_b.fill_(float(value))
