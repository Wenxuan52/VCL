from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alg.onlinevi import kl_param_shared, reset_shared_logsig, update_shared_prior
from alg.helper_functions import sample_gaussian
from models.bayesian_generator import BayesianDecoder
from models.encoder_no_shared import EncoderNoShared


def run_smoke_test() -> None:
    torch.manual_seed(0)
    bsz = 8
    encoder = EncoderNoShared(dimX=784, dimH=500, dimZ=50, n_layers=3)
    decoder = BayesianDecoder(dimZ=50, dimH=500, dimX=784)

    x = torch.rand(bsz, 784)
    mu, log_sig = encoder(x)
    assert mu.shape == (bsz, 50), mu.shape
    assert log_sig.shape == (bsz, 50), log_sig.shape

    z = sample_gaussian(mu, log_sig)
    assert z.shape == (bsz, 50), z.shape

    mu_x_sample = decoder(z, task_id=0, sample_W=True)
    mu_x_mean = decoder(z, task_id=0, sample_W=False)
    assert mu_x_sample.shape == (bsz, 784), mu_x_sample.shape
    assert mu_x_mean.shape == (bsz, 784), mu_x_mean.shape
    assert float(mu_x_sample.min()) >= 0.0 and float(mu_x_sample.max()) <= 1.0
    assert float(mu_x_mean.min()) >= 0.0 and float(mu_x_mean.max()) <= 1.0

    prior = update_shared_prior(decoder)
    kl = kl_param_shared(decoder, prior)
    assert torch.isfinite(kl)
    assert float(kl.abs()) < 1e-6, float(kl)

    reset_shared_logsig(decoder, value=-6.0)
    logsig_means = []
    for layer in decoder.shared.layers:
        logsig_means.append(layer.log_sig_W.mean().item())
        logsig_means.append(layer.log_sig_b.mean().item())
    for m in logsig_means:
        assert abs(m + 6.0) < 1e-6, m

    print("onlinevi component smoke test passed")


if __name__ == "__main__":
    run_smoke_test()
