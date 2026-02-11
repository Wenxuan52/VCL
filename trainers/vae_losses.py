import torch
import torch.nn.functional as F


def recon_loss_bernoulli_probs(recon_probs, x):
    """Per-sample Bernoulli NLL (sum over 784 dims) from probabilities."""
    if x.dim() == 4:
        x = x.view(x.size(0), -1)
    elif x.dim() != 2:
        raise ValueError(f"x must be [B,1,28,28] or [B,784], got {tuple(x.shape)}")

    probs = torch.clamp(recon_probs, min=1e-6, max=1.0 - 1e-6)
    bce = F.binary_cross_entropy(probs, x, reduction="none")
    return bce.sum(dim=1)


def latent_kl(mu, logvar):
    """Per-sample KL(q(z|x) || N(0,1))."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
