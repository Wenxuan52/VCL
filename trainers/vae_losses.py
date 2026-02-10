import torch
import torch.nn.functional as F


def recon_loss_bce_with_logits(recon_logits, x):
    """Per-sample BCE reconstruction loss (sum over 784 dims)."""
    if x.dim() == 4:
        x = x.view(x.size(0), -1)
    elif x.dim() != 2:
        raise ValueError(f"x must be [B,1,28,28] or [B,784], got {tuple(x.shape)}")

    bce = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="none")
    return bce.sum(dim=1)


def latent_kl(mu, logvar):
    """Per-sample KL(q(z|x) || N(0,1))."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
