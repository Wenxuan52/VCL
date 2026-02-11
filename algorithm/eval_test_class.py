from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def generated_images_nll(
    decoder,
    classifier,
    task_id: int,
    n_samples: int = 100,
    device: torch.device | None = None,
    sample_W: bool = False,
) -> tuple[float, float]:
    if device is None:
        device = next(decoder.parameters()).device

    decoder.eval()
    classifier.eval()

    z_dim = getattr(decoder, "dimZ", 50)
    z = torch.randn(n_samples, z_dim, device=device)
    x_gen = decoder(z, task_id=task_id, sample_W=sample_W)
    logits = classifier(x_gen)
    log_probs = F.log_softmax(logits, dim=1)

    target = torch.full((n_samples,), task_id, dtype=torch.long, device=device)
    nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

    mean = nll.mean().item()
    ste = (nll.var(unbiased=False) / max(1, nll.numel())).sqrt().item()
    return mean, ste
