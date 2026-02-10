import argparse
import os

import torch
from torch.optim import Adam
from torchvision.utils import make_grid, save_image

from data.mnist import make_digit_task_loaders
from models.variational import DiagGaussianPosterior
from models.variational_layers import VarLinear
from models.vcl_vae import MultiHeadVCLVAE
from trainers.vae_losses import latent_kl, recon_loss_bce_with_logits
from utils.checkpoint import save_checkpoint


def snapshot_posteriors(module):
    """Snapshot all VarLinear posteriors as cloned tensors."""
    snap = []
    for submodule in module.modules():
        if isinstance(submodule, VarLinear):
            snap.append((
                submodule.q_weight.mu.detach().clone(),
                submodule.q_weight.log_sigma.detach().clone(),
            ))
            if submodule.q_bias is not None:
                snap.append((
                    submodule.q_bias.mu.detach().clone(),
                    submodule.q_bias.log_sigma.detach().clone(),
                ))
    return snap


def _posterior_kl_to_snapshot(posterior, snap_mu, snap_log_sigma):
    ref = DiagGaussianPosterior(
        shape=tuple(snap_mu.shape),
        init_mu=0.0,
        init_log_sigma=0.0,
        device=posterior.mu.device,
        dtype=posterior.mu.dtype,
    )
    with torch.no_grad():
        ref.mu.copy_(snap_mu.to(device=posterior.mu.device, dtype=posterior.mu.dtype))
        ref.log_sigma.copy_(
            snap_log_sigma.to(device=posterior.mu.device, dtype=posterior.mu.dtype)
        )
    return posterior.kl_to(ref)


def kl_module_to_snapshot(module, snapshot):
    """KL between current module posteriors and a previous snapshot."""
    total = None
    idx = 0
    for submodule in module.modules():
        if isinstance(submodule, VarLinear):
            mu, log_sigma = snapshot[idx]
            idx += 1
            kl_w = _posterior_kl_to_snapshot(submodule.q_weight, mu, log_sigma)
            total = kl_w if total is None else total + kl_w

            if submodule.q_bias is not None:
                mu, log_sigma = snapshot[idx]
                idx += 1
                kl_b = _posterior_kl_to_snapshot(submodule.q_bias, mu, log_sigma)
                total = kl_b if total is None else total + kl_b

    if total is None:
        return torch.tensor(0.0)
    return total


def kl_module_to_standard_normal(module):
    """KL(q || N(0,1)) over all VarLinear posteriors in module."""
    total = None
    for submodule in module.modules():
        if isinstance(submodule, VarLinear):
            w_prior = DiagGaussianPosterior.from_prior(
                tuple(submodule.q_weight.mu.shape),
                prior_mu=0.0,
                prior_sigma=1.0,
                device=submodule.q_weight.mu.device,
                dtype=submodule.q_weight.mu.dtype,
            )
            kl_w = submodule.q_weight.kl_to(w_prior)
            total = kl_w if total is None else total + kl_w

            if submodule.q_bias is not None:
                b_prior = DiagGaussianPosterior.from_prior(
                    tuple(submodule.q_bias.mu.shape),
                    prior_mu=0.0,
                    prior_sigma=1.0,
                    device=submodule.q_bias.mu.device,
                    dtype=submodule.q_bias.mu.dtype,
                )
                kl_b = submodule.q_bias.kl_to(b_prior)
                total = kl_b if total is None else total + kl_b

    if total is None:
        return torch.tensor(0.0)
    return total


def reset_log_sigma(module, log_sigma_init):
    for submodule in module.modules():
        if isinstance(submodule, VarLinear):
            with torch.no_grad():
                submodule.q_weight.log_sigma.fill_(float(log_sigma_init))
                if submodule.q_bias is not None:
                    submodule.q_bias.log_sigma.fill_(float(log_sigma_init))


def save_task_samples(model, task_id, digit_for_head, out_dir, device, sample_theta=False):
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, model.z_dim, device=device)
        imgs_logits = model.decode(digit_for_head, z, sample_theta=sample_theta, reshape=True)
        imgs = torch.sigmoid(imgs_logits)
        grid = make_grid(imgs, nrow=8)
    sample_path = os.path.join(out_dir, f"samples_task{task_id}_digit{digit_for_head}.png")
    save_image(grid, sample_path)
    print(f"saved samples: {sample_path}")


def train_task(
    model,
    task_id,
    loader,
    shared_snapshot,
    root="data",
    epochs=200,
    lr=1e-4,
    device=None,
    out_dir="outputs/vcl",
    sample_theta=True,
    log_sigma_init=1e-6,
):
    del root
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    os.makedirs(out_dir, exist_ok=True)

    model.ensure_task(task_id)
    model = model.to(device)

    task_key = str(task_id)

    # Appendix E-style init: keep mu, reset log_sigma for trainable variational blocks.
    reset_log_sigma(model.shared, log_sigma_init)
    reset_log_sigma(model.heads[task_key], log_sigma_init)

    params = list(model.encoders[task_key].parameters())
    params += list(model.heads[task_key].parameters())
    params += list(model.shared.parameters())
    optimizer = Adam(params, lr=lr)

    n_t = len(loader.dataset)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_klz = 0.0
        epoch_param_kl = 0.0
        n_samples = 0

        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)

            out = model.forward(task_id, x, sample_theta=sample_theta)
            recon = recon_loss_bce_with_logits(out["recon_logits"], x)
            klz = latent_kl(out["mu"], out["logvar"])

            shared_kl = kl_module_to_snapshot(model.shared, shared_snapshot)
            head_kl = kl_module_to_standard_normal(model.heads[task_key])
            param_kl = shared_kl + head_kl

            loss = (recon + klz).mean() + (param_kl / max(1, n_t))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            n_samples += bs
            epoch_loss += loss.item() * bs
            epoch_recon += recon.mean().item() * bs
            epoch_klz += klz.mean().item() * bs
            epoch_param_kl += param_kl.item() * bs

        print(
            f"task {task_id} | epoch {epoch:03d}/{epochs} | "
            f"loss={epoch_loss / max(1, n_samples):.4f} "
            f"recon={epoch_recon / max(1, n_samples):.4f} "
            f"klz={epoch_klz / max(1, n_samples):.4f} "
            f"param_kl={epoch_param_kl / max(1, n_samples):.4f}"
        )

    ckpt_path = os.path.join(out_dir, f"ckpt_task{task_id}.pt")
    save_checkpoint(
        ckpt_path,
        model,
        optimizer=optimizer,
        epoch=epochs,
        extra={"task_id": task_id},
    )
    print(f"saved checkpoint: {ckpt_path}")

    save_task_samples(model, task_id, task_id, out_dir, device, sample_theta=False)
    if task_id >= 1:
        save_task_samples(model, task_id, 0, out_dir, device, sample_theta=False)

    return model


def train_all(
    num_tasks=10,
    root="data",
    epochs=200,
    lr=1e-4,
    batch_size=128,
    num_workers=2,
    pin_memory=True,
    device=None,
    out_dir="outputs/vcl",
    sample_theta=True,
    log_sigma_init=1e-6,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = MultiHeadVCLVAE().to(device)

    loaders = make_digit_task_loaders(
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        train=True,
    )

    # q_{-1}(shared) starts as initial posterior.
    shared_snapshot = snapshot_posteriors(model.shared)

    for task_id in range(num_tasks):
        print(f"\n===== Training task {task_id} =====")
        model = train_task(
            model=model,
            task_id=task_id,
            loader=loaders[task_id],
            shared_snapshot=shared_snapshot,
            root=root,
            epochs=epochs,
            lr=lr,
            device=device,
            out_dir=out_dir,
            sample_theta=sample_theta,
            log_sigma_init=log_sigma_init,
        )

        # q_t(shared) snapshot for next task anchor.
        shared_snapshot = snapshot_posteriors(model.shared)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VCL trainer for MNIST digit generation")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="outputs/vcl")
    parser.add_argument("--log_sigma_init", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample_theta", action="store_true")

    args = parser.parse_args()

    train_all(
        num_tasks=10,
        root=args.root,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        out_dir=args.out_dir,
        sample_theta=args.sample_theta,
        log_sigma_init=args.log_sigma_init,
    )
