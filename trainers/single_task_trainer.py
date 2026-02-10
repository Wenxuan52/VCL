import argparse
import os

import torch
from torch.optim import Adam
from torchvision.utils import make_grid, save_image

from data.mnist import make_digit_task_loaders
from models.vcl_vae import MultiHeadVCLVAE
from trainers.vae_losses import latent_kl, recon_loss_bce_with_logits
from utils.checkpoint import save_checkpoint


def train_single_task(
    task_id,
    root="data",
    epochs=200,
    lr=1e-4,
    batch_size=128,
    device=None,
    out_dir="outputs/single_task",
    sample_theta=True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    os.makedirs(out_dir, exist_ok=True)

    loaders = make_digit_task_loaders(root=root, batch_size=batch_size, train=True)
    if task_id not in loaders:
        raise ValueError(f"task_id must be in [0..9], got {task_id}")
    loader = loaders[task_id]

    model = MultiHeadVCLVAE()
    model.ensure_task(task_id)
    model = model.to(device)

    task_key = str(task_id)
    params = list(model.encoders[task_key].parameters())
    params += list(model.heads[task_key].parameters())
    params += list(model.shared.parameters())

    optimizer = Adam(params, lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_klz = 0.0
        n_samples = 0

        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)

            out = model.forward(task_id, x, sample_theta=sample_theta)
            recon = recon_loss_bce_with_logits(out["recon_logits"], x)
            klz = latent_kl(out["mu"], out["logvar"])

            loss = (recon + klz).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            n_samples += bs
            epoch_loss += loss.item() * bs
            epoch_recon += recon.mean().item() * bs
            epoch_klz += klz.mean().item() * bs

        avg_loss = epoch_loss / max(1, n_samples)
        avg_recon = epoch_recon / max(1, n_samples)
        avg_klz = epoch_klz / max(1, n_samples)
        print(
            f"epoch {epoch:03d}/{epochs} | "
            f"loss={avg_loss:.4f} recon={avg_recon:.4f} klz={avg_klz:.4f}"
        )

    ckpt_path = os.path.join(out_dir, f"ckpt_task{task_id}.pt")
    save_checkpoint(ckpt_path, model, optimizer=optimizer, epoch=epochs, extra={"task_id": task_id})
    print(f"saved checkpoint: {ckpt_path}")

    model.eval()
    with torch.no_grad():
        z = torch.randn(64, model.z_dim, device=device)
        imgs_logits = model.decode(task_id, z, sample_theta=False, reshape=True)
        imgs = torch.sigmoid(imgs_logits)
        grid = make_grid(imgs, nrow=8)

    sample_path = os.path.join(out_dir, f"samples_task{task_id}.png")
    save_image(grid, sample_path)
    print(f"saved samples: {sample_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-task VAE baseline trainer")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="outputs/single_task")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    train_single_task(
        task_id=args.task_id,
        root=args.root,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        out_dir=args.out_dir,
    )
