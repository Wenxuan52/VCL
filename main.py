from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithm.eval_test_ll import eval_elbo_on_dataset
from algorithm.helper_functions import kl_diag_gaussians, log_bernoulli_prob, sample_gaussian
from algorithm.onlinevi import kl_param_shared, reset_shared_logsig, update_shared_prior
from config import get_config
from models.bayesian_generator import BayesianDecoder
from models.encoder_no_shared import EncoderNoShared
from models.mnist import MNIST_NOT_FOUND_MSG, load_mnist, split_train_valid
from models.utils import build_checkpoint, ensure_dir, get_device, save_checkpoint, set_seed
from models.visualisation import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch DGM continual learning (onlinevi)")
    parser.add_argument("data_name", type=str)
    parser.add_argument("method", type=str)
    parser.add_argument("lbd", nargs="?", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_iter", type=int, default=None, help="Override epochs for quick debug")
    parser.add_argument("--sample_w_eval", action="store_true", help="Sample decoder weights during eval/visualization")
    return parser.parse_args()


def _validate_data(name: str, x: torch.Tensor) -> None:
    if x.dtype != torch.float32:
        raise TypeError(f"{name} dtype must be float32, got {x.dtype}")
    if x.ndim != 2 or x.shape[1] != 784:
        raise ValueError(f"{name} shape must be [N,784], got {tuple(x.shape)}")
    if x.numel() == 0:
        raise ValueError(f"{name} is empty")
    if x.min().item() < 0.0 or x.max().item() > 1.0:
        raise ValueError(f"{name} must be in [0,1], got [{x.min().item()}, {x.max().item()}]")


def _make_loader(x: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(x)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _collect_encoder_states(encoders: dict[int, EncoderNoShared]) -> dict[str, dict]:
    return {str(k): v.state_dict() for k, v in sorted(encoders.items(), key=lambda kv: kv[0])}


@torch.no_grad()
def _generate_task_images(
    decoder: BayesianDecoder,
    seen_tasks: list[int],
    dimZ: int,
    device: torch.device,
    n_gen: int = 100,
    sample_W: bool = False,
) -> dict[int, np.ndarray]:
    decoder.eval()
    out = {}
    for task_id in seen_tasks:
        z = torch.randn(n_gen, dimZ, device=device)
        mu_x = decoder(z, task_id=task_id, sample_W=sample_W)
        out[task_id] = mu_x.detach().cpu().numpy()
    return out


def main() -> None:
    args = parse_args()
    if args.data_name.lower() != "mnist":
        raise ValueError("Only mnist is supported in this script")
    if args.method.lower() != "onlinevi":
        raise ValueError("This task implements only onlinevi")

    cfg = get_config(args.data_name)
    n_iter = cfg.n_iter if args.n_iter is None else int(args.n_iter)

    set_seed(args.seed)
    device = get_device()

    method_name = f"{args.method.lower()}"
    if cfg.K_mc > 1:
        method_name = f"{method_name}_K{cfg.K_mc}"
    path_name = f"{cfg.data_name}_{method_name}"

    save_dir = Path("save") / path_name
    figs_dir = Path("figs") / path_name
    results_dir = Path("results")
    ensure_dir(save_dir)
    ensure_dir(figs_dir)
    ensure_dir(results_dir)

    print(f"device={device}")
    print(f"config={cfg.to_dict()}")
    print(f"Note: KL_theta scaling uses train_size only (90% split), matching TF behavior intent.")

    decoder = BayesianDecoder(dimZ=cfg.dimZ, dimH=cfg.dimH, dimX=cfg.dimX).to(device)
    shared_prior_params = update_shared_prior(decoder)

    encoders: dict[int, EncoderNoShared] = {}
    valid_sets: list[torch.Tensor] = []
    test_sets: list[torch.Tensor] = []
    valid_elbo_matrix: list[list[float]] = []
    gen_all_rows: list[np.ndarray] = []

    for task_id, digits in enumerate(cfg.labels):
        print("=" * 90)
        print(f"Task {task_id} / digit {digits}")
        try:
            x_train_full, x_test = load_mnist(digits=digits, data_root="data")
        except RuntimeError as exc:
            if str(exc) == MNIST_NOT_FOUND_MSG:
                raise RuntimeError(MNIST_NOT_FOUND_MSG) from exc
            raise

        x_train, x_valid = split_train_valid(x_train_full, valid_ratio=0.1, seed=args.seed)

        _validate_data("X_train", x_train)
        _validate_data("X_valid", x_valid)
        _validate_data("X_test", x_test)
        print(
            f"N_train={x_train.shape[0]} N_valid={x_valid.shape[0]} N_test={x_test.shape[0]} "
            f"dtype={x_train.dtype} range=[{x_train.min().item():.4f},{x_train.max().item():.4f}]"
        )

        valid_sets.append(x_valid)
        test_sets.append(x_test)

        encoder = EncoderNoShared(dimX=cfg.dimX, dimH=cfg.dimH, dimZ=cfg.dimZ, n_layers=3, name=f"enc_{task_id}").to(device)
        encoders[task_id] = encoder

        # Materialize current task head
        _ = decoder.get_head(task_id)

        train_loader = _make_loader(x_train, cfg.batch_size, shuffle=True)
        optim_params = list(decoder.parameters()) + list(encoder.parameters())
        optimizer = torch.optim.Adam(optim_params, lr=cfg.lr)
        n_data = x_train.shape[0]

        decoder.train()
        encoder.train()
        for epoch in range(1, n_iter + 1):
            epoch_loss = 0.0
            epoch_kl_theta = 0.0
            epoch_kl_z = 0.0
            epoch_recon = 0.0
            n_seen = 0
            for (x_batch,) in train_loader:
                x_batch = x_batch.to(device)
                bsz = x_batch.shape[0]

                mu_qz, log_sig_qz = encoder(x_batch)
                kl_z = kl_diag_gaussians(mu_qz, log_sig_qz, 0.0, 0.0, reduce_dims=True)

                recon_mc = torch.zeros_like(kl_z)
                for _ in range(cfg.K_mc):
                    z = sample_gaussian(mu_qz, log_sig_qz)
                    mu_x = decoder(z, task_id=task_id, sample_W=True)
                    recon_mc = recon_mc + log_bernoulli_prob(x_batch, mu_x) / float(cfg.K_mc)

                elbo = recon_mc - kl_z
                bound = elbo.mean()
                kl_theta = kl_param_shared(decoder, shared_prior_params)
                loss_total = -bound + kl_theta / float(n_data)

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                epoch_loss += loss_total.item() * bsz
                epoch_kl_theta += kl_theta.item() * bsz
                epoch_kl_z += kl_z.mean().item() * bsz
                epoch_recon += recon_mc.mean().item() * bsz
                n_seen += bsz

            print(
                f"task={task_id} epoch={epoch:03d}/{n_iter} "
                f"loss={epoch_loss/n_seen:.4f} recon={epoch_recon/n_seen:.4f} "
                f"kl_z={epoch_kl_z/n_seen:.4f} kl_theta={epoch_kl_theta/n_seen:.4f}"
            )

        # Valid ELBO on all seen tasks
        row = []
        for seen_task in range(task_id + 1):
            va_loader = _make_loader(valid_sets[seen_task], cfg.batch_size, shuffle=False)
            va_elbo = eval_elbo_on_dataset(
                encoder=encoders[seen_task],
                decoder=decoder,
                dataloader=va_loader,
                task_id=seen_task,
                device=device,
                K=50,
                sample_W=args.sample_w_eval,
            )
            row.append(float(va_elbo))
            print(f"  valid_elbo(task={seen_task})={va_elbo:.4f}")
        valid_elbo_matrix.append(row)

        # Save checkpoint
        ckpt = build_checkpoint(
            task=task_id,
            cfg_dict=cfg.to_dict() | {"method": args.method.lower(), "n_iter": n_iter, "seed": args.seed},
            decoder_state=decoder.state_dict(),
            encoder_states=_collect_encoder_states(encoders),
            shared_prior=shared_prior_params,
        )
        ckpt_path = save_dir / f"checkpoint_{task_id}.pkl"
        save_checkpoint(ckpt, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # Generate samples for all seen tasks (100 each)
        seen_tasks = list(range(task_id + 1))
        gen_dict = _generate_task_images(
            decoder,
            seen_tasks=seen_tasks,
            dimZ=cfg.dimZ,
            device=device,
            n_gen=100,
            sample_W=args.sample_w_eval,
        )
        for seen_task, imgs in gen_dict.items():
            img_path = figs_dir / f"{cfg.data_name}_gen_task{task_id+1}_{seen_task+1}.png"
            save_image_grid(str(img_path), imgs, n_rows=10, n_cols=10)

        # Build cumulative gen_all rows (10 columns, unseen -> zero image)
        row_imgs = np.zeros((10, cfg.dimX), dtype=np.float32)
        for seen_task in seen_tasks:
            row_imgs[seen_task] = gen_dict[seen_task][0]
        gen_all_rows.append(row_imgs)
        gen_all = np.concatenate(gen_all_rows, axis=0)
        gen_all_path = figs_dir / f"{cfg.data_name}_gen_all.png"
        save_image_grid(str(gen_all_path), gen_all, n_rows=len(gen_all_rows), n_cols=10)

        # Update prior and reset shared log sigma
        shared_prior_params = update_shared_prior(decoder)
        reset_shared_logsig(decoder, value=-6.0)
        shared_logsig_means = []
        for layer in decoder.shared.layers:
            shared_logsig_means.append(layer.log_sig_W.mean().item())
            shared_logsig_means.append(layer.log_sig_b.mean().item())
        print(
            "Updated shared prior and reset shared log_sig. "
            f"mean(log_sig)={sum(shared_logsig_means)/len(shared_logsig_means):.4f}"
        )

    result_path = results_dir / f"{cfg.data_name}_{method_name}.pkl"
    payload = {
        "valid_elbo_matrix": valid_elbo_matrix,
        "n_iter": n_iter,
        "config": cfg.to_dict(),
    }
    with open(result_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved results: {result_path}")


if __name__ == "__main__":
    main()
