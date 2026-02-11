from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import time
import numpy as np
import torch
from torch import optim

from config import get_config
from data import load_mnist, load_notmnist, to_tensor
from eval_utils import importance_sample_ll
from methods import ContinualRegularizer, elbo, make_fisher_diag
from models import DecoderHead, DecoderShared, Encoder, TaskModel
from visualisation import plot_images


def get_data(data_name: str, data_path: str, digits: list[int]):
    if data_name == "mnist":
        return load_mnist(digits)
    if data_name == "notmnist":
        return load_notmnist(data_path, digits)
    raise ValueError(data_name)


def run(args: argparse.Namespace) -> None:
    cfg = get_config(args.data_name)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    save_root = Path("original_code_pytorch/dgm/save") / f"{args.data_name}_{args.method}{'' if args.method in ['noreg','onlinevi'] else f'_lbd{args.lbd:.1f}'}"
    fig_root = Path("original_code_pytorch/dgm/figs") / save_root.name
    res_root = Path("original_code_pytorch/dgm/results")
    save_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)
    res_root.mkdir(parents=True, exist_ok=True)

    decoder_shared = DecoderShared(cfg.dim_x, args.dim_h, n_layers=2, last_activation="sigmoid").to(device)
    task_models: list[TaskModel] = []
    eval_sets: list[np.ndarray] = []
    result_list: list[list[tuple[float, float]]] = []

    reg = ContinualRegularizer(kind=args.method, lbd=args.lbd)
    if args.method == "onlinevi":
        reg.prior_mean = [p.detach().clone() for p in decoder_shared.parameters()]

    x_gen_all = []

    for task, digits in enumerate(cfg.labels, start=1):
        x_train, x_test, _, _ = get_data(args.data_name, args.data_path, digits)
        n_train = int(0.9 * len(x_train))
        x_valid = x_train[n_train:]
        x_train = x_train[:n_train]
        eval_sets.append(x_valid)

        model = TaskModel(
            encoder=Encoder(cfg.dim_x, args.dim_h, args.dim_z, n_layers=3).to(device),
            decoder_head=DecoderHead(args.dim_z, args.dim_h, n_layers=2).to(device),
            decoder_shared=decoder_shared,
        ).to(device)
        task_models.append(model)

        params = list(model.encoder.parameters()) + list(model.decoder_head.parameters()) + list(model.decoder_shared.parameters())
        opt = optim.Adam(params, lr=args.lr)

        shared_params = [p for p in model.decoder_shared.parameters()]
        n_batch = max(1, len(x_train) // args.batch_size)

        if args.method == "si":
            old_shared = [p.detach().clone() for p in shared_params]
            w_acc = [torch.zeros_like(p) for p in shared_params]

        max_epoch = args.n_iter_override if args.n_iter_override > 0 else cfg.n_iter
        for epoch in range(1, max_epoch + 1):
            t0 = time.time()
            idx = np.random.permutation(len(x_train))
            bound_epoch = 0.0
            for j in range(n_batch):
                b = idx[j * args.batch_size : (j + 1) * args.batch_size]
                if len(b) == 0:
                    continue
                xb = to_tensor(x_train[b], device)
                x_hat, mu, log_sig = model(xb, sample=True)
                bound = elbo(xb, x_hat, mu, log_sig, cfg.ll).mean()
                pen = reg.penalty(shared_params) / max(1, len(x_train))
                loss = -bound + pen

                opt.zero_grad()
                loss.backward()
                if args.method == "si":
                    grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in shared_params]
                opt.step()

                if args.method == "si":
                    new_shared = [p.detach().clone() for p in shared_params]
                    w_acc = [w + g * (n - o) for w, g, n, o in zip(w_acc, grads, new_shared, old_shared)]
                    old_shared = new_shared

                bound_epoch += bound.item() / n_batch
            print(f"task {task} epoch {epoch}: bound={bound_epoch:.2f}, time={time.time()-t0:.2f}s")

        ckpt = {
            "task": task,
            "decoder_shared": decoder_shared.state_dict(),
            "encoder": model.encoder.state_dict(),
            "decoder_head": model.decoder_head.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, save_root / f"checkpoint_{task-1}.pt")

        with torch.no_grad():
            samples = model.sample(100, args.dim_z, device).cpu().numpy()
            plot_images(samples, cfg.image_shape, str(fig_root), f"{args.data_name}_gen_task{task}_{task}")
            one_per_task = []
            for m in task_models:
                s = m.sample(100, args.dim_z, device).cpu().numpy()
                one_per_task.append(s[np.random.randint(0, len(s)) : np.random.randint(0, len(s)) + 1])
            canvas = np.zeros((10, cfg.dim_x), dtype=np.float32)
            concat = np.concatenate(one_per_task, axis=0)
            canvas[: len(concat)] = concat
            x_gen_all.append(canvas)

        task_scores = []
        for i, m in enumerate(task_models):
            xv = to_tensor(eval_sets[i], device)
            mean, var = importance_sample_ll(m, xv[: min(len(xv), 500)], cfg.ll, k=args.eval_k)
            ste = np.sqrt(var / max(1, len(xv)))
            print(f"eval task {i+1}: test_ll={mean:.2f}, ste={ste:.4f}")
            task_scores.append((mean, ste))
        result_list.append(task_scores)

        fisher = None
        if args.method in {"ewc", "laplace"}:
            xb = to_tensor(x_train[np.random.permutation(len(x_train))[: args.batch_size]], device)
            x_hat, mu, log_sig = model(xb, sample=True)
            b = elbo(xb, x_hat, mu, log_sig, cfg.ll).mean()
            fisher = make_fisher_diag(-b, shared_params)
        if args.method == "si":
            prev_omega = reg.omega if reg.omega is not None else [torch.zeros_like(p) for p in shared_params]
            prev_params = reg.old_params if reg.old_params is not None else [torch.zeros_like(p) for p in shared_params]
            cur = [p.detach().clone() for p in shared_params]
            si_omega = []
            for w, c, o, om in zip(w_acc, cur, prev_params, prev_omega):
                delta = c - o
                si_omega.append(om + w / (delta**2 + 1e-6))
            reg.omega = si_omega

        reg.update_after_task(shared_params, fisher_estimate=fisher, si_new_omega=(reg.omega if args.method == "si" else None))

    all_vis = np.concatenate(x_gen_all, axis=0)
    plot_images(all_vis, cfg.image_shape, str(fig_root), f"{args.data_name}_gen_all")

    result_file = res_root / f"{save_root.name}.pkl"
    with result_file.open("wb") as f:
        pickle.dump(result_list, f)
    print(f"saved results to {result_file}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("data_name", choices=["mnist", "notmnist"])
    p.add_argument("method", choices=["noreg", "ewc", "laplace", "si", "onlinevi"])
    p.add_argument("lbd", type=float, nargs="?", default=1.0)
    p.add_argument("--data-path", default="./data")
    p.add_argument("--dim-z", type=int, default=50)
    p.add_argument("--dim-h", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-k", type=int, default=100)
    p.add_argument("--n-iter-override", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
