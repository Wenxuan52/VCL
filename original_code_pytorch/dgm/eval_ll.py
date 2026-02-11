from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np
import torch

from config import get_config
from data import load_mnist, load_notmnist, to_tensor
from eval_utils import importance_sample_ll
from models import DecoderHead, DecoderShared, Encoder, TaskModel
from visualisation import plot_images


def get_data(data_name: str, data_path: str, digits: list[int]):
    if data_name == "mnist":
        return load_mnist(digits)
    return load_notmnist(data_path, digits)


def main(args: argparse.Namespace) -> None:
    cfg = get_config(args.data_name)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    run_name = f"{args.data_name}_{args.method}{'' if args.method in ['noreg','onlinevi'] else f'_lbd{args.lbd:.1f}'}"
    save_root = Path("original_code_pytorch/dgm/save") / run_name
    fig_root = Path("original_code_pytorch/dgm/figs/visualisation")
    fig_root.mkdir(parents=True, exist_ok=True)

    eval_models: list[TaskModel] = []
    test_sets: list[np.ndarray] = []
    result = []
    vis_rows = []

    for task, digits in enumerate(cfg.labels, start=1):
        # --- load task data ---
        x_train, x_test, _, _ = get_data(args.data_name, args.data_path, digits)
        del x_train
        test_sets.append(x_test)

        # --- load checkpoint for this task ---
        ckpt_file = save_root / f"checkpoint_{task-1}.pt"
        if not ckpt_file.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_file}\n"
                f"Expected checkpoints under: {save_root}\n"
                f"Tip: confirm training produced checkpoint_0.pt ... checkpoint_{len(cfg.labels)-1}.pt"
            )

        ckpt = torch.load(ckpt_file, map_location=device)

        # ✅ IMPORTANT FIX:
        # Create a NEW DecoderShared per task-model. Do NOT reuse one shared instance across tasks.
        shared_t = DecoderShared(cfg.dim_x, args.dim_h, n_layers=2, last_activation="sigmoid").to(device)
        shared_t.load_state_dict(ckpt["decoder_shared"])

        model = TaskModel(
            encoder=Encoder(cfg.dim_x, args.dim_h, args.dim_z, n_layers=3).to(device),
            decoder_head=DecoderHead(args.dim_z, args.dim_h, n_layers=2).to(device),
            decoder_shared=shared_t,
        ).to(device)

        model.encoder.load_state_dict(ckpt["encoder"])
        model.decoder_head.load_state_dict(ckpt["decoder_head"])
        model.eval()  # ✅ eval mode for stable evaluation

        eval_models.append(model)

        # --- visualisation row for current task (generated samples) ---
        with torch.no_grad():
            one = model.sample(100, args.dim_z, device).cpu().numpy()  # (100, dim_x)
            row = np.zeros((10, cfg.dim_x), dtype=np.float32)
            row[:task] = one[:task]  # show first `task` images, pad the rest with zeros
            vis_rows.append(row)

        # --- evaluate on all previous tasks (including current) ---
        task_scores = []
        for i, m in enumerate(eval_models):
            x = to_tensor(test_sets[i], device)
            n = min(len(x), args.max_eval_n)
            x_eval = x[:n]

            with torch.no_grad():
                mean, var = importance_sample_ll(m, x_eval, cfg.ll, k=args.eval_k)

            # ✅ ste should use the number of evaluated samples (n), not full test-set length
            ste = np.sqrt(var / max(1, n))
            print(f"task {task} eval_on_{i+1}: ll={mean:.2f}, ste={ste:.4f}")
            task_scores.append((mean, ste))

        result.append(task_scores)

    # --- save visualisation (all tasks) ---
    plot_images(
        np.concatenate(vis_rows, axis=0),
        cfg.image_shape,
        str(fig_root),
        f"{args.data_name}_gen_all_{args.method}",
    )

    # --- dump results ---
    out_file = Path("original_code_pytorch/dgm/results") / f"{run_name}_eval.pkl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_name", choices=["mnist", "notmnist"])
    ap.add_argument("method", choices=["noreg", "ewc", "laplace", "si", "onlinevi"])
    ap.add_argument("lbd", type=float, nargs="?", default=1.0)
    ap.add_argument("--data-path", default="./data")
    ap.add_argument("--dim-z", type=int, default=50)
    ap.add_argument("--dim-h", type=int, default=500)
    ap.add_argument("--eval-k", type=int, default=500)
    ap.add_argument("--max-eval-n", type=int, default=1000, help="max number of test samples per task for evaluation")
    ap.add_argument("--cpu", action="store_true")
    main(ap.parse_args())
