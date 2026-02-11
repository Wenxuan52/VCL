from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from algorithm.eval_test_ll import IS_estimate
from config import get_config
from models.bayesian_generator import BayesianDecoder
from models.encoder_no_shared import EncoderNoShared
from models.mnist import MNIST_NOT_FOUND_MSG, load_mnist
from models.utils import ensure_dir, get_device, load_checkpoint
from models.visualisation import save_image_grid


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch DGM offline test-LL evaluation (importance sampling)")
    parser.add_argument("data_name", type=str)
    parser.add_argument("method", type=str)
    parser.add_argument("lbd", nargs="?", default=None)
    parser.add_argument("--K", type=int, default=5000)
    parser.add_argument("--sample_W", type=str, default="false")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--chunk_k", type=int, default=500)
    parser.add_argument("--n_preview", type=int, default=1, help="How many samples to average per digit in preview image")
    return parser.parse_args()


def _resolve_path_name(cfg, method: str) -> str:
    candidates = []
    method = method.lower()
    if method == "onlinevi" and cfg.K_mc > 1:
        candidates.append(f"{cfg.data_name}_{method}_K{cfg.K_mc}")
    candidates.append(f"{cfg.data_name}_{method}")

    for name in candidates:
        if (Path("save") / name).is_dir():
            return name
    # default to first candidate (likely training naming)
    return candidates[0]


@torch.no_grad()
def _generate_single_preview_row(
    decoder: BayesianDecoder,
    seen_tasks: list[int],
    dimZ: int,
    device: torch.device,
    sample_W: bool,
    n_preview: int = 1,
) -> np.ndarray:
    if n_preview < 1:
        raise ValueError(f"n_preview must be >= 1, got {n_preview}")

    row = np.zeros((10, 784), dtype=np.float32)

    for i in seen_tasks:
        acc = torch.zeros(784, device=device)
        for _ in range(n_preview):
            z = torch.randn(1, dimZ, device=device)
            x = decoder(z, task_id=i, sample_W=sample_W)  # [1, 784]
            acc += x[0]
        avg = acc / float(n_preview)

        # 可选：确保在 [0,1]（若 decoder 输出本来就是概率，这行可以不加）
        avg = avg.clamp(0.0, 1.0)

        row[i] = avg.detach().cpu().numpy()

    return row


def main() -> None:
    args = parse_args()
    if args.data_name.lower() != "mnist":
        raise ValueError("Only mnist is supported")
    if args.method.lower() != "onlinevi":
        raise ValueError("This evaluator currently supports onlinevi only")

    sample_W = str2bool(args.sample_W)
    cfg = get_config(args.data_name)
    device = get_device()

    path_name = _resolve_path_name(cfg, args.method)
    save_dir = Path("save") / path_name
    if not save_dir.is_dir():
        raise FileNotFoundError(f"Missing save directory: {save_dir}")

    print(f"device={device}")
    print(f"Evaluating checkpoints under: {save_dir}")
    print(f"IS settings: K={args.K}, batch_size={args.batch_size}, chunk_k={args.chunk_k}, sample_W={sample_W}")

    # preload all task test sets
    test_sets = []
    for task_id, digits in enumerate(cfg.labels):
        try:
            _, x_test = load_mnist(digits=digits, data_root="data")
        except RuntimeError as exc:
            if str(exc) == MNIST_NOT_FOUND_MSG:
                raise RuntimeError(MNIST_NOT_FOUND_MSG) from exc
            raise
        test_sets.append(x_test)
        print(f"Loaded test set task={task_id} digits={digits} N={x_test.shape[0]}")

    T = len(cfg.labels)
    result_matrix: list[list[tuple[float, float]]] = []
    gen_all_rows: list[np.ndarray] = []

    for t in range(T):
        ckpt_path = save_dir / f"checkpoint_{t}.pkl"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = load_checkpoint(ckpt_path, map_location=device)

        decoder = BayesianDecoder(dimZ=cfg.dimZ, dimH=cfg.dimH, dimX=cfg.dimX).to(device)
        decoder.load_state_dict(ckpt["decoder_state"])

        encoder_states = ckpt["encoder_states"]
        row: list[tuple[float, float]] = []
        print("=" * 90)
        print(f"Checkpoint task t={t}: evaluating seen tasks i<=t")

        for i in range(t + 1):
            if str(i) not in encoder_states:
                raise KeyError(f"Encoder state for task {i} missing in checkpoint {ckpt_path}")
            encoder = EncoderNoShared(dimX=cfg.dimX, dimH=cfg.dimH, dimZ=cfg.dimZ, n_layers=3, name=f"enc_{i}").to(device)
            encoder.load_state_dict(encoder_states[str(i)])

            mean_ll, ste_ll = IS_estimate(
                encoder=encoder,
                decoder=decoder,
                x_data=test_sets[i],
                task_id=i,
                device=device,
                K=args.K,
                sample_W=sample_W,
                batch_size=args.batch_size,
                chunk_k=args.chunk_k,
            )
            row.append((mean_ll, ste_ll))
            print(f"  task {i}: test_ll={mean_ll:.4f}, ste={ste_ll:.4f}")

        result_matrix.append(row)

        # gen_all preview row for this checkpoint
        seen_tasks = list(range(t + 1))
        row_imgs = _generate_single_preview_row(
                        decoder=decoder,
                        seen_tasks=seen_tasks,
                        dimZ=cfg.dimZ,
                        device=device,
                        sample_W=sample_W,
                        n_preview=args.n_preview,
                    )
        gen_all_rows.append(row_imgs)

    results_dir = Path("results")
    ensure_dir(results_dir)
    method_name = path_name.replace(f"{cfg.data_name}_", "", 1)
    result_path = results_dir / f"{cfg.data_name}_{method_name}.pkl"
    payload = {
        "result_matrix": result_matrix,
        "K": args.K,
        "sample_W": sample_W,
        "batch_size": args.batch_size,
        "chunk_k": args.chunk_k,
        "path_name": path_name,
    }
    with open(result_path, "wb") as f:
        pickle.dump(payload, f)

    vis_dir = Path("figs") / "visualisation"
    ensure_dir(vis_dir)
    gen_all = np.concatenate(gen_all_rows, axis=0)
    vis_path = vis_dir / f"{cfg.data_name}_gen_all_{args.method.lower()}.png"
    save_image_grid(str(vis_path), gen_all, n_rows=len(gen_all_rows), n_cols=10)

    print("=" * 90)
    print(f"Result matrix shape: [{len(result_matrix)}, <= {T}] (row t has t+1 entries)")
    print(f"Saved results to: {result_path}")
    print(f"Saved visualisation to: {vis_path}")


if __name__ == "__main__":
    main()
