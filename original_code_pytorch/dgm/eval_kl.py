from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import torch

from alg.eval_test_class import generated_images_nll
from config import get_config
from load_classifier import load_model
from models.bayesian_generator import BayesianDecoder
from models.utils import ensure_dir, get_device, load_checkpoint


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classifier uncertainty on generated samples")
    parser.add_argument("data_name", type=str)
    parser.add_argument("method", type=str)
    parser.add_argument("lbd", nargs="?", default=None)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--sample_W", type=str, default="true")
    return parser.parse_args()


def _resolve_path_name(cfg, method: str) -> str:
    method = method.lower()
    candidates = []
    if method == "onlinevi" and cfg.K_mc > 1:
        candidates.append(f"{cfg.data_name}_{method}_K{cfg.K_mc}")
    candidates.append(f"{cfg.data_name}_{method}")

    for name in candidates:
        if (Path("save") / name).is_dir():
            return name
    return candidates[0]


def main() -> None:
    args = parse_args()
    if args.data_name.lower() != "mnist":
        raise ValueError("Only mnist is supported")
    if args.method.lower() != "onlinevi":
        raise ValueError("Only onlinevi is currently supported")

    cfg = get_config(args.data_name)
    device = get_device()
    sample_W = str2bool(args.sample_W)

    path_name = _resolve_path_name(cfg, args.method)
    save_dir = Path("save") / path_name
    if not save_dir.is_dir():
        raise FileNotFoundError(f"Missing save directory: {save_dir}")

    classifier = load_model(args.data_name.lower(), device=device)

    T = len(cfg.labels)
    results: list[list[tuple[float, float]]] = []

    print(f"device={device}")
    print(f"checkpoint_dir={save_dir}")
    print(f"n_samples={args.n_samples}, sample_W={sample_W}")

    for t in range(T):
        ckpt_path = save_dir / f"checkpoint_{t}.pkl"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path, map_location=device)

        decoder = BayesianDecoder(dimZ=cfg.dimZ, dimH=cfg.dimH, dimX=cfg.dimX).to(device)
        decoder.load_state_dict(ckpt["decoder_state"])
        decoder.eval()

        row: list[tuple[float, float]] = []
        print("=" * 88)
        print(f"checkpoint task={t}")
        for i in range(t + 1):
            mean, ste = generated_images_nll(
                decoder=decoder,
                classifier=classifier,
                task_id=i,
                n_samples=args.n_samples,
                device=device,
                sample_W=sample_W,
            )
            if not math.isfinite(mean) or not math.isfinite(ste):
                raise ValueError(f"Non-finite metric at checkpoint {t} task {i}: mean={mean}, ste={ste}")
            row.append((mean, ste))
            print(f"  task={i}: nll={mean:.4f}, ste={ste:.4f}")
        results.append(row)

    method_name = path_name.replace(f"{cfg.data_name}_", "", 1)
    out_path = Path("results") / f"{cfg.data_name}_{method_name}_gen_class.pkl"
    ensure_dir(out_path.parent)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "result_matrix": results,
                "n_samples": args.n_samples,
                "sample_W": sample_W,
                "checkpoint_dir": str(save_dir),
            },
            f,
        )

    print("=" * 88)
    print(f"saved results to {out_path}")
    print(f"result matrix shape: [{len(results)}, <= {T}] (row t has t+1 entries)")


if __name__ == "__main__":
    main()
