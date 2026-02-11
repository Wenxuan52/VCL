from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import numpy as np
import torch

from config import get_config
from load_classifier import load_model
from models import DecoderHead, DecoderShared, Encoder, TaskModel


def main(args):
    cfg = get_config(args.data_name)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run_name = f"{args.data_name}_{args.method}{'' if args.method in ['noreg','onlinevi'] else f'_lbd{args.lbd:.1f}'}"
    save_root = Path("original_code_pytorch/dgm/save") / run_name

    shared = DecoderShared(cfg.dim_x, args.dim_h, n_layers=2, last_activation="sigmoid").to(device)
    clf = load_model(args.data_name, device)
    results = []

    for task in range(1, 11):
        ckpt = torch.load(save_root / f"checkpoint_{task-1}.pt", map_location=device)
        shared.load_state_dict(ckpt["decoder_shared"])
        model = TaskModel(
            encoder=Encoder(cfg.dim_x, args.dim_h, args.dim_z, n_layers=3).to(device),
            decoder_head=DecoderHead(args.dim_z, args.dim_h, n_layers=2).to(device),
            decoder_shared=shared,
        ).to(device)
        model.encoder.load_state_dict(ckpt["encoder"])
        model.decoder_head.load_state_dict(ckpt["decoder_head"])
        model.eval()

        with torch.no_grad():
            kls = []
            for class_id in range(task):
                z = torch.randn(args.n_gen, args.dim_z, device=device)
                x = model.decode(z)
                logits = clf(x)
                probs = logits.softmax(dim=1)
                target = torch.zeros_like(probs)
                target[:, class_id] = 1.0
                kl = -(target * (probs.clamp_min(1e-9).log())).sum(dim=1)
                kls.append((kl.mean().item(), kl.std().item() / np.sqrt(args.n_gen)))
            results.append(kls)
            print(f"task {task}: {kls}")

    out = Path("original_code_pytorch/dgm/results") / f"{run_name}_kl.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_name", choices=["mnist", "notmnist"])
    ap.add_argument("method", choices=["noreg", "ewc", "laplace", "si", "onlinevi"])
    ap.add_argument("lbd", type=float, nargs="?", default=1.0)
    ap.add_argument("--dim-z", type=int, default=50)
    ap.add_argument("--dim-h", type=int, default=500)
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--cpu", action="store_true")
    main(ap.parse_args())
