from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_classifier import ClassifierMLP
from models.utils import ensure_dir, get_device, set_seed

MNIST_NOT_FOUND_MSG = "MNIST not found under data/. Please prepare it (download=False)."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP classifier for DGM classifier-uncertainty eval")
    parser.add_argument("data_name", type=str)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_mnist_split(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        ds = MNIST(root="data", train=train, download=False)
    except RuntimeError as exc:
        raise RuntimeError(MNIST_NOT_FOUND_MSG) from exc

    x = ds.data.float().view(-1, 784) / 255.0
    y = ds.targets.long()
    return x, y


def _eval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            n_correct += (pred == y).sum().item()
            n_total += y.numel()
    return 100.0 * n_correct / max(1, n_total)


def main() -> None:
    args = parse_args()
    if args.data_name.lower() != "mnist":
        raise ValueError("Only mnist is supported")

    set_seed(args.seed)
    device = get_device()

    x_train, y_train = _load_mnist_split(train=True)
    x_test, y_test = _load_mnist_split(train=False)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = ClassifierMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"device={device}")
    print(f"train={len(x_train)} test={len(x_test)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            n_seen += x.size(0)

        train_acc = _eval_accuracy(model, train_loader, device)
        test_acc = _eval_accuracy(model, test_loader, device)
        print(
            f"epoch={epoch:02d}/{args.epochs} "
            f"loss={loss_sum/max(1,n_seen):.4f} train_acc={train_acc:.2f}% test_acc={test_acc:.2f}%"
        )

    final_test_acc = _eval_accuracy(model, test_loader, device)
    print(f"Final test accuracy: {final_test_acc:.2f}%")

    save_dir = Path("original_code_pytorch/dgm/classifier/save")
    ensure_dir(save_dir)
    save_path = save_dir / f"{args.data_name.lower()}_weights.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved classifier weights to {save_path}")


if __name__ == "__main__":
    main()
