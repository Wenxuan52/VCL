from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data import load_mnist, load_notmnist, to_tensor


class Classifier(nn.Module):
    def __init__(self, dim_in: int = 784, dim_h: int = 1000, n_layer: int = 3, n_class: int = 10):
        super().__init__()
        layers = [nn.Linear(dim_in, dim_h), nn.ReLU(), nn.Dropout(0.2)]
        for _ in range(n_layer - 1):
            layers += [nn.Linear(dim_h, dim_h), nn.ReLU(), nn.Dropout(0.2)]
        layers += [nn.Linear(dim_h, n_class)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main(args):
    if args.data_name == "mnist":
        x_train, x_test, y_train, y_test = load_mnist()
    else:
        xs_tr, xs_te, ys_tr, ys_te = [], [], [], []
        for i in range(10):
            a, b, c, d = load_notmnist(args.data_path, [i])
            xs_tr.append(a); xs_te.append(b); ys_tr.append(c); ys_te.append(d)
        x_train, x_test = np.concatenate(xs_tr), np.concatenate(xs_te)
        y_train, y_test = np.concatenate(ys_tr), np.concatenate(ys_te)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    clf = Classifier().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.argmax(1)))
    test_x = to_tensor(x_test, device)
    test_y = torch.from_numpy(y_test.argmax(1)).to(device)

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    for ep in range(1, args.epochs + 1):
        clf.train()
        loss_total = 0.0
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device)
            logits = clf(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_total += loss.item()
        if ep % 10 == 0 or ep == 1:
            clf.eval()
            with torch.no_grad():
                acc = (clf(test_x).argmax(1) == test_y).float().mean().item()
            print(f"epoch {ep}: loss={loss_total/len(loader):.4f}, test_acc={acc:.4f}")

    out = Path("original_code_pytorch/dgm/classifier/save")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(clf.state_dict(), out / f"{args.data_name}_weights.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-name", default="mnist", choices=["mnist", "notmnist"])
    ap.add_argument("--data-path", default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--cpu", action="store_true")
    main(ap.parse_args())
