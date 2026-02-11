from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms


def _to_one_hot(y: np.ndarray, n_class: int = 10) -> np.ndarray:
    out = np.zeros((len(y), n_class), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def load_mnist(digits: list[int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, transform=tfm, download=True)
    test_ds = datasets.MNIST(root="./data", train=False, transform=tfm, download=True)

    x_train = train_ds.data.float().view(-1, 784).numpy() / 255.0
    y_train = train_ds.targets.numpy()
    x_test = test_ds.data.float().view(-1, 784).numpy() / 255.0
    y_test = test_ds.targets.numpy()

    if digits is not None:
        train_mask = np.isin(y_train, digits)
        test_mask = np.isin(y_test, digits)
        x_train, y_train = x_train[train_mask], y_train[train_mask]
        x_test, y_test = x_test[test_mask], y_test[test_mask]

    return x_train.astype(np.float32), x_test.astype(np.float32), _to_one_hot(y_train), _to_one_hot(y_test)


def load_notmnist(data_path: str, digits: list[int] | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.io import loadmat

    mat_path = Path(data_path) / "notMNIST" / "notMNIST_small.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"notMNIST mat file not found: {mat_path}")

    out = loadmat(mat_path)
    x = out["images"].transpose(2, 0, 1) / 255.0
    y = out["labels"].reshape(-1)

    if digits is not None:
        mask = np.isin(y, digits)
        x = x[mask]
        y = y[mask]

    x = x.reshape(x.shape[0], -1).astype(np.float32)
    y_oh = _to_one_hot(y.astype(np.int64))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(x.shape[0])
    n_train = int(0.9 * x.shape[0])

    tr, te = idx[:n_train], idx[n_train:]
    return x[tr], x[te], y_oh[tr], y_oh[te]


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).to(device=device, dtype=torch.float32)
