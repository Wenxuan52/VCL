from __future__ import annotations

from typing import List, Tuple

import torch
from torchvision.datasets import MNIST


MNIST_NOT_FOUND_MSG = "MNIST not found under data/. Please prepare it (download=False)."


def _load_split(data_root: str, train: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        ds = MNIST(root=data_root, train=train, download=False)
    except RuntimeError as exc:
        raise RuntimeError(MNIST_NOT_FOUND_MSG) from exc

    x = ds.data.float().view(-1, 28 * 28) / 255.0
    y = ds.targets
    return x, y


def load_mnist(digits: List[int], data_root: str = "data") -> Tuple[torch.Tensor, torch.Tensor]:
    if len(digits) == 0:
        raise ValueError("digits must be a non-empty list")

    x_train, y_train = _load_split(data_root, train=True)
    x_test, y_test = _load_split(data_root, train=False)

    digits_t = torch.tensor(digits, dtype=y_train.dtype)
    train_mask = (y_train[:, None] == digits_t[None, :]).any(dim=1)
    test_mask = (y_test[:, None] == digits_t[None, :]).any(dim=1)

    x_train = x_train[train_mask].to(dtype=torch.float32)
    x_test = x_test[test_mask].to(dtype=torch.float32)

    return x_train, x_test


def split_train_valid(
    x_train: torch.Tensor, valid_ratio: float = 0.1, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 < valid_ratio < 1.0):
        raise ValueError("valid_ratio must be in (0, 1)")
    n = x_train.shape[0]
    if n == 0:
        raise ValueError("x_train is empty")

    g = torch.Generator(device=x_train.device)
    g.manual_seed(seed)
    indices = torch.randperm(n, generator=g, device=x_train.device)
    n_valid = max(1, int(round(n * valid_ratio)))

    valid_idx = indices[:n_valid]
    train_idx = indices[n_valid:]

    if train_idx.numel() == 0:
        raise ValueError("valid split consumed all samples; reduce valid_ratio")

    return x_train[train_idx], x_train[valid_idx]
