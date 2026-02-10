from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class _DigitOnlyDataset(Dataset):
    """Wrap an MNIST dataset, keep only one digit, and return image tensors only."""

    def __init__(
        self,
        base_dataset: datasets.MNIST,
        digit: int,
        binarize: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        if digit < 0 or digit > 9:
            raise ValueError(f"digit must be in [0, 9], got {digit}")
        if binarize not in {None, "dynamic", "threshold"}:
            raise ValueError(
                f"binarize must be one of None/'dynamic'/'threshold', got {binarize}"
            )

        self.base_dataset = base_dataset
        self.binarize = binarize
        self.threshold = threshold

        targets = base_dataset.targets
        self.indices = (targets == digit).nonzero(as_tuple=True)[0].tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = self.base_dataset[self.indices[index]]

        if self.binarize == "dynamic":
            image = torch.bernoulli(image)
        elif self.binarize == "threshold":
            image = (image >= self.threshold).to(image.dtype)

        return image


def get_mnist_datasets(
    root: str,
    train: bool = True,
    transform: Callable | None = None,
    download: bool = False,
) -> datasets.MNIST:
    """Load MNIST dataset from local path."""
    if transform is None:
        transform = transforms.ToTensor()

    return datasets.MNIST(root=root, train=train, transform=transform, download=download)


def make_digit_task_loaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    train: bool = True,
    transform: Callable | None = None,
    binarize: str | None = None,
) -> dict[int, DataLoader]:
    """Create one DataLoader per digit class (0-9), each yielding image batches only."""
    mnist_dataset = get_mnist_datasets(
        root=root,
        train=train,
        transform=transform,
        download=False,
    )

    loaders: dict[int, DataLoader] = {}
    for digit in range(10):
        digit_dataset = _DigitOnlyDataset(
            base_dataset=mnist_dataset,
            digit=digit,
            binarize=binarize,
        )
        loaders[digit] = DataLoader(
            digit_dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loaders


def print_task_stats(loaders: dict[int, DataLoader]) -> None:
    """Print sample counts and one-batch pixel range per digit loader."""
    for digit in sorted(loaders.keys()):
        loader = loaders[digit]
        sample_count = len(loader.dataset)
        batch = next(iter(loader))
        pixel_min = batch.min().item()
        pixel_max = batch.max().item()
        print(
            f"digit={digit}: samples={sample_count}, "
            f"batch_shape={tuple(batch.shape)}, min={pixel_min:.4f}, max={pixel_max:.4f}"
        )


if __name__ == "__main__":
    data_root = "data"

    print("=== Train task loaders ===")
    train_loaders = make_digit_task_loaders(root=data_root, train=True)
    print_task_stats(train_loaders)

    print("\n=== Test task loaders ===")
    test_loaders = make_digit_task_loaders(root=data_root, train=False)
    print_task_stats(test_loaders)

    digit = 0
    batch = next(iter(train_loaders[digit]))
    print(
        f"\nRandom batch from digit={digit}: shape={tuple(batch.shape)}, "
        f"min={batch.min().item():.4f}, max={batch.max().item():.4f}"
    )
