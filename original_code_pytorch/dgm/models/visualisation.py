from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _to_flat_images(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images)
    if arr.ndim == 2 and arr.shape[1] == 784:
        return arr
    if arr.ndim == 4 and arr.shape[1:] == (1, 28, 28):
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 3 and arr.shape[1:] == (28, 28):
        return arr.reshape(arr.shape[0], -1)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def reshape_and_tile_images(
    images: np.ndarray,
    n_rows: int = 10,
    n_cols: int = 10,
    img_shape: tuple[int, int] = (28, 28),
) -> np.ndarray:
    flat = _to_flat_images(images)
    h, w = img_shape
    canvas = np.zeros((n_rows * h, n_cols * w), dtype=np.float32)

    total = n_rows * n_cols
    for idx in range(min(total, flat.shape[0])):
        r = idx // n_cols
        c = idx % n_cols
        tile = flat[idx].reshape(h, w)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile

    return canvas


def save_image_grid(
    path: str,
    images: np.ndarray,
    n_rows: int = 10,
    n_cols: int = 10,
    img_shape: tuple[int, int] = (28, 28),
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = reshape_and_tile_images(images, n_rows=n_rows, n_cols=n_cols, img_shape=img_shape)
    grid = np.clip(grid, 0.0, 1.0)
    Image.fromarray((grid * 255).astype(np.uint8), mode="L").save(out_path)
    return out_path
