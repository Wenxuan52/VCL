from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reshape_and_tile_images(array: np.ndarray, shape=(28, 28), n_cols: int | None = None) -> np.ndarray:
    if n_cols is None:
        n_cols = int(math.sqrt(array.shape[0]))
    n_rows = int(np.ceil(float(array.shape[0]) / n_cols))

    def cell(i: int, j: int) -> np.ndarray:
        idx = i * n_cols + j
        if idx < array.shape[0]:
            return array[idx].reshape(*shape)
        return np.zeros(shape, dtype=array.dtype)

    rows = [np.concatenate([cell(i, j) for j in range(n_cols)], axis=1) for i in range(n_rows)]
    return np.concatenate(rows, axis=0)


def plot_images(images: np.ndarray, shape: tuple[int, int], path: str, filename: str, n_rows: int = 10) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = reshape_and_tile_images(images, shape, n_rows)
    out = out_dir / f"{filename}.png"
    plt.imsave(out, canvas, cmap="Greys_r")
    plt.close()
    return out
