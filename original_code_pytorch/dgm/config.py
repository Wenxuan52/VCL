from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DGMConfig:
    labels: list[list[int]]
    n_iter: int
    dim_x: int
    image_shape: tuple[int, int]
    ll: str


def get_config(data_name: str) -> DGMConfig:
    if data_name == "mnist":
        return DGMConfig(
            labels=[[i] for i in range(10)],
            n_iter=200,
            dim_x=28 * 28,
            image_shape=(28, 28),
            ll="bernoulli",
        )
    if data_name == "notmnist":
        return DGMConfig(
            labels=[[i] for i in range(10)],
            n_iter=400,
            dim_x=28 * 28,
            image_shape=(28, 28),
            ll="bernoulli",
        )
    raise ValueError(f"unsupported dataset: {data_name}")
