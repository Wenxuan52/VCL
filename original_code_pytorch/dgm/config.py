from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass(frozen=True)
class DGMConfig:
    data_name: str
    labels: List[List[int]]
    dimX: int
    dimZ: int
    dimH: int
    batch_size: int
    lr: float
    n_iter: int
    K_mc: int
    ll: str

    def to_dict(self) -> Dict:
        return asdict(self)


def get_config(data_name: str) -> DGMConfig:
    data_name = data_name.lower()
    if data_name == "mnist":
        return DGMConfig(
            data_name="mnist",
            labels=[[i] for i in range(10)],
            dimX=28 * 28,
            dimZ=50,
            dimH=500,
            batch_size=256,
            lr=1e-4,
            n_iter=200,
            K_mc=10,
            ll="bernoulli",
        )
    raise ValueError(f"Unsupported dataset: {data_name}")
