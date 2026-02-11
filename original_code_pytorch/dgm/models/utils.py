from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def build_rng_state() -> dict[str, Any]:
    state = {
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def build_checkpoint(
    task: int,
    cfg_dict: dict[str, Any],
    decoder_state: dict[str, Any],
    encoder_states: dict[str, dict[str, Any]],
    shared_prior: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return {
        "task": task,
        "config": cfg_dict,
        "decoder_state": decoder_state,
        "encoder_states": encoder_states,
        "shared_prior": shared_prior,
        "rng_state": build_rng_state(),
    }


def save_checkpoint(obj: Any, path: str | os.PathLike[str]) -> None:
    ensure_dir(Path(path).parent)
    torch.save(obj, path)


def load_checkpoint(path: str | os.PathLike[str], map_location: str | torch.device = "cpu") -> Any:
    return torch.load(path, map_location=map_location)
