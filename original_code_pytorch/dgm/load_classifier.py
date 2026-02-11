from __future__ import annotations

from pathlib import Path
import torch

from classifier.train_classifier import Classifier


def load_model(data_name: str = "mnist", device: torch.device | None = None) -> Classifier:
    device = device or torch.device("cpu")
    model = Classifier().to(device)
    ckpt = Path("original_code_pytorch/dgm/classifier/save") / f"{data_name}_weights.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model
