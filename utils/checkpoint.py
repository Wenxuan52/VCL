import torch


def save_checkpoint(path, model, optimizer=None, epoch=None, extra: dict | None = None):
    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if extra is not None:
        state["extra"] = extra

    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    return state
