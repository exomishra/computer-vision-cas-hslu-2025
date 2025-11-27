"""Device helpers that keep notebooks CPU-safe by default."""

from __future__ import annotations

from typing import Iterable

import torch


def get_device(allow_gpu: bool = True) -> torch.device:
    """Return ``cuda`` when available and allowed, otherwise ``cpu``."""

    if allow_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(obj, device: torch.device | None = None):
    """Move tensors, modules, or collections to ``device`` recursively."""

    target = device or get_device()
    if isinstance(obj, torch.nn.Module):
        return obj.to(target)
    if isinstance(obj, torch.Tensor):
        return obj.to(target)
    if isinstance(obj, dict):
        return {k: to_device(v, target) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        converted = [to_device(item, target) for item in obj]
        if isinstance(obj, tuple):
            return tuple(converted)
        if isinstance(obj, set):
            return set(converted)
        return converted
    return obj
