"""Simple metric helpers."""

from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return categorical accuracy for ``logits`` vs ``targets``."""

    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()
