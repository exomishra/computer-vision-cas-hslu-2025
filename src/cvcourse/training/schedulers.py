"""Scheduler convenience wrappers."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR


def build_scheduler(optimizer: Optimizer, step_size: int = 5, gamma: float = 0.5) -> StepLR:
    """Return a ``StepLR`` scheduler with gentle decay."""

    return StepLR(optimizer, step_size=step_size, gamma=gamma)
