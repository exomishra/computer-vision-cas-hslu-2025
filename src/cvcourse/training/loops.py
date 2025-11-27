"""Classic train/eval loops kept intentionally compact for teaching."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..utils.device import to_device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    scheduler: Optional[Iterable] = None,
) -> float:
    """Train ``model`` for a single epoch and return the average loss."""

    criterion = loss_fn or nn.CrossEntropyLoss()
    model = to_device(model, device)
    model.train()
    total_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = to_device(inputs, device), to_device(targets, device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)

    if scheduler:
        if hasattr(scheduler, "step"):
            scheduler.step()  # type: ignore[call-arg]
        else:
            for _ in scheduler:
                pass

    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
) -> float:
    """Return ``(accuracy, average_loss)`` for ``model`` on ``dataloader``."""

    criterion = loss_fn or nn.CrossEntropyLoss()
    model = to_device(model, device)
    model.eval()
    correct = 0
    total = 0
    loss_accumulator = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = to_device(inputs, device), to_device(targets, device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss_accumulator += loss.item() * inputs.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    return correct / max(total, 1), loss_accumulator / max(total, 1)
