"""DataLoader helpers shared across notebooks."""

from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader, Subset

from ..config import ExperimentConfig
from .datasets import get_dataset
from .transforms import build_transforms


def get_dataloaders(config: ExperimentConfig) -> Tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader)`` for the requested dataset."""

    extras = config.extras or {}
    image_size = int(extras.get("image_size", 32))
    train_tf = build_transforms(image_size=image_size, augment=bool(extras.get("augment", True)))
    eval_tf = build_transforms(image_size=image_size, augment=False)

    train_ds = get_dataset(config.dataset, train=True, transform=train_tf)
    val_ds = get_dataset(config.dataset, train=False, transform=eval_tf)

    limit_train = extras.get("limit_train_samples")
    limit_val = extras.get("limit_val_samples")
    if limit_train:
        train_ds = Subset(train_ds, range(int(limit_train)))
    if limit_val:
        val_ds = Subset(val_ds, range(int(limit_val)))

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader
