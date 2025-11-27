"""Image visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def show_image(image: torch.Tensor, title: str | None = None) -> None:
    """Display a single CHW tensor image."""

    if image.dim() != 3:
        raise ValueError("show_image expects a 3D CHW tensor")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image.permute(1, 2, 0).cpu())
    ax.axis("off")
    if title:
        ax.set_title(title)
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


def show_batch(batch: torch.Tensor, title: str | None = None) -> None:
    """Display a grid of images from a mini-batch."""

    grid = make_grid(batch[:16], nrow=4, normalize=True, pad_value=0.5)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid.permute(1, 2, 0).cpu())
    ax.axis("off")
    if title:
        ax.set_title(title)
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()
