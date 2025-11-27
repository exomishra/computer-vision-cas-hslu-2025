"""Helpers for visualizing intermediate feature maps."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import torch


def visualize_feature_maps(activations: Iterable[torch.Tensor], num_maps: int = 4) -> None:
    """Plot a few feature maps from a hook or forward pass."""

    for tensor in activations:
        maps = tensor[: num_maps].detach().cpu()
        fig, axes = plt.subplots(1, num_maps, figsize=(num_maps * 2, 2))
        for idx in range(num_maps):
            axes[idx].imshow(maps[idx], cmap="viridis")
            axes[idx].axis("off")
        plt.show()
