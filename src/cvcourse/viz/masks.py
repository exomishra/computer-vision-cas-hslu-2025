"""Segmentation mask overlays."""

from __future__ import annotations

import torch


def overlay_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return an image tensor with ``mask`` blended in a given ``color``."""

    if image.dim() != 3:
        raise ValueError("overlay_mask expects CHW image tensor")

    mask = mask.squeeze().to(image.device, image.dtype)
    if mask.dim() != 2:
        raise ValueError("mask must reduce to HW after squeeze")

    color_tensor = torch.tensor(color, device=image.device, dtype=image.dtype).view(3, 1, 1)
    mask_3c = mask.unsqueeze(0).clamp(0, 1)
    blended = image * (1 - alpha * mask_3c) + color_tensor * (alpha * mask_3c)
    return blended.clamp(0, 1)
