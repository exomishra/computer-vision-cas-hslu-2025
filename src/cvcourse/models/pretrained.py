"""Wrappers around torchvision classification backbones."""

from __future__ import annotations

from typing import Literal

import torch
from torchvision import models

from ..utils.device import get_device, to_device


def get_pretrained_resnet(
    num_classes: int,
    feature_extract: bool = True,
    variant: Literal["resnet18", "resnet34"] = "resnet18",
    allow_gpu: bool = True,
) -> torch.nn.Module:
    """Return a pretrained ResNet tailored to the requested head size."""

    device = get_device(allow_gpu=allow_gpu)
    if variant == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT
        backbone = models.resnet34(weights=weights)
    else:
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

    if feature_extract:
        for param in backbone.parameters():
            param.requires_grad = False

    in_features = backbone.fc.in_features
    backbone.fc = torch.nn.Linear(in_features, num_classes)
    return to_device(backbone, device)


def get_resnet50_for_classification(
    num_classes: int,
    mode: Literal["feature_extraction", "partial_finetune", "full_finetune"] = "feature_extraction",
    allow_gpu: bool = True,
) -> torch.nn.Module:
    """Return a pretrained ResNet-50 configured for transfer learning.
    
    Args:
        num_classes: Number of output classes
        mode: Transfer learning strategy
            - feature_extraction: freeze all layers, train only new classifier head
            - partial_finetune: freeze early layers, train layer4 + head
            - full_finetune: train all layers (slowest but most flexible)
        allow_gpu: Whether to use GPU if available
    
    Returns:
        ResNet-50 model ready for training
    """
    device = get_device(allow_gpu=allow_gpu)
    
    # Load pretrained ResNet-50
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    # Configure freezing based on mode
    if mode == "feature_extraction":
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
    
    elif mode == "partial_finetune":
        # Freeze early layers (conv1, bn1, layer1, layer2, layer3)
        for name, param in model.named_parameters():
            if not any(layer in name for layer in ["layer4", "fc"]):
                param.requires_grad = False
    
    elif mode == "full_finetune":
        # Keep all layers trainable
        pass
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    
    return to_device(model, device)

