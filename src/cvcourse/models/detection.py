"""Pretrained detection helpers built on torchvision."""

from __future__ import annotations

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
)

from ..utils.device import get_device, to_device


def load_detection_model(allow_gpu: bool = True) -> torch.nn.Module:
    """Return a pretrained Faster R-CNN model ready for inference."""

    device = get_device(allow_gpu=allow_gpu)
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    model.eval()
    return to_device(model, device)


def get_faster_rcnn(allow_gpu: bool = True, use_resnet: bool = False) -> torch.nn.Module:
    """Return a pretrained Faster R-CNN for object detection.
    
    Args:
        allow_gpu: Whether to move model to GPU if available.
        use_resnet: If True, use ResNet-50 backbone (slower, more accurate).
                   If False, use MobileNetV3 (faster, lighter).
    """
    device = get_device(allow_gpu=allow_gpu)
    if use_resnet:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
    else:
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    model.eval()
    return to_device(model, device)


def get_mask_rcnn(allow_gpu: bool = True) -> torch.nn.Module:
    """Return a pretrained Mask R-CNN for instance segmentation."""
    device = get_device(allow_gpu=allow_gpu)
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return to_device(model, device)
