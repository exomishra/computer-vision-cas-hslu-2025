"""Model zoo for the course (CNNs, detection, segmentation, pretrained backbones)."""

from .cnn_classification import SimpleCNN
from .detection import load_detection_model
from .pretrained import get_pretrained_resnet
from .segmentation import load_segmentation_model

__all__ = [
    "SimpleCNN",
    "load_detection_model",
    "load_segmentation_model",
    "get_pretrained_resnet",
]
