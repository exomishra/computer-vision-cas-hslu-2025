"""Dataset factories built on top of torchvision for CPU-friendly demos."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple, Type

import torch
from torchvision import datasets
from torchvision.transforms import Compose

from ..utils.paths import get_data_dir

DATASET_REGISTRY: Dict[str, Type[datasets.VisionDataset]] = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}

# Mapping from COCO-91 IDs (used by pretrained torchvision models) to contiguous 1-80 IDs
COCO_91_TO_80 = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20,
    22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
    35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
    46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50,
    56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60,
    67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70,
    80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80,
}


def get_dataset(
    name: str,
    train: bool,
    transform: Compose | None = None,
    download: bool = True,
    root: str | Path | None = None,
):
    """Return a torchvision dataset configured with sane defaults."""

    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY)}")
    dataset_cls = DATASET_REGISTRY[key]
    target_root = Path(root) if root else get_data_dir()
    return dataset_cls(root=target_root, train=train, transform=transform, download=download)


def load_coco_annotation(label_path: Path) -> Tuple[int, torch.Tensor]:
    """Parse a YOLO-format COCO label file and return (class_id, bbox_tensor)."""
    with open(label_path) as f:
        lines = f.readlines()
    if not lines:
        return -1, torch.empty((0, 4))
    class_id = int(lines[0].split()[0])
    boxes = []
    for line in lines:
        parts = line.strip().split()
        x_center, y_center, w, h = map(float, parts[1:5])
        boxes.append([x_center, y_center, w, h])
    return class_id, torch.tensor(boxes)


def denormalize_bbox(
    bbox: torch.Tensor, image_width: int, image_height: int
) -> torch.Tensor:
    """Convert normalized YOLO bbox [x_center, y_center, w, h] to pixel [x1, y1, x2, y2]."""
    x_center, y_center, w, h = bbox
    x1 = (x_center - w / 2) * image_width
    y1 = (y_center - h / 2) * image_height
    x2 = (x_center + w / 2) * image_width
    y2 = (y_center + h / 2) * image_height
    return torch.tensor([x1, y1, x2, y2])
