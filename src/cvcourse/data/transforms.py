"""Shared torchvision transforms for quick experiments."""

from __future__ import annotations

from typing import List, Literal

from torchvision import transforms


def build_transforms(image_size: int = 32, augment: bool = False) -> transforms.Compose:
    """Return a transform pipeline with optional augmentations."""

    pipeline: List[transforms.Compose | transforms.Normalize | transforms.RandomHorizontalFlip] = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if augment:
        pipeline.insert(1, transforms.RandomHorizontalFlip())
        pipeline.insert(2, transforms.ColorJitter(brightness=0.1, contrast=0.1))
    pipeline.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(pipeline)


def build_augmentation_pipeline(
    preset: Literal["baseline", "classic", "heavy"] = "baseline",
    image_size: int = 224,
) -> transforms.Compose:
    """Build an augmentation pipeline for classification training.
    
    Args:
        preset: Augmentation strength
            - baseline: minimal (resize + normalize only)
            - classic: standard (+ horizontal flip)
            - heavy: aggressive (+ color jitter, rotation, crop)
        image_size: Target image size for resize/crop
    
    Returns:
        Composed transform pipeline
    """
    if preset == "baseline":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    elif preset == "classic":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    elif preset == "heavy":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    else:
        raise ValueError(f"Unknown preset: {preset}. Choose from: baseline, classic, heavy")

