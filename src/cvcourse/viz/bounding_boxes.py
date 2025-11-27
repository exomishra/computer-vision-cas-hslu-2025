"""Helpers for drawing bounding boxes on tensors."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes as _tv_draw_boxes


def draw_bounding_boxes_on_image(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
) -> None:
    """Render bounding boxes on ``image`` and display the result."""

    annotated = _tv_draw_boxes(
        (image * 255).byte(),
        boxes,
        labels=labels,
        colors=colors or "cyan",
        width=10,
    )
    plt.figure(figsize=(5, 5))
    plt.imshow(annotated.permute(1, 2, 0))
    plt.axis("off")
    plt.show()


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1_max = max(box1[0].item(), box2[0].item())
    y1_max = max(box1[1].item(), box2[1].item())
    x2_min = min(box1[2].item(), box2[2].item())
    y2_min = min(box1[3].item(), box2[3].item())

    inter_w = max(0.0, x2_min - x1_max)
    inter_h = max(0.0, y2_min - y1_max)
    inter_area = inter_w * inter_h

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area.item() if union_area > 0 else 0.0


def collect_predictions(
    model: torch.nn.Module,
    images: List[torch.Tensor],
    device: torch.device,
    score_threshold: float = 0.5,
) -> List[dict]:
    """Run model inference and filter predictions by confidence threshold."""
    model.eval()
    with torch.no_grad():
        outputs = model([img.to(device) for img in images])
    
    filtered = []
    for out in outputs:
        mask = out["scores"] > score_threshold
        filtered.append({
            "boxes": out["boxes"][mask].cpu(),
            "labels": out["labels"][mask].cpu(),
            "scores": out["scores"][mask].cpu(),
        })
    return filtered


def compute_mean_iou(
    predictions: List[dict],
    ground_truths: List[torch.Tensor],
) -> float:
    """Compute mean IoU between predicted boxes and ground truth boxes.
    
    Args:
        predictions: List of dicts with 'boxes' key containing predicted boxes.
        ground_truths: List of ground truth box tensors [N, 4].
    """
    ious = []
    for pred, gt in zip(predictions, ground_truths):
        if len(pred["boxes"]) == 0 or len(gt) == 0:
            continue
        for pred_box in pred["boxes"]:
            for gt_box in gt:
                ious.append(box_iou(pred_box, gt_box))
    return sum(ious) / len(ious) if ious else 0.0


def plot_detection_grid(
    images: List[torch.Tensor],
    predictions: List[dict],
    coco_labels: dict,
    max_images: int = 4,
) -> None:
    """Display a grid of images with detection overlays.
    
    Args:
        images: List of image tensors [C, H, W].
        predictions: List of detection dicts with 'boxes', 'labels', 'scores'.
        coco_labels: Dict mapping label IDs to class names.
        max_images: Maximum number of images to display.
    """
    n = min(len(images), max_images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    
    for idx, (img, pred, ax) in enumerate(zip(images[:n], predictions[:n], axes)):
        labels = [
            f"{coco_labels.get(lbl.item(), '?')} {score:.2f}"
            for lbl, score in zip(pred["labels"], pred["scores"])
        ]
        annotated = _tv_draw_boxes(
            (img * 255).byte(),
            pred["boxes"],
            labels=labels,
            colors="cyan",
            width=2,
        )
        ax.imshow(annotated.permute(1, 2, 0))
        ax.axis("off")
        ax.set_title(f"Image {idx + 1}")
    
    plt.tight_layout()
    plt.show()
