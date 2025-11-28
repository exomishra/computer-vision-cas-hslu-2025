"""Segmentation helpers built on torchvision and SAM."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torchvision.models.segmentation import (
    FCN_ResNet50_Weights,
    fcn_resnet50,
)

from ..utils.device import get_device, to_device
from ..utils.paths import get_data_dir


def load_segmentation_model(allow_gpu: bool = True) -> torch.nn.Module:
    """Return a pretrained FCN-ResNet50 model for semantic segmentation."""

    device = get_device(allow_gpu=allow_gpu)
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model.eval()
    return to_device(model, device)


def get_sam_model(
    variant: str = "vit_b",
    allow_gpu: bool = True,
    checkpoint_path: Path | None = None,
):
    """Load SAM model and return a SamPredictor for interactive segmentation.
    
    Args:
        variant: Model variant ('vit_b', 'vit_l', 'vit_h').
        allow_gpu: Whether to use GPU if available.
        checkpoint_path: Path to checkpoint file. If None, looks in data/pretrained/.
    """
    from segment_anything import SamPredictor, sam_model_registry
    
    device = get_device(allow_gpu=allow_gpu)
    
    if checkpoint_path is None:
        ckpt_name = f"sam_{variant}_01ec64.pth"
        checkpoint_path = get_data_dir() / "pretrained" / ckpt_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at {checkpoint_path}. "
                f"Download from https://github.com/facebookresearch/segment-anything"
            )
    
    sam = sam_model_registry[variant](checkpoint=str(checkpoint_path))
    sam = to_device(sam, device)
    return SamPredictor(sam)


def segment_with_prompts(
    predictor,
    image: torch.Tensor,
    point_coords: torch.Tensor | None = None,
    point_labels: torch.Tensor | None = None,
    box: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run SAM segmentation with point or box prompts.
    
    Args:
        predictor: Initialized SamPredictor.
        image: Image tensor [C, H, W] in range [0, 1].
        point_coords: Point prompts as [N, 2] array of (x, y) coordinates.
        point_labels: Point labels as [N] array (1=foreground, 0=background).
        box: Box prompt as [4] array [x1, y1, x2, y2].
    
    Returns:
        Tuple of (masks, scores, logits) where masks is [N, H, W].
    """
    # Convert to numpy format expected by SAM
    image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    predictor.set_image(image_np)
    
    # Convert prompts to numpy
    point_coords_np = point_coords.cpu().numpy() if point_coords is not None else None
    point_labels_np = point_labels.cpu().numpy() if point_labels is not None else None
    box_np = box.cpu().numpy() if box is not None else None
    
    masks, scores, logits = predictor.predict(
        point_coords=point_coords_np,
        point_labels=point_labels_np,
        box=box_np,
        multimask_output=True,
    )
    
    return (
        torch.from_numpy(masks),
        torch.from_numpy(scores),
        torch.from_numpy(logits),
    )


def compute_stability_score(
    logits: torch.Tensor, mask_threshold: float = 0.0, threshold_offset: float = 1.0
) -> torch.Tensor:
    """Compute stability score for mask predictions.
    
    Estimates how much the mask would change if the threshold were shifted slightly.
    Higher scores indicate more confident, stable predictions.
    
    Args:
        logits: Raw mask logits [N, H, W].
        mask_threshold: Threshold for converting logits to binary masks.
        threshold_offset: Amount to shift threshold for stability estimation.
    """
    intersections = (
        (logits > mask_threshold + threshold_offset).sum(dim=(-1, -2)).float()
    )
    unions = (logits > mask_threshold - threshold_offset).sum(dim=(-1, -2)).float()
    return intersections / unions


def select_best_mask(
    masks: torch.Tensor,
    scores: torch.Tensor,
    logits: torch.Tensor,
    use_stability: bool = True,
) -> Tuple[torch.Tensor, float]:
    """Select the best mask from multiple predictions.
    
    Args:
        masks: Predicted masks [N, H, W].
        scores: Confidence scores [N].
        logits: Raw logits [N, H, W].
        use_stability: If True, combine confidence with stability score.
    
    Returns:
        Tuple of (best_mask, best_score).
    """
    if use_stability:
        stability_scores = compute_stability_score(logits)
        combined_scores = scores * stability_scores
        best_idx = combined_scores.argmax()
    else:
        best_idx = scores.argmax()
    
    return masks[best_idx], scores[best_idx].item()

