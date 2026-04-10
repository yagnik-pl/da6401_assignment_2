"""Shared helpers for Assignment 2."""

from __future__ import annotations

import os
import random
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    """Create a directory if it does not exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def resolve_path(path: str, base_dir: Optional[str] = None) -> str:
    """Resolve a possibly-relative path against the project root or a base directory."""
    if os.path.isabs(path):
        return path
    anchor = base_dir or os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(anchor, path))


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    """Normalize plain and wrapped checkpoints to a state_dict."""
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Checkpoint must be a state_dict or a dict containing 'state_dict'.")


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
    """Load a checkpoint from disk and return its state_dict."""
    checkpoint = torch.load(path, map_location=map_location)
    return extract_state_dict(checkpoint)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    """Save a standard checkpoint payload."""
    payload: Dict[str, object] = {"state_dict": model.state_dict()}
    if epoch is not None:
        payload["epoch"] = epoch
    if best_metric is not None:
        payload["best_metric"] = float(best_metric)
    if extra:
        payload.update(extra)

    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    torch.save(payload, path)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [x_center, y_center, width, height] boxes to corner format."""
    centers = boxes[..., :2]
    sizes = boxes[..., 2:].clamp_min(0.0)
    half_sizes = sizes / 2.0
    top_left = centers - half_sizes
    bottom_right = centers + half_sizes
    return torch.cat([top_left, bottom_right], dim=-1)


def box_iou_xywh(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute IoU for batches of xywh boxes."""
    pred_xyxy = xywh_to_xyxy(pred_boxes)
    target_xyxy = xywh_to_xyxy(target_boxes)

    top_left = torch.maximum(pred_xyxy[..., :2], target_xyxy[..., :2])
    bottom_right = torch.minimum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    intersection_wh = (bottom_right - top_left).clamp_min(0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    pred_wh = (pred_xyxy[..., 2:] - pred_xyxy[..., :2]).clamp_min(0.0)
    target_wh = (target_xyxy[..., 2:] - target_xyxy[..., :2]).clamp_min(0.0)
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    target_area = target_wh[..., 0] * target_wh[..., 1]
    union = pred_area + target_area - intersection

    return intersection / (union + eps)


def dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute mean multiclass Dice score."""
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dim=dims)
    denominator = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice_per_class = (2.0 * intersection + eps) / (denominator + eps)
    return dice_per_class.mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for multiclass segmentation."""
    return 1.0 - dice_score(logits, targets, eps=eps)


def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute pixel accuracy for segmentation."""
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean()


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor [C, H, W] to uint8 HWC format."""
    image = image_tensor.detach().cpu().float().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * np.asarray(IMAGENET_STD, dtype=np.float32) + np.asarray(IMAGENET_MEAN, dtype=np.float32)
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Map trimap labels {0,1,2} to RGB colors."""
    palette = np.asarray(
        [
            [25, 25, 25],
            [46, 204, 113],
            [52, 152, 219],
        ],
        dtype=np.uint8,
    )
    mask = np.asarray(mask, dtype=np.int64)
    mask = np.clip(mask, 0, len(palette) - 1)
    return palette[mask]


def blend_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Overlay a segmentation mask on top of an RGB image."""
    color_mask = mask_to_color(mask)
    blended = image.astype(np.float32) * (1.0 - alpha) + color_mask.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_boxes(
    image: np.ndarray,
    pred_box_xywh: Optional[Iterable[float]] = None,
    target_box_xywh: Optional[Iterable[float]] = None,
    text_lines: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Draw predicted and target boxes on an RGB image."""
    canvas = Image.fromarray(image.astype(np.uint8))
    drawer = ImageDraw.Draw(canvas)

    if target_box_xywh is not None:
        target_box = torch.tensor(list(target_box_xywh), dtype=torch.float32).unsqueeze(0)
        x1, y1, x2, y2 = xywh_to_xyxy(target_box)[0].tolist()
        drawer.rectangle([x1, y1, x2, y2], outline=(46, 204, 113), width=3)

    if pred_box_xywh is not None:
        pred_box = torch.tensor(list(pred_box_xywh), dtype=torch.float32).unsqueeze(0)
        x1, y1, x2, y2 = xywh_to_xyxy(pred_box)[0].tolist()
        drawer.rectangle([x1, y1, x2, y2], outline=(231, 76, 60), width=3)

    if text_lines is not None:
        x_offset = 6
        y_offset = 6
        for line in text_lines:
            drawer.rectangle([x_offset - 2, y_offset - 2, x_offset + 6 * len(line) + 4, y_offset + 14], fill=(0, 0, 0))
            drawer.text((x_offset, y_offset), line, fill=(255, 255, 255))
            y_offset += 16

    return np.asarray(canvas)
