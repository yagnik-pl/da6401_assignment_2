"""Custom IoU loss."""

import torch
import torch.nn as nn

from common import box_iou_xywh


class IoULoss(nn.Module):
    """IoU loss for bounding box regression."""

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction '{reduction}'. Use 'none', 'mean', or 'sum'.")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        losses = 1.0 - box_iou_xywh(pred_boxes, target_boxes, eps=self.eps).clamp(0.0, 1.0)
        if self.reduction == "none":
            return losses
        if self.reduction == "sum":
            return losses.sum()
        return losses.mean()
