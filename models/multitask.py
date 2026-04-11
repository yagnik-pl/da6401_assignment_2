"""Unified multi-task model."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from common import load_checkpoint, resolve_path
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Task-preserving multi-task wrapper built from the best single-task checkpoints."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
        dropout_p: float = 0.5,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.image_size = 224

        self.classifier_model = VGG11Classifier(
            num_classes=num_breeds,
            in_channels=in_channels,
            dropout_p=dropout_p,
            use_batch_norm=use_batch_norm,
        )
        self.localizer_model = VGG11Localizer(
            in_channels=in_channels,
            dropout_p=dropout_p,
            image_size=self.image_size,
            use_batch_norm=use_batch_norm,
        )
        self.segmenter_model = VGG11UNet(
            num_classes=seg_classes,
            in_channels=in_channels,
            dropout_p=max(dropout_p * 0.5, 0.1),
            use_batch_norm=use_batch_norm,
        )

        self._initialize_from_checkpoints(
            classifier_path=classifier_path,
            localizer_path=localizer_path,
            unet_path=unet_path,
            num_breeds=num_breeds,
            seg_classes=seg_classes,
            in_channels=in_channels,
            dropout_p=dropout_p,
            use_batch_norm=use_batch_norm,
        )

    def _resolve_checkpoint(self, path: str) -> str:
        return resolve_path(path, base_dir=self.project_root)

    def _safe_load(self, model: nn.Module, checkpoint_path: str) -> bool:
        resolved = self._resolve_checkpoint(checkpoint_path)
        if not os.path.exists(resolved):
            return False
        state_dict = load_checkpoint(resolved, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        return True

    def _initialize_from_checkpoints(
        self,
        classifier_path: str,
        localizer_path: str,
        unet_path: str,
        num_breeds: int,
        seg_classes: int,
        in_channels: int,
        dropout_p: float,
        use_batch_norm: bool,
    ) -> None:
        self._safe_load(self.classifier_model, classifier_path)
        self._safe_load(self.localizer_model, localizer_path)
        self._safe_load(self.segmenter_model, unet_path)

    def _predict_localization(self, x: torch.Tensor) -> torch.Tensor:
        box_logits = self.localizer_model(x)
        x_flipped = torch.flip(x, dims=[3])
        flipped_boxes = self.localizer_model(x_flipped)
        flipped_boxes = flipped_boxes.clone()
        flipped_boxes[:, 0] = float(self.image_size) - flipped_boxes[:, 0]
        return 0.5 * (box_logits + flipped_boxes)

    def _predict_segmentation(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.segmenter_model(x)
        x_flipped = torch.flip(x, dims=[3])
        flipped_logits = torch.flip(self.segmenter_model(x_flipped), dims=[3])
        return 0.5 * (logits + flipped_logits)

    def _predict_classification(self, x: torch.Tensor, boxes_xywh: torch.Tensor) -> torch.Tensor:
        logits = self.classifier_model(x)
        flip_logits = self.classifier_model(torch.flip(x, dims=[3]))
        return 0.5 * (logits + flip_logits)

    def forward(self, x: torch.Tensor):
        localization = self._predict_localization(x)
        segmentation = self._predict_segmentation(x)
        classification = self._predict_classification(x, localization)
        return {
            "classification": classification,
            "localization": localization,
            "segmentation": segmentation,
        }
