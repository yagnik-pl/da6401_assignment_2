"""Unified multi-task model."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from common import load_checkpoint, resolve_path
from .classification import VGG11Classifier
from .localization import LocalizationHead, VGG11Localizer
from .segmentation import UNetDecoder, VGG11UNet
from .vgg11 import ClassificationHead, VGG11Encoder


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

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

        self.encoder = VGG11Encoder(in_channels=in_channels, use_batch_norm=use_batch_norm)
        self.classification_head = ClassificationHead(
            num_classes=num_breeds,
            dropout_p=dropout_p,
            use_batch_norm=use_batch_norm,
        )
        self.localization_head = LocalizationHead(
            dropout_p=dropout_p,
            use_batch_norm=use_batch_norm,
        )
        self.segmentation_decoder = UNetDecoder(
            num_classes=seg_classes,
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
        backbone_loaded = False

        classifier_model = VGG11Classifier(
            num_classes=num_breeds,
            in_channels=in_channels,
            dropout_p=dropout_p,
            use_batch_norm=use_batch_norm,
        )
        if self._safe_load(classifier_model, classifier_path):
            self.encoder.load_state_dict(classifier_model.encoder.state_dict(), strict=True)
            self.classification_head.load_state_dict(classifier_model.head.state_dict(), strict=True)
            backbone_loaded = True

        localizer_model = VGG11Localizer(
            in_channels=in_channels,
            dropout_p=dropout_p,
            use_batch_norm=use_batch_norm,
        )
        if self._safe_load(localizer_model, localizer_path):
            if not backbone_loaded:
                self.encoder.load_state_dict(localizer_model.encoder.state_dict(), strict=True)
                backbone_loaded = True
            self.localization_head.load_state_dict(localizer_model.head.state_dict(), strict=True)

        unet_model = VGG11UNet(
            num_classes=seg_classes,
            in_channels=in_channels,
            dropout_p=max(dropout_p * 0.5, 0.1),
            use_batch_norm=use_batch_norm,
        )
        if self._safe_load(unet_model, unet_path):
            if not backbone_loaded:
                self.encoder.load_state_dict(unet_model.encoder.state_dict(), strict=True)
            self.segmentation_decoder.load_state_dict(unet_model.decoder.state_dict(), strict=True)

    def forward(self, x: torch.Tensor):
        bottleneck, features = self.encoder(x, return_features=True)
        return {
            "classification": self.classification_head(bottleneck),
            "localization": self.localization_head(bottleneck),
            "segmentation": self.segmentation_decoder(bottleneck, features),
        }
