"""Localization modules."""

import torch
import torch.nn as nn

from common import DEFAULT_IMAGE_SIZE
from .layers import CustomDropout
from .vgg11 import VGG11Encoder


class LocalizationHead(nn.Module):
    """Regression head that predicts xywh bounding boxes in image pixel space."""

    def __init__(self, dropout_p: float = 0.5, image_size: int = DEFAULT_IMAGE_SIZE, use_batch_norm: bool = True):
        super().__init__()
        self.image_size = float(image_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.regressor(x)
        return x * self.image_size


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(
        self,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        image_size: int = DEFAULT_IMAGE_SIZE,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batch_norm=use_batch_norm)
        self.head = LocalizationHead(
            dropout_p=dropout_p,
            image_size=image_size,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.head(x)


__all__ = ["LocalizationHead", "VGG11Localizer"]
