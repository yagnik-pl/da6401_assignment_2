"""VGG11 encoder and classifier."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from common import DEFAULT_IMAGE_SIZE
from .layers import CustomDropout


class SingleConvBlock(nn.Module):
    """A VGG-style block with one 3x3 convolution."""

    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DoubleConvBlock(nn.Module):
    """A VGG-style block with two stacked 3x3 convolutions."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, use_batch_norm: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels) if use_batch_norm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class VGG11Encoder(nn.Module):
    """VGG11-style convolutional backbone with skip-feature support."""

    def __init__(self, in_channels: int = 3, use_batch_norm: bool = True):
        super().__init__()
        self.input_size = DEFAULT_IMAGE_SIZE
        self.output_channels = 512

        self.block1 = SingleConvBlock(in_channels, 64, use_batch_norm=use_batch_norm)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2 = SingleConvBlock(64, 128, use_batch_norm=use_batch_norm)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3 = DoubleConvBlock(128, 256, 256, use_batch_norm=use_batch_norm)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4 = DoubleConvBlock(256, 512, 512, use_batch_norm=use_batch_norm)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block5 = DoubleConvBlock(512, 512, 512, use_batch_norm=use_batch_norm)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        features: Dict[str, torch.Tensor] = {}

        x = self.block1(x)
        features["enc1"] = x
        x = self.pool1(x)

        x = self.block2(x)
        features["enc2"] = x
        x = self.pool2(x)

        x = self.block3(x)
        features["enc3"] = x
        x = self.pool3(x)

        x = self.block4(x)
        features["enc4"] = x
        x = self.pool4(x)

        x = self.block5(x)
        features["enc5"] = x
        x = self.pool5(x)

        if return_features:
            return x, features
        return x


class ClassificationHead(nn.Module):
    """Standard VGG classifier head with BatchNorm and custom dropout."""

    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5, use_batch_norm: bool = True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        return self.classifier(x)


class VGG11(nn.Module):
    """Full VGG11 classifier composed of the encoder and a classification head."""

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batch_norm=use_batch_norm)
        self.head = ClassificationHead(num_classes=num_classes, dropout_p=dropout_p, use_batch_norm=use_batch_norm)

        self.avgpool = self.head.avgpool
        self.classifier = self.head.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.head(x)
