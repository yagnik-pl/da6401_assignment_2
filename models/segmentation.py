"""Segmentation model."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .layers import CustomDropout
from .vgg11 import VGG11Encoder


class DecoderRefineBlock(nn.Module):
    """Two-convolution refinement block used after skip concatenation."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0, use_batch_norm: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = CustomDropout(dropout_p) if dropout_p > 0.0 else nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    """A transposed-convolution upsampling block with encoder skip fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = DecoderRefineBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            dropout_p=dropout_p,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            target_h, target_w = x.shape[-2:]
            skip = skip[:, :, :target_h, :target_w]
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)


class UNetDecoder(nn.Module):
    """Decoder with progressive channel reduction — slim file ~94 MB."""

    def __init__(self, num_classes: int = 3, dropout_p: float = 0.1, use_batch_norm: bool = True):
        super().__init__()
        # bottleneck=512, enc5=512 → 512
        self.up5 = UpBlock(512, 512, 512, dropout_p=dropout_p, use_batch_norm=use_batch_norm)
        # 512 + enc4=512 → 384
        self.up4 = UpBlock(512, 512, 384, dropout_p=dropout_p, use_batch_norm=use_batch_norm)
        # 384 + enc3=256 → 128
        self.up3 = UpBlock(384, 256, 128, dropout_p=dropout_p, use_batch_norm=use_batch_norm)
        # 128 + enc2=128 → 64
        self.up2 = UpBlock(128, 128,  64, dropout_p=dropout_p, use_batch_norm=use_batch_norm)
        # 64  + enc1=64  → 32
        self.up1 = UpBlock( 64,  64,  32, dropout_p=dropout_p, use_batch_norm=use_batch_norm)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, bottleneck: torch.Tensor, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.up5(bottleneck, features["enc5"])
        x = self.up4(x, features["enc4"])
        x = self.up3(x, features["enc3"])
        x = self.up2(x, features["enc2"])
        x = self.up1(x, features["enc1"])
        return self.head(x)


class VGG11UNet(nn.Module):
    """U-Net style segmentation network."""

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batch_norm=use_batch_norm)
        self.decoder = UNetDecoder(num_classes=num_classes, dropout_p=dropout_p, use_batch_norm=use_batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, features = self.encoder(x, return_features=True)
        return self.decoder(bottleneck, features)


__all__ = ["UNetDecoder", "VGG11UNet"]