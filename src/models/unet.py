"""U-Net semantic segmentation model with a ResNet34 encoder."""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34

LOGGER = logging.getLogger("mangal_drishti.models.unet")


class DecoderBlock(nn.Module):
    """Upsample + skip concatenation + two 3x3 Conv-BN-ReLU layers."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class UNetResNet34(nn.Module):
    """U-Net architecture with ResNet34 backbone returning [B, num_classes, H, W] logits."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_loaded = False

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            backbone = resnet34(weights=weights)
            self.pretrained_loaded = pretrained
        except Exception as exc:
            LOGGER.warning("Could not load ResNet34 ImageNet weights (%s); using random init.", exc)
            backbone = resnet34(weights=None)

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64, H/2
        self.pool0 = backbone.maxpool
        self.enc1 = backbone.layer1  # 64,  H/4
        self.enc2 = backbone.layer2  # 128, H/8
        self.enc3 = backbone.layer3  # 256, H/16
        self.enc4 = backbone.layer4  # 512, H/32

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.dec0 = DecoderBlock(64, 0, 32)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        s0 = self.enc0(x)
        s1 = self.enc1(self.pool0(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.enc4(s3)

        d = self.dec4(b, s3)
        d = self.dec3(d, s2)
        d = self.dec2(d, s1)
        d = self.dec1(d, s0)
        d = self.dec0(d)

        logits = self.head(d)
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits
