"""Genuine DeepLabV3+ semantic segmentation model with a ResNet50 encoder."""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

LOGGER = logging.getLogger("mangal_drishti.models.deeplabv3plus")


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool(x)
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: Tuple[int, int, int] = (6, 12, 18),
    ) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = ASPPConv(in_channels, out_channels, atrous_rates[0])
        self.branch3 = ASPPConv(in_channels, out_channels, atrous_rates[1])
        self.branch4 = ASPPConv(in_channels, out_channels, atrous_rates[2])
        self.branch5 = ASPPPooling(in_channels, out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(
            torch.cat(
                [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x), self.branch5(x)],
                dim=1,
            )
        )


class DeepLabV3PlusResNet50(nn.Module):
    """DeepLabV3+ (ASPP + low-level layer1 fusion decoder) with ResNet50 backbone."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_loaded = False

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        try:
            backbone = resnet50(weights=weights, replace_stride_with_dilation=[False, False, True])
            self.pretrained_loaded = pretrained
        except Exception as exc:
            LOGGER.warning("Could not load ResNet50 ImageNet weights (%s); using random init.", exc)
            backbone = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 256 ch, stride 4 (H/4)
        self.layer2 = backbone.layer2  # 512 ch, stride 8
        self.layer3 = backbone.layer3  # 1024 ch, stride 16
        self.layer4 = backbone.layer4  # 2048 ch, stride 16 (dilated)

        self.aspp = ASPP(in_channels=2048, out_channels=256, atrous_rates=(6, 12, 18))
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        low_level = self.layer1(self.stem(x))
        high_level = self.layer4(self.layer3(self.layer2(low_level)))

        aspp_up = F.interpolate(
            self.aspp(high_level), size=low_level.shape[2:], mode="bilinear", align_corners=False
        )
        fused = torch.cat([aspp_up, self.low_level_proj(low_level)], dim=1)
        logits = self.decoder(fused)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
