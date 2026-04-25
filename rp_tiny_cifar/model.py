"""Mutable model file for Tiny CIFAR-10 Model Golf.

The autonomous evolver should edit this file only.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class DWBlock(nn.Module):
    """Small depthwise-separable residual block."""

    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            ConvBNAct(channels, hidden, kernel_size=1),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class Model(nn.Module):
    """Tiny depthwise residual CNN baseline.

    Input:  [B, 3, 32, 32]
    Output: [B, 10]
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 24, kernel_size=3, stride=1),
            ConvBNAct(24, 32, kernel_size=3, stride=2),
        )
        self.body = nn.Sequential(
            DWBlock(32),
            ConvBNAct(32, 48, kernel_size=3, stride=2),
            DWBlock(48),
            DWBlock(48),
            ConvBNAct(48, 64, kernel_size=3, stride=2),
            DWBlock(64),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.05),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.head(x)


def describe_model() -> str:
    return "tiny depthwise residual CNN baseline"
