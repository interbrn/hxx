"""
跨分辨率 RGB/红外特征适配与模态门控模块。

这是本毕设新增的核心模块：
1. MultiReceptiveFieldAdapter 用 1x1 投影、深度可分离卷积和不同膨胀率
   处理 RGB 与低分辨率红外特征之间的语义差异；
2. ModalityGate 学习每个位置、每个通道更应该相信 RGB 还是红外；
3. CrossResolutionFusion 按配置组合简单平均、特征适配和门控融合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiReceptiveFieldAdapter(nn.Module):
    """Project a modality feature and aggregate local context at two receptive fields."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 4)
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        self.dw1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        identity = self.project(x)
        context = self.mix(torch.cat((self.dw1(identity), self.dw2(identity)), dim=1))
        return identity + self.gate(context) * context


class ModalityGate(nn.Module):
    """Learn per-channel RGB/IR weights with optional spatial gating."""

    def __init__(self, channels, spatial=False):
        super().__init__()
        hidden = max(8, channels // 4)
        self.spatial = spatial
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels * 2, 1),
        )
        if spatial:
            self.spatial_gate = nn.Conv2d(channels * 2, 2, 3, padding=1)

    def forward(self, rgb, ir):
        pair = torch.cat((rgb, ir), dim=1)
        channel_logits = self.channel_gate(pair).view(pair.shape[0], 2, rgb.shape[1], 1, 1)
        channel_weights = channel_logits.softmax(dim=1)
        if self.spatial:
            spatial_weights = self.spatial_gate(pair).softmax(dim=1).unsqueeze(2)
            weights = channel_weights * spatial_weights
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            weights = channel_weights
        return weights[:, 0] * rgb + weights[:, 1] * ir


class CrossResolutionFusion(nn.Module):
    """Configurable simple/adapter/gated RGB-IR fusion for one feature stage."""

    def __init__(self, channels, use_adapter=False, use_gate=False, spatial_gate=False):
        super().__init__()
        self.rgb_adapter = MultiReceptiveFieldAdapter(channels) if use_adapter else nn.Identity()
        self.ir_adapter = MultiReceptiveFieldAdapter(channels) if use_adapter else nn.Identity()
        self.gate = ModalityGate(channels, spatial=spatial_gate) if use_gate else None

    def forward(self, rgb, ir):
        rgb = self.rgb_adapter(rgb)
        ir = self.ir_adapter(ir)
        if self.gate is not None:
            return self.gate(rgb, ir)
        return 0.5 * (rgb + ir)


class DirectConcatFusion(nn.Module):
    """Baseline fusion: concatenate RGB/IR channels, then project back to one width."""

    def __init__(self, channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

    def forward(self, rgb, ir):
        """Join two aligned feature maps along channels and restore the expected width."""
        return self.project(torch.cat((rgb, ir), dim=1))
