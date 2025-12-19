"""
Neural network architectures for forest disturbance detection.
Contains U-Net model, multi-kernel convolutions, attention, and focal loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import TemporalSelfAttention, MultiKernelConv1d, TemporalDropout


class UNet_1D_W30(nn.Module):
    """
    3-level 1D U-Net for longer time series (e.g., input length ~30).
    Uses MultiKernelConv1d blocks, temporal self-attention, and skip connections.
    """

    def __init__(self, in_channels, base=16, p_drop=0.2,
                 norm='ln',
                 kernel_sizes_small=(1,3,5),
                 kernel_sizes_big=(3,5,7),
                 tdrop_rate=0.2):
        super().__init__()

        # ---- Encoder ----
        self.enc1 = MultiKernelConv1d(in_channels, base, p_drop=p_drop,
                                      kernel_sizes=kernel_sizes_small, norm=norm)
        self.pool1 = nn.AvgPool1d(2, ceil_mode=True)

        self.enc2 = MultiKernelConv1d(3*base, 2*base, p_drop=p_drop,
                                      kernel_sizes=kernel_sizes_small, norm=norm)
        self.pool2 = nn.AvgPool1d(2, ceil_mode=True)

        self.enc3 = MultiKernelConv1d(6*base, 4*base, p_drop=p_drop,
                                      kernel_sizes=kernel_sizes_big, norm=norm)
        self.pool3 = nn.AvgPool1d(2, ceil_mode=True)

        # ---- Bottleneck ----
        bottleneck_in = 12*base
        bottleneck_ch = 8*base

        if norm == 'ln':
            norm_layer = nn.GroupNorm(1, bottleneck_ch)
        elif norm.startswith('gn'):
            g = int(norm[2:]) if norm[2:].isdigit() else 8
            norm_layer = nn.GroupNorm(g, bottleneck_ch)
        else:
            norm_layer = nn.BatchNorm1d(bottleneck_ch)

        self.bn_conv = nn.Sequential(
            nn.Conv1d(bottleneck_in, bottleneck_ch, 1),
            norm_layer,
            nn.ReLU(inplace=True),
        )
        self.attn = TemporalSelfAttention(embed_dim=bottleneck_ch)

        # ---- Decoder ----
        # Decoder 3 (up to enc3 feature length)
        if norm == 'ln':
            norm3 = nn.GroupNorm(1, 12*base)
        elif norm.startswith('gn'):
            g = int(norm[2:]) if norm[2:].isdigit() else 8
            norm3 = nn.GroupNorm(g, 12*base)
        else:
            norm3 = nn.BatchNorm1d(12*base)

        self.dec3_reduce = nn.Sequential(
            nn.Conv1d(bottleneck_ch + 12*base, 12*base, 1),
            norm3,
            nn.ReLU(inplace=True),
        )

        # Decoder 2 (up to enc2 feature length)
        if norm == 'ln':
            norm2 = nn.GroupNorm(1, 6*base)
        elif norm.startswith('gn'):
            g = int(norm[2:]) if norm[2:].isdigit() else 8
            norm2 = nn.GroupNorm(g, 6*base)
        else:
            norm2 = nn.BatchNorm1d(6*base)

        self.dec2_reduce = nn.Sequential(
            nn.Conv1d(12*base + 6*base, 6*base, 1),
            norm2,
            nn.ReLU(inplace=True),
        )

        # Decoder 1 (up to enc1 feature length)
        if norm == 'ln':
            norm1 = nn.GroupNorm(1, 3*base)
        elif norm.startswith('gn'):
            g = int(norm[2:]) if norm[2:].isdigit() else 8
            norm1 = nn.GroupNorm(g, 3*base)
        else:
            norm1 = nn.BatchNorm1d(3*base)

        self.dec1_reduce = nn.Sequential(
            nn.Conv1d(6*base + 3*base, 3*base, 1),
            norm1,
            nn.ReLU(inplace=True),
        )

        # ---- Head ----
        self.out_conv = nn.Conv1d(3*base, 1, 1)

    def forward(self, x):
        # Encoder
        x1f = self.enc1(x)
        x1f = x1f.flip(dims=[2])
        x1  = self.pool1(x1f)
        x1 = x1.flip(dims=[2])

        x2f = self.enc2(x1)
        x2f = x2f.flip(dims=[2])
        x2  = self.pool2(x2f)
        x2 = x2.flip(dims=[2])

        x3f = self.enc3(x2)
        x3  = self.pool3(x3f)

        # Bottleneck
        xb = self.bn_conv(x3)
        xb = self.attn(xb)

        # Decoder 3
        y = F.interpolate(xb, size=x3f.size(2), mode='linear', align_corners=False)
        y = torch.cat([y, x3f], dim=1)
        y = self.dec3_reduce(y)

        # Decoder 2
        y = F.interpolate(y, size=x2f.size(2), mode='linear', align_corners=False)
        y = torch.cat([y, x2f], dim=1)
        y = self.dec2_reduce(y)

        # Decoder 1
        y = F.interpolate(y, size=x1f.size(2), mode='linear', align_corners=False)
        y = torch.cat([y, x1f], dim=1)
        y = self.dec1_reduce(y)

        # Output
        out = self.out_conv(y).squeeze(1)
        return out
