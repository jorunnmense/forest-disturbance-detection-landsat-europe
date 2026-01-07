"""
Neural network architectures for forest disturbance detection.
Contains U-Net model, multi-kernel convolutions, attention, and focal loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import TemporalSelfAttention, MultiKernelConv1d, TemporalDropout, make_norm_layer


class UNet_1D_W3(nn.Module):
    """1-level U-Net variant of SmallUNet1D for very short inputs (T≈3)."""
    def __init__(self, in_channels, base=16, p_drop=0.2, norm='ln', kernel_sizes_small=(1,3,5)):
        super().__init__()

        # Encoder
        self.enc1 = MultiKernelConv1d(in_channels, base, p_drop=p_drop,
                                      kernel_sizes=kernel_sizes_small, norm=norm)  # (B, len(kernel_sizes)*base, T)
        self.pool1 = nn.AvgPool1d(2, ceil_mode=True)

        # Bottleneck - FIX: Calculate based on actual kernel count
        bottleneck_ch = len(kernel_sizes_small) * base  # <-- CHANGED from 3 * base


        self.bn_conv = nn.Sequential(
            nn.Conv1d(bottleneck_ch, bottleneck_ch, 1),
            make_norm_layer(norm, bottleneck_ch),
            nn.ReLU(inplace=True),
        )
        self.attn = TemporalSelfAttention(embed_dim=bottleneck_ch)

        # Decoder (only one level) - FIX: Update channel count here too
        decoder_in_ch = 2 * bottleneck_ch  # concat of bottleneck and encoder features


        self.dec1_reduce = nn.Sequential(
            nn.Conv1d(decoder_in_ch, bottleneck_ch, 1),  # <-- CHANGED from 2*3*base to decoder_in_ch
            make_norm_layer(norm, bottleneck_ch),
            nn.ReLU(inplace=True),
        )

        # Head
        self.out_conv = nn.Conv1d(bottleneck_ch, 1, 1)  # <-- CHANGED from 3*base to bottleneck_ch

    def forward(self, x):
        x1f = self.enc1(x)
        x1f = x1f.flip(dims=[2])
        x1 = self.pool1(x1f)
        x1 = x1.flip(dims=[2])

        xb = self.bn_conv(x1)
        xb = self.attn(xb)

        # Upsample to match encoder resolution
        y = F.interpolate(xb, size=x1f.size(2), mode='linear', align_corners=False)
        y = torch.cat([y, x1f], dim=1)
        y = self.dec1_reduce(y)

        out = self.out_conv(y).squeeze(1)
        return out
