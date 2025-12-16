"""
Neural network architectures for forest disturbance detection.
Contains U-Net model, multi-kernel convolutions, attention, and focal loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from . components import TemporalSelfAttention, MultiKernelConv1d, TemporalDropout


class UNet_1D_W5to7(nn.Module):
    """
    2-level 1D U-Net that works for tiny/odd time series lengths.
    Uses MultiKernelConv1d blocks (multi-scale convolutions with BN/LN+ReLU+Dropout).
    
    Architecture:
    - Encoder: 2 levels (MKConv -> AvgPool with ceil_mode=True)
    - Bottleneck: 1x1 Conv -> Normalization -> ReLU + temporal self-attention
    - Decoder: interpolate up -> concat skip -> 1x1 Conv (reduce channels)
    - Head: 1x1 Conv -> logits per timestep
    
    Args:
        in_channels: Number of input features
        base: Base number of channels (all layers are multiples of this)
        p_drop: Dropout probability for convolutions
        norm: Normalization type ('bn', 'ln', or 'gnK')
        kernel_sizes_small: Kernel sizes for small time series (e.g., (1,3,5))
        tdrop_rate: Temporal dropout rate (dropout over time dimension)
    """
    def __init__(self, in_channels, base=16, p_drop=0.2,
                 norm='ln',
                 kernel_sizes_small=(1,3,5),
                 kernel_sizes_big=(3,5,7),
                 tdrop_rate=0.2):
        super().__init__()
        
        # Temporal dropout at input
        #self.tdrop = TemporalDropout(tdrop_rate)

        # Choose kernel sizes (can be made conditional on expected time series length)
        #ks1 = kernel_sizes_small
        #ks2 = kernel_sizes_small

        # ---- Encoder ----
        # Level 1: in_channels -> base*len(ks1) channels after concat
        self.enc1 = MultiKernelConv1d(in_channels, base, p_drop=p_drop, 
                                      kernel_sizes=kernel_sizes_small, norm=norm)  # -> (B, 3*base, T)
        self.pool1 = nn.AvgPool1d(2, ceil_mode=True)

        # Level 2: 3*base -> 2*base*len(ks2) channels after concat
        self.enc2 = MultiKernelConv1d(3*base, 2*base, p_drop=p_drop, 
                                      kernel_sizes=kernel_sizes_small, norm=norm)  # -> (B, 6*base, T/2)
        
        self.pool2 = nn.AvgPool1d(2, ceil_mode=True)

        # ---- Bottleneck ----
        bottleneck_ch = 4*base
        
        # Build normalization layer based on norm type
        if norm == 'ln':
            norm_layer = nn.GroupNorm(1, bottleneck_ch)
        elif norm.startswith('gn'):
            g = int(norm[2:]) if norm[2:].isdigit() else 8
            norm_layer = nn.GroupNorm(g, bottleneck_ch)
        else:  # 'bn'
            norm_layer = nn.BatchNorm1d(bottleneck_ch)
        
        self.bn_conv = nn.Sequential(
            nn.Conv1d(6*base, bottleneck_ch, 1),
            norm_layer,
            nn.ReLU(inplace=True),
        )
        self.attn = TemporalSelfAttention(embed_dim=bottleneck_ch)

        # ---- Decoder ----
        # Up to enc2 feature length
        if norm == 'ln':
            norm_layer2 = nn.GroupNorm(1, 6*base)
        elif norm.startswith('gn'):
            g = int(norm[2:]) if norm[2:].isdigit() else 8
            norm_layer2 = nn.GroupNorm(g, 6*base)
        else:
            norm_layer2 = nn.BatchNorm1d(6*base)
            
        self.dec2_reduce = nn.Sequential(
            nn.Conv1d(bottleneck_ch + 6*base, 6*base, 1),
            norm_layer2,
            nn.ReLU(inplace=True),
        )
        
        # Up to enc1 feature length
        if norm == 'ln':
            norm_layer3 = nn.GroupNorm(1, 3*base)
        elif norm.startswith('gn'):
            g = int(norm[2:]) if norm[2:].isdigit() else 8
            norm_layer3 = nn.GroupNorm(g, 3*base)
        else:
            norm_layer3 = nn.BatchNorm1d(3*base)
            
        self.dec1_reduce = nn.Sequential(
            nn.Conv1d(6*base + 3*base, 3*base, 1),
            norm_layer3,
            nn.ReLU(inplace=True),
        )

        # ---- Head ----
        self.out_conv = nn.Conv1d(3*base, 1, 1)

    def forward(self, x):
        # Apply temporal dropout at input
        #x = self.tdrop(x)
        
        # Encoder
        x1f = self.enc1(x)                 # (B, 3*base, T1)
        # flip the order of the time dimension
        #x1f = x1f.flip(dims=[2])
        x1  = self.pool1(x1f)              # (B, 3*base, T1p)
        #x1 = x1.flip(dims=[2])

        x2f = self.enc2(x1)                # (B, 6*base, T2)
        #x2f = x2f.flip(dims=[2])
        x2  = self.pool2(x2f)              # (B, 6*base, T2p)
        #x2 = x2.flip(dims=[2])

        # Bottleneck
        xb = self.bn_conv(x2)              # (B, 4*base, T2p)
        xb = self.attn(xb)                 # (B, 4*base, T2p) with attention

        # Decoder stage 2: upsample to x2f length
        y = F.interpolate(xb, size=x2f.size(2), mode='linear', align_corners=False)
        y = torch.cat([y, x2f], dim=1)     # (B, 4*base+6*base, T2)
        y = self.dec2_reduce(y)            # (B, 6*base, T2)

        # Decoder stage 1: upsample to x1f length
        y = F.interpolate(y, size=x1f.size(2), mode='linear', align_corners=False)
        y = torch.cat([y, x1f], dim=1)     # (B, 6*base+3*base, T1)
        y = self.dec1_reduce(y)            # (B, 3*base, T1)

        # Output
        out = self.out_conv(y).squeeze(1)  # (B, T1)
        return out