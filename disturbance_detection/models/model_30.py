"""
Neural network architectures for forest disturbance detection.
Contains U-Net model, multi-kernel convolutions, attention, and focal loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDropout(nn.Module):
    """
    Dropout over the temporal dimension to make model robust to gaps in time series.
    Drops entire timesteps rather than individual features.
    """
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        if not self.training or self.dropout_rate == 0.0:
            return x
        B, C, T = x.shape
        # Create mask: (B, T) with probability (1 - dropout_rate)
        mask = (torch.rand(B, T, device=x.device) > self.dropout_rate).float().unsqueeze(1)
        return x * mask


class MultiKernelConv1d(nn.Module):
    """
    Multi-scale convolutional block. Applies several 1D convolutions in parallel 
    with different kernel sizes and concatenates their outputs.
    Captures temporal patterns at multiple scales (e.g., sharp vs gradual changes).
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels per kernel
        p_drop: Dropout probability
        kernel_sizes: Tuple of kernel sizes to use (e.g., (1,3,5) or (3,5,7))
        norm: Normalization type - 'bn' (BatchNorm), 'ln' (LayerNorm via GroupNorm), 'gn8' (GroupNorm)
    """
    def __init__(self, in_channels, out_channels, p_drop=0.2,
                 kernel_sizes=(1,3,5), norm='bn'):
        super().__init__()
        self.branches = nn.ModuleList()
        
        for k in kernel_sizes:
            layers = [nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2)]
            
            # Add normalization layer
            if norm == 'bn':
                layers += [nn.BatchNorm1d(out_channels)]
            elif norm == 'ln':
                layers += [nn.GroupNorm(1, out_channels)]     # LayerNorm-like
            elif norm.startswith('gn'):
                g = int(norm[2:]) if norm[2:].isdigit() else 8
                layers += [nn.GroupNorm(g, out_channels)]
            else:
                raise ValueError("norm must be 'bn', 'ln', or 'gnK'")
            
            layers += [nn.ReLU(inplace=True)]
            self.branches.append(nn.Sequential(*layers))
        
        self.dropout = nn.Dropout1d(p_drop)

    def forward(self, x):
        # Concatenate all kernel outputs along channel dimension
        y = torch.cat([b(x) for b in self.branches], dim=1)  # (B, len(kernel_sizes)*out_channels, T)
        return self.dropout(y)


class TemporalSelfAttention(nn.Module):
    """
    Self-attention mechanism over the temporal dimension.
    Allows the model to focus on relevant timesteps.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        # x: (batch, channels, time)
        x = x.permute(0, 2, 1)  # -> (batch, time, channels)
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Scaled dot-product attention
        attn_weights = torch.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        attended = attn_weights @ V

        return attended.permute(0, 2, 1)  # -> (batch, channels, time)


class MediumUNet1D(nn.Module):
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
        x1  = self.pool1(x1f)

        x2f = self.enc2(x1)
        x2  = self.pool2(x2f)

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
