import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


# ---------------------------
# Utilities
# ---------------------------

class TemporalDropout(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if not self.training or self.dropout_rate == 0.0:
            return x
        B, C, T = x.shape
        mask = (torch.rand(B, T, device=x.device) > self.dropout_rate).float()
        mask = mask.unsqueeze(1)
        return x * mask


class MultiKernelConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1,3,5]):
        super().__init__()
        n = len(kernel_sizes)
        base_ch = out_channels // n
        remainder = out_channels - base_ch * n
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            ch = base_ch + (1 if i < remainder else 0)
            self.convs.append(nn.Conv1d(in_channels, ch, k, padding=k//2))
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x_cat = torch.cat([conv(x) for conv in self.convs], dim=1)
        x_cat = self.norm(x_cat.permute(0,2,1)).permute(0,2,1)
        return F.relu(x_cat)


class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        x = x.permute(0,2,1)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = torch.softmax((Q @ K.transpose(-2,-1)) * self.scale, dim=-1)
        out = attn @ V
        return out.permute(0,2,1)


# ---------------------------
# Focal Loss
# ---------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=4.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = torch.where(targets==1, self.alpha, 1-self.alpha)
        loss = alpha_t * (1-pt)**self.gamma * BCE_loss
        return loss.mean() if self.reduction=='mean' else loss.sum()


# ---------------------------
# Fixed 5-year 1D U-Net
# ---------------------------

class UNet5Years(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, kernel_sizes=[1,3,5]):
        super().__init__()
        self.dropout = TemporalDropout(0.2)
        self.kernel_sizes = kernel_sizes

        # --- Downsampling ---
        self.conv1 = MultiKernelConv1d(in_channels, 16, kernel_sizes)   # (B,8,5) -> (B,16,5)
        self.pool1 = nn.AvgPool1d(2, ceil_mode=True)                     # 5 → 3

        # --- Bottleneck ---
        self.bottleneck = MultiKernelConv1d(16, 32, kernel_sizes)        # (B,16,3) -> (B,32,3)
        self.attention = TemporalSelfAttention(embed_dim=32)

        # --- Upsampling ---
        self.conv_up = MultiKernelConv1d(32 + 16, 16, kernel_sizes)      # skip + bottleneck

        # --- Final ---
        self.final_conv = nn.Conv1d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.dropout(x)

        # Down
        x1 = self.conv1(x)
        x_pool = self.pool1(x1)

        # Bottleneck
        x_b = self.bottleneck(x_pool)
        x_b = self.attention(x_b)

        # Upsample and merge skip connection
        x_up = F.interpolate(x_b, size=x1.size(2), mode='linear', align_corners=False)
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = self.conv_up(x_up)

        # Output
        out = self.final_conv(x_up)
        return out.squeeze(1)   # (B, 5)


# ---------------------------
# Test example
# ---------------------------

if __name__ == "__main__":
    x = torch.randn(2, 8, 5)   # (batch=2, channels=8, time=5)
    model = UNet5Years(in_channels=8)
    y = model(x)
    print("Output shape:", y.shape)  # should be [2, 5]
