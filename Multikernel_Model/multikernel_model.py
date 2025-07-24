import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalDropout(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        # x shape: (B, C, T)
        if not self.training or self.dropout_rate == 0.0:
            return x

        B, C, T = x.shape
        mask = (torch.rand(B, T, device=x.device) > self.dropout_rate).float()
        mask = mask.unsqueeze(1)  # shape: (B, 1, T)
        return x * mask

class MultiKernelConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiKernelConv1d, self).__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(out_channels*3)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x_cat = torch.cat([x3, x5, x7], dim=1)  # Concatenate channel-wise
        x_cat = self.norm(x_cat.permute(0, 2, 1)).permute(0, 2, 1)
        return F.relu(x_cat)

class FocalLoss(nn.Module):
    def __init__(self, alpha = 1, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # weight for class 1 (disturbance)
        self.gamma = gamma  # focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute binary cross entropy loss without reduction
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Convert logits to probabilities
        pt = torch.exp(-BCE_loss)  # pt is the probability of correct prediction

        # Compute focal loss
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return loss.mean() if self.reduction == 'mean' else loss.sum()

class TemporalSelfAttention(nn.Module):
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

        attn_weights = torch.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        attended = attn_weights @ V

        return attended.permute(0, 2, 1)  # -> (batch, channels, time)
        

class AlbasUNet(nn.Module):
    def __init__(self):
        super(AlbasUNet, self).__init__()

        n_ch1 = 16
        n_ch2 = 32
        n_ch3 = 64
        n_ch_bottleneck = 128

        # Multi-kernel convs output 3x channels
        self.dropout = TemporalDropout(dropout_rate=0.2)
        self.conv1 = MultiKernelConv1d(in_channels=6, out_channels=n_ch1)  # Outputs 16*3=48 channels
        self.pool1 = nn.AvgPool1d(kernel_size=2)

        self.conv2 = MultiKernelConv1d(in_channels=48, out_channels=n_ch2)  # Outputs 32*3=96 channels
        self.pool2 = nn.AvgPool1d(kernel_size=2)

        self.conv3 = MultiKernelConv1d(in_channels=96, out_channels=n_ch3)  # Outputs 64*3=192 channels
        self.pool3 = nn.AvgPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(in_channels=192, out_channels=n_ch_bottleneck, kernel_size=1)
        self.attention = TemporalSelfAttention(embed_dim=n_ch_bottleneck)

        # Upsampling layers
        self.up3 = nn.ConvTranspose1d(in_channels=n_ch_bottleneck, out_channels=192, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose1d(in_channels=192*2, out_channels=96, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose1d(in_channels=96*2, out_channels=48, kernel_size=2, stride=2)

        self.out_conv = nn.Conv1d(in_channels=48*2, out_channels=1, kernel_size=1)
    
    def forward(self, x):
        x = self.dropout(x)
        x1_features = self.conv1(x)
        x1 = self.pool1(x1_features)

        x2_features = self.conv2(x1)
        x2 = self.pool2(x2_features)

        x3_features = self.conv3(x2)
        x3 = self.pool3(x3_features)

        x4 = F.relu(self.conv4(x3))  # Bottleneck features

        x4 = self.attention(x4)  # [batch, channels, time] in, returns same shape


        x_up3 = self.up3(x4)
        if x_up3.shape[2] != x3.shape[2]:
            x3 = F.pad(x3, (0, x_up3.shape[2] - x3.shape[2]))
        x_up2 = self.up2(torch.cat([x_up3, x3], dim=1))
        if x_up2.shape[2] != x2.shape[2]:
            x2 = F.pad(x2, (0, x_up2.shape[2] - x2.shape[2]))
        x_up1 = self.up1(torch.cat([x_up2, x2], dim=1))
        if x_up1.shape[2] != x1.shape[2]:
            x1 = F.pad(x1, (0, x_up1.shape[2] - x1.shape[2]))

        out = self.out_conv(torch.cat([x_up1, x1], dim=1))
        return out.squeeze(1)