import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCNN(nn.Module):
    def __init__(self, input_channels, hidden_dim=64, output_dim=1, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(hidden_dim)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(hidden_dim)

        self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                 # x: (B, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)                 # (B, 1, T)
        return x.squeeze(1)               # (B, T)