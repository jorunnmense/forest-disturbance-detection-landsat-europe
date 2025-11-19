import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMultiScaleUNet(nn.Module):
    """
    Novel architecture combining TempCNN's simplicity with U-Net's multi-scale features.
    Key innovation: Multi-scale temporal convolutions with selective skip connections.
    """
    def __init__(self, in_channels, base_dim=64, scales=[1, 3, 5], dropout=0.4):
        super().__init__()
        
        # Multi-scale temporal feature extraction (like TempCNN but parallel)
        self.temporal_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, base_dim, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(base_dim),
                nn.ReLU(inplace=True)
            ) for k in scales
        ])
        
        # Temporal attention for scale fusion (FIXED: preserve channels)
        self.scale_attention = nn.Sequential(
            nn.Conv1d(len(scales) * base_dim, len(scales) * base_dim, 1),  # Keep same channels
            nn.Sigmoid()
        )
        
        # Lightweight encoder-decoder with learned skip connections
        self.encoder = nn.Sequential(
            nn.Conv1d(len(scales) * base_dim, base_dim * 2, 3, padding=1),  # padding=1 preserves time
            nn.BatchNorm1d(base_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout)
        )
        
        # Adaptive skip connection (learnable)
        self.skip_gate = nn.Sequential(
            nn.Conv1d(len(scales) * base_dim, base_dim * 2, 1),
            nn.Sigmoid()
        )
        
        # Output (FIXED: preserve temporal dimension)
        self.classifier = nn.Sequential(
            nn.Conv1d(base_dim * 2, base_dim, 3, padding=1),  # padding=1 preserves time
            nn.BatchNorm1d(base_dim),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),
            nn.Conv1d(base_dim, 1, 1)  # Final output: (B, 1, T)
        )
    
    def forward(self, x):
        # Multi-scale feature extraction
        multi_scale = torch.cat([branch(x) for branch in self.temporal_branches], dim=1)
        
        # Scale attention (FIXED: now channels match)
        attention = self.scale_attention(multi_scale)
        attended_features = multi_scale * attention
        
        # Encoder
        encoded = self.encoder(attended_features)
        
        # Adaptive skip connection
        skip = self.skip_gate(multi_scale) * encoded
        
        # Output - preserve temporal dimension
        output = self.classifier(encoded + skip)  # (B, 1, T)
        return output.squeeze(1)  # (B, T) - same length as input