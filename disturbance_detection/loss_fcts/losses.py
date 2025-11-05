"""
Loss functions for forest disturbance detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in binary classification.
    
    Focal Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - alpha: Weight for positive class (>0.5 focuses on minority class)
    - gamma: Focusing parameter that reduces loss for well-classified examples
      - gama=0: equivalent to standard cross-entropy
      - gamma=1-2: moderate focusing
      - gamma=2-4: strong focusing for extreme imbalance
    
    Args:
        alpha: Weight for positive class (0 to 1)
        gamma: Focusing parameter (typically 0-5)
        reduction: 'none', 'mean', or 'sum'
    """
    def __init__(self, alpha=0.85, gamma=2.5, reduction='none'):
        super().__init__()
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (not probabilities), same shape as targets (e.g., B x T)
            targets: Binary labels {0,1} as float tensor
            
        Returns:
            Loss tensor (shape depends on reduction parameter)
        """
        # Binary cross-entropy with logits (numerically stable)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Probability of the true class
        pt = torch.exp(-bce)
        
        # Alpha weighting: alpha for class 1, (1-alpha) for class 0
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal loss formula
        loss = alpha_t * (1 - pt) ** self.gamma * bce

        if self.reduction == 'none':
            return loss                 # per-element (B x T)
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma_pos=2.0, gamma_neg=1.0, reduction='mean'):
        """
        Args:
            alpha: Weight for positive class
            gamma_pos: Focusing parameter for positive class (disturbances)
            gamma_neg: Focusing parameter for negative class (undisturbed)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs (B, T)
            targets: Ground truth labels (B, T)
        """
        probs = torch.sigmoid(logits)
        
        # Positive class loss (disturbances)
        pos_loss = -self.alpha * ((1 - probs) ** self.gamma_pos) * torch.log(probs + 1e-8)
        
        # Negative class loss (undisturbed)
        neg_loss = -(1 - self.alpha) * (probs ** self.gamma_neg) * torch.log(1 - probs + 1e-8)
        
        # Combine based on targets
        loss = targets * pos_loss + (1 - targets) * neg_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss