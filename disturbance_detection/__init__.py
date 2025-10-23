"""
Disturbance Detection Package
==============================

A modular PyTorch implementation for forest disturbance detection using 1D U-Net 
with multi-kernel convolutions on Landsat time series data.

Main components:
- config: Configuration management
- data_preprocessing: Data loading, windowing, normalization
- models: Neural network architectures (U-Net, attention, multi-kernel conv)
- losses: Loss functions (Focal Loss)
- training: Training loops (single position and full supervision)
- evaluation: Metrics, threshold selection, visualization
- utils: Helper functions (seed setting, split inspection)
"""

__version__ = "1.0.0"
__author__ = "Jorunn Anna Mense, Alba Viana-Soto"

# ============== Core Configuration ==============
from .config import Config

# ============== Data Processing ==============
from .data_preprocessing import (
    prepare_data,
    make_or_load_uid_splits
)

# ============== Models ==============
from .models import (
    SmallUNet1D,
    MultiKernelConv1d,
    TemporalSelfAttention,
    TemporalDropout
)

# ============== Loss Functions ==============
from .losses import FocalLoss

# ============== Training ==============
from .training import (
    train_model,
    train_and_select_best,
    make_target_mask,
    safe_to_device
)

# Optional: full supervision training
try:
    from .training_full_supervision import (
        train_model_full_supervision,
        train_and_select_best_full_supervision
    )
except ImportError:
    pass  # Full supervision module is optional

# ============== Evaluation ==============
from .evaluation import (
    collect_probs,
    pick_threshold_by_f1,
    eval_at_threshold,
    final_eval_with_val_threshold,
    count_supervised_positives,
    print_classification_report,
    plot_history,
    plot_precision_recall_curve,
    plot_f1_vs_threshold,
    plot_confusion_matrix
)

# ============== Utilities ==============
from .utils import (
    set_seed,
    print_split_balances
)

# ============== Convenience Exports ==============
__all__ = [
    # Config
    'Config',
    
    # Data
    'prepare_data',
    'make_or_load_uid_splits',
    
    # Models
    'SmallUNet1D',
    'MultiKernelConv1d',
    'TemporalSelfAttention',
    'TemporalDropout',
    
    # Loss
    'FocalLoss',
    
    # Training
    'train_model',
    'train_and_select_best',
    'make_target_mask',
    'safe_to_device',
    
    # Training (full supervision)
    'train_model_full_supervision',
    'train_and_select_best_full_supervision',
    
    # Evaluation
    'collect_probs',
    'pick_threshold_by_f1',
    'eval_at_threshold',
    'final_eval_with_val_threshold',
    'count_supervised_positives',
    'print_classification_report',
    'plot_history',
    'plot_precision_recall_curve',
    'plot_f1_vs_threshold',
    'plot_confusion_matrix',
    
    # Utils
    'set_seed',
    'print_split_balances',
]