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
- training full supervision: Training loops (single position and full supervision)
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
# ============== Models ==============
from .models import (
    UNet_1D_W5to7,
    MultiKernelConv1d,
    TemporalSelfAttention,
    TemporalDropout,
    TemporalCNN,
    get_model,
    UNet_1D_W5to7_MultiLevel_Attention,
    UNet_1D_W3,
    UNet_1D_W30,
    get_valid_kernel_sizes
)

# ============== Loss Functions ==============
from .loss_fcts import FocalLoss

# ============== Training ==============

# Optional: full supervision training
from .training_full_supervision import (
    train_epoch_full_supervision,
    validate_epoch_full_supervision,
    train_full_supervision_with_selection)


# ============== Evaluation ==============
from .evaluations import (
    collect_probs,
    pick_threshold_by_f1,
    eval_at_threshold,
    final_eval_with_val_threshold,
    count_supervised_positives,
    print_classification_report,
    plot_history,
    plot_precision_recall_curve,
    plot_f1_vs_threshold,
    plot_confusion_matrix,
    make_target_mask,
    safe_to_device
)

from .evaluations import (
    CaptumEvaluator
)

# ============== Utilities ==============
from .utils import (
    set_seed,
    print_split_balances
)

# ============== Test Cases ==============
# from .test_cases import test_case_1
# (basically just write the function name instead of test_case_1)

# ============== Convenience Exports ==============
__all__ = [
    # Config
    'Config',
    
    # Data
    'prepare_data',
    'make_or_load_uid_splits',
    
    # Models
    'UNet_1D_W5to7',
    'MultiKernelConv1d',
    'TemporalSelfAttention',
    'TemporalDropout',
    'TemporalCNN',
    'get_model',
    'UNet_1D_W30',
    'UNet_1D_W3',
    'UNet_1D_W5to7_MultiLevel_Attention',
    'get_valid_kernel_sizes',

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
    'CaptumEvaluator',
    'make_target_mask',
    'safe_to_device',
    'count_supervised_positives',
    'collect_probs',

    # Utils
    'set_seed',
    'print_split_balances',

    #Test cases
    # 'test_case_1',  
]