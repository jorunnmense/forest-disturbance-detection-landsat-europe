"""
Easy model selection.
"""
from .models import SmallUNet1D, MultiKernelConv1d, TemporalSelfAttention, TemporalDropout
from .TempCNN import TemporalCNN
from .model_30 import MediumUNet1D


def get_model(config):
    """
    Factory function to create model based on config.
    
    Args:
        config: Config object with model_name and other parameters
        
    Returns:
        PyTorch model ready for training
    """
    num_features = config.get_num_features()
    
    if config.model_name == "SmallUNet1D":
        model = SmallUNet1D(
            in_channels=num_features,
            base =config.base_channels,
            tdrop_rate=config.temporal_dropout_rate,
            kernel_sizes_small=config.kernel_sizes_small,
            kernel_sizes_big=config.kernel_sizes_big,
            norm=config.norm_type,
            p_drop=config.dropout_rate
        )
    
    elif config.model_name == "TemporalCNN":
        model = TemporalCNN(
            input_channels=num_features,
            hidden_dim=config.base_channels * 4,  # 64 for base_channels=16
            output_dim=1,
            dropout=config.dropout_rate
        )

    elif config.model_name == "UNet30":
        model = SmallUNet1D(
            in_channels=num_features,
            base =config.base_channels,
            tdrop_rate=config.temporal_dropout_rate,
            kernel_sizes_small=config.kernel_sizes_input_30_small,
            kernel_sizes_big=config.kernel_sizes_input_30_big,
        )

    elif config.model_name == "MediumUNet1D":
        model = MediumUNet1D(
            in_channels=num_features,
            base = 12,
            tdrop_rate=config.temporal_dropout_rate,
            kernel_sizes_small=config.kernel_sizes_input_30_small,
            kernel_sizes_big=config.kernel_sizes_input_30_big,
        )
    
    else:
        raise ValueError(f"Unknown model: {config.model_name}. "
                        f"Available: 'SmallUNet1D', 'TemporalCNN'")
    
    return model