"""
Easy model selection.
"""
from .models import SmallUNet1D, MultiKernelConv1d, TemporalSelfAttention, TemporalDropout
from .TempCNN import TemporalCNN


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
            norm=config.norm_type
        )
    
    elif config.model_name == "TemporalCNN":
        model = TemporalCNN(
            input_channels=num_features,
            hidden_dim=config.base_channels * 4,  # 64 for base_channels=16
            output_dim=1,
            dropout=config.dropout_rate
        )
    
    else:
        raise ValueError(f"Unknown model: {config.model_name}. "
                        f"Available: 'SmallUNet1D', 'TemporalCNN'")
    
    return model