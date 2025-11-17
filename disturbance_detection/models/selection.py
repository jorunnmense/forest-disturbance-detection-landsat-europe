"""
Easy model selection.
"""
from .UNet_1D_W3 import UNet_1D_W3
from .TempCNN import TemporalCNN
from .UNet_1D_W5to7 import UNet_1D_W5to7
from .UNet_1D_W5to7_MultiLevel_Attention import UNet_1D_W5to7_MultiLevel_Attention
from .UNet_1D_W30 import UNet_1D_W30
from .components import TemporalSelfAttention, MultiKernelConv1d, TemporalDropout

def get_model(config):
    """
    Factory function to create model based on config.
    
    Args:
        config: Config object with model_name and other parameters
        
    Returns:
        PyTorch model ready for training
    """
    num_features = config.get_num_features()
    
    if config.model_name == "UNet_1D_W5to7":
        model = UNet_1D_W5to7(
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

    elif config.model_name == "UNet_1D_W30":
        model = UNet_1D_W30(
            in_channels=num_features,
            base =config.base_channels,
            tdrop_rate=config.temporal_dropout_rate,
            kernel_sizes_small=config.kernel_sizes_input_30_small,
            kernel_sizes_big=config.kernel_sizes_input_30_big,
        )
    
    elif config.model_name == "UNet_1D_W3":
        model = UNet_1D_W3(
            in_channels=num_features,
            base =config.base_channels,
            p_drop=config.dropout_rate,
            norm=config.norm_type,
        kernel_sizes_small=config.kernel_sizes_small,
        )
    
    elif config.model_name == "UNet_1D_W5to7_MultiLevel_Attention":
        model = UNet_1D_W5to7_MultiLevel_Attention(
            in_channels=num_features,
            base =config.base_channels,
            p_drop=config.dropout_rate,
            norm=config.norm_type,
            kernel_sizes_small=config.kernel_sizes_small,
            kernel_sizes_big=config.kernel_sizes_big,
        )
    else:
        raise ValueError(f"Unknown model: {config.model_name}. "
                        f"Available: 'SmallUNet1D', 'TinyUNet1D', 'TemporalCNN', 'UNet30', 'MediumUNet1D'")
    
    return model

    '''elif config.model_name == "MediumUNet1D":
        model = MediumUNet1D(
            in_channels=num_features,
            base = 12,
            tdrop_rate=config.temporal_dropout_rate,
            kernel_sizes_small=config.kernel_sizes_input_30_small,
            kernel_sizes_big=config.kernel_sizes_input_30_big,
        )'''