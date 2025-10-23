''' All configuration for the disturbance detection model '''

class Config:
    """Main configuration class for the disturbance detection pipeline"""
    
    def __init__(self):
        # ============== DATA PATHS ==============
        # Input data
        self.csv_path = '/home/ubuntu/work/saved_data/landsat_disturbance_detection/clean_1D_U_Net/deep_disturbance/Data/Training/forest_dataset_timesync_alltiles_landsatbands_indices3_classes_ed5_calibration.csv'
        self.csv_separator = ';'
        
        # Split file (for reproducible train/val/test splits)
        self.split_path = "/home/ubuntu/work/saved_data/landsat_disturbance_detection/clean_1D_U_Net/deep_disturbance/Data/new_class2_v5_seed42_uids_w5_v5_results.npz"
        
        # Model save paths
        self.model_save_path = "/home/ubuntu/work/saved_data/landsat_disturbance_detection/clean_1D_U_Net/deep_disturbance/disturbance_detection/1dunet_w5_last_bands_indices_v5.pth"
        self.checkpoint_path = "best_model.pt"
        
        # Normalization statistics (saved during training, loaded during inference)
        self.min_train_path = "/home/ubuntu/work/saved_data/landsat_disturbance_detection/clean_1D_U_Net/deep_disturbance/disturbance_detection/min_train_w5_back_bands_indices_v5_results.npy"
        self.max_train_path = "/home/ubuntu/work/saved_data/landsat_disturbance_detection/clean_1D_U_Net/deep_disturbance/disturbance_detection/max_train_w5_back_bands_indices_v5_results.npy"
        
        # ============== DATA PREPARATION ==============
        self.window_size = 5
        self.max_window = 5
        self.batch_size = 64
        self.seed = 42
        
        # Feature selection: "bands", "indices", "bands_indices", "prev8"
        self.features_mode = "bands"
        
        # Target position for supervision: "last", "center", "second_last", "idx:<int>"
        self.target_mode = "last"
        
        # DataLoader settings
        self.num_workers = 4
        self.pin_memory = False
        self.persistent_workers = False
        
        # ============== MODEL ARCHITECTURE ==============
        self.base_channels = 16
        self.dropout_rate = 0.2
        self.temporal_dropout_rate = 0.2
        
        # Normalization: 'bn' (BatchNorm), 'ln' (LayerNorm via GroupNorm), 'gn8' (GroupNorm)
        # Use 'ln' for small batch sizes (<16), 'bn' for larger batches
        self.norm_type = 'ln'
        
        # Kernel sizes for multi-scale convolutions
        self.kernel_sizes_small = (1, 3, 5)
        self.kernel_sizes_big = (3, 5, 7)
        
        # ============== TRAINING ==============
        self.num_epochs = 30
        self.learning_rate = 1e-4
        self.weight_decay = 1e-2  # for AdamW
        
        # Optimizer settings (AdamW)
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        
        # Gradient clipping
        self.grad_clip = 1.0
        
        # Model selection criterion: "auprc" or "loss"
        self.select_by = "auprc"
        
        # ============== LOSS FUNCTION ==============
        # Focal Loss parameters for handling class imbalance
        self.focal_alpha = 0.5  # Weight for positive class (>0.5 focuses on positives)
        self.focal_gamma = 1.9   # Focusing parameter (2-4 for strong focusing on hard examples)
        self.loss_reduction = 'none'
        
        # ============== FEATURE SETS ==============
        self.FEATURE_SETS = {
            "bands": ["BLU", "GRN", "RED", "NIR", "SW1", "SW2"],
            "indices": ["NBR", "NDVI", "TCB", "TCG", "TCW", "DIn"],
            "bands_indices": ["BLU", "GRN", "RED", "NIR", "SW1", "SW2", 
                            "NBR", "NDVI", "TCB", "TCG", "TCW", "DIn"],
            "prev8": ["RED", "SW1", "SW2", "NBR", "NDVI", "TCG", "TCW", "DIn"],
        }
    
    def get_features(self):
        """Get the list of features based on features_mode"""
        return self.FEATURE_SETS[self.features_mode]
    
    def get_num_features(self):
        """Get the number of features"""
        return len(self.get_features())
    
    def __repr__(self):
        """Nice string representation for debugging"""
        return (f"Config(features_mode='{self.features_mode}', "
                f"num_features={self.get_num_features()}, "
                f"window_size={self.window_size}, "
                f"batch_size={self.batch_size}, "
                f"num_epochs={self.num_epochs})")