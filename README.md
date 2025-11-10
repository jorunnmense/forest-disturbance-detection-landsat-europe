# Forest Disturbance Detection with 1D U-Net

Python Version: 3.12.3

## Overview
Detection of Disturbances in around 34 (?) years of Landsat Data with a 1D U-Net for continental Europe. Different Tests for different window sizes, tested against the TempCNN by Perbet et al. We further test the performance of different landsat bands as well as computed vegetational indices.

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```python
# Example usage code
```

## Project Structure

deep_disturbance/
│
├── README.md
├── changes_not_tested.md
├── best_model.pt
├── data_splits.npz
├── test_modular_training.ipynb
├── Train_multikernel_5_smallUnet_last.ipynb
│
├── disturbance_detection/                          # Main Python package
│   ├── __init__.py                                 # Package initialization & exports
│   ├── config.py                                   # Configuration management
│   ├── data_preprocessing.py                       # Data loading & windowing
│   ├── evaluation.py                               # Metrics & visualization
│   ├── training_full_supervision.py                # Training loops
│   ├── utils.py                                    # Helper functions (seed, etc.)
│   │
│   ├── min_train_w5_back_bands_indices_v5_results.npy
│   ├── max_train_w5_back_bands_indices_v5_results.npy
│   │
│   ├── models/                                     # Neural network architectures
│   │   ├── __init__.py
│   │   ├── models.py                               # SmallUNet1D, MultiKernel, Attention
│   │   ├── model_30.py                             # MediumUNet1D
│   │   ├── model_3.py                              # Additional model variants
│   │   ├── TempCNN.py                              # TemporalCNN architecture
│   │   └── selection.py                            # get_model() factory function
│   │
│   └── loss_fcts/                                  # Loss functions
│       ├── __init__.py
│       └── losses.py                               # FocalLoss implementation
│
└── Data/                                           # Dataset directory
    ├── new_class2_v5_seed42_uids_w5_v5_results.npz
    │
    ├── Training/                                   # Training datasets (CSV files)
    ├── Validation/
    ├── Test/
    └── Final/

