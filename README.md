# Deep Learning-Based Forest Disturbance Detection in Europe Using Landsat Time Series

Python Version: 3.12.3


PyTorch implementation for detecting forest disturbances from Landsat time series data using 1D U-Net architectures.

## Overview

This repository contains the PyTorch code used for forest disturbance detection from Landsat time series in continental Europe. The main models are 1D U-Net variants, and we also include a modified TempCNN baseline following Perbet et al. and Pelletier et al. The code supports different temporal window sizes and feature sets (spectral bands and vegetation indices) to compare model settings in a consistent way. It also includes data preprocessing, training, and evaluation utilities.

## Relation to Paper
This repository provides the implementation used in the study “Deep learning-based forest disturbance detection for Europe using Landsat time series” (Viana-Soto et al., 2026, Remote Sensing of Environment). It includes the 1D U-Net and modified TempCNN models used in the experiments, together with the corresponding preprocessing, training, and evaluation components required to reproduce the reported analyses.

[DOI](https://doi.org/10.1016/j.rse.2026.115670) / [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0034425726004402)


## Installation

pip install -r requirements.txt

## Usage

from disturbance_detection import Config, prepare_data, get_model, train_full_supervision_with_selection

config = Config()

config.window_size = 7

config.features_mode = "bands_indices"

train_loader, val_loader, test_loader = prepare_data(config)

model = get_model(config)

train_full_supervision_with_selection(config, model, train_loader, val_loader)

## Project Structure

- `disturbance_detection/`: Main package
  - `config.py`: Configuration management
  - `preprocessing.py`: Data loading and preprocessing
  - `models/`: Neural network architectures
  - `loss_fcts/`: Loss functions
  - `training.py`: Training loops
  - `evaluations.py`: Evaluation metrics

## Configuration

Model and training parameters are configured through the `Config` class, including data paths, model architecture, window size, feature selection, and training hyperparameters.

## Citation

If you use this repository, please cite:

Viana-Soto, A., Mense, J. A., Kowalski, K., Pauls, J., Gieseke, F., & Senf, C. (2026).  
*Deep learning-based forest disturbance detection for Europe using Landsat time series*.  
Remote Sensing of Environment, 347, 115670.  
https://doi.org/10.1016/j.rse.2026.115670

```bibtex
@article{viana-soto2026deep,
  title   = {Deep learning-based forest disturbance detection for Europe using Landsat time series},
  author  = {Viana-Soto, Alba and Mense, Jorunn Anna and Kowalski, Katja and Pauls, Jan and Gieseke, Fabian and Senf, Cornelius},
  journal = {Remote Sensing of Environment},
  volume  = {347},
  pages   = {115670},
  year    = {2026},
  doi     = {10.1016/j.rse.2026.115670}
}

## Authors

Jorunn Anna Mense, Alba Viana-Soto

