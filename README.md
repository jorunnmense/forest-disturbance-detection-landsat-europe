# Forest Disturbance Detection with 1D U-Net

Python Version: 3.12.3


PyTorch implementation for detecting forest disturbances from Landsat time series data using 1D U-Net architectures.

## Overview

This codebase implements 1D U-Net models for forest disturbance detection on Landsat time series spanning approximately 34 years across continental Europe. The framework supports different window sizes and feature combinations (Landsat spectral bands and vegetation indices), with comparisons against a modified TemporalCNN (Perbet et al., Pelletier et al.).

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

## Authors

Jorunn Anna Mense, Alba Viana-Soto

