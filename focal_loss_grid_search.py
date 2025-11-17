#!/usr/bin/env python3
"""
Focal Loss Grid Search - Background Runner
==========================================

Converts the test_modular_training.ipynb notebook into a grid search script
for focal loss hyperparameters (alpha and gamma values).

Results are saved to a comprehensive text file with all metrics and configurations.

Usage:
    screen -S focal_loss_grid_search
    python run_focal_loss_grid_search.py
    # Detach with Ctrl+A, D
"""

import os
import sys
import time
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = "/home/ubuntu/work/saved_data/landsat_disturbance_detection/clean_1D_U_Net/deep_disturbance"
sys.path.insert(0, project_root)

# Import from disturbance detection package
from disturbance_detection import (
    Config, set_seed,
    prepare_data, print_split_balances, make_or_load_uid_splits,
    SmallUNet1D, FocalLoss,
    safe_to_device,
    final_eval_with_val_threshold,
    count_supervised_positives,
    plot_history, plot_precision_recall_curve, 
    plot_f1_vs_threshold, print_classification_report,
    get_model,
    TemporalCNN,
    MediumUNet1D,
    TinyUNet1D,
    CaptumEvaluator
)

from disturbance_detection.training_full_supervision import train_full_supervision_with_selection

def setup_output_file():
    """Create timestamped output file for results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"/home/ubuntu/work/saved_data/landsat_disturbance_detection/clean_1D_U_Net/deep_disturbance/focal_loss_grid_search_results_{timestamp}.txt"
    return output_file

def log_message(message, output_file, print_to_console=True):
    """Log message to both file and console"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}] {message}"
    
    if print_to_console:
        print(formatted_message)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(formatted_message + '\n')
        f.flush()

def load_and_prepare_data(config, output_file):
    """Load dataset and prepare data loaders"""
    log_message("="*80, output_file)
    log_message("LOADING AND PREPARING DATA", output_file)
    log_message("="*80, output_file)
    
    # Load dataset
    df = pd.read_csv(config.csv_path, index_col=0, sep=config.csv_separator)
    log_message(f"Loaded {len(df)} samples", output_file)

    # Create disturbance labels
    df['class'] = df['class_level1'].apply(lambda x: 1 if x == 'disturbance' else 0)

    # Filter doubtful samples (from original notebook)
    df = df[~((df['class_level1'] == 'treed') & (df['NBR'] < 0.5))]
    df = df[~((df['class_level1'] == 'treed') & (df['NDVI'] < 0.5))]
    df = df[~((df['class_level1'] == 'non-treed') & (df['NBR'] < 0.5))]
    df = df[~((df['class_level1'] == 'non-treed') & (df['NDVI'] < 0.5))]

    log_message(f"After filtering: {len(df)} samples", output_file)
    
    class_dist = df['class_level1'].value_counts()
    log_message(f"Class distribution:", output_file)
    for class_name, count in class_dist.items():
        log_message(f"  {class_name}: {count}", output_file)

    # Prepare train/val/test splits and dataloaders
    train_loader, val_loader, test_loader, n_features, features = prepare_data(df, config)

    log_message(f"Data preparation complete!", output_file)
    log_message(f"Features used: {features}", output_file)
    log_message(f"Number of features: {n_features}", output_file)
    log_message(f"DataLoader sizes:", output_file)
    log_message(f"  Train batches: {len(train_loader)}", output_file)
    log_message(f"  Val batches: {len(val_loader)}", output_file)
    log_message(f"  Test batches: {len(test_loader)}", output_file)

    return train_loader, val_loader, test_loader, n_features, features

def check_class_balance(train_loader, val_loader, test_loader, config, device, output_file):
    """Check and log class balance information"""
    log_message("\n" + "="*60, output_file)
    log_message("CLASS BALANCE ANALYSIS", output_file)
    log_message("="*60, output_file)
    
    # Check class balance at supervised position
    tr_pos, tr_tot, tr_rate = count_supervised_positives(train_loader, config.target_mode, device)
    va_pos, va_tot, va_rate = count_supervised_positives(val_loader, config.target_mode, device)
    te_pos, te_tot, te_rate = count_supervised_positives(test_loader, config.target_mode, device)

    log_message(f"Class balance at target position ('{config.target_mode}'):", output_file)
    log_message(f"  Train: {tr_pos}/{tr_tot} ({100*tr_rate:.3f}% positive)", output_file)
    log_message(f"  Val:   {va_pos}/{va_tot} ({100*va_rate:.3f}% positive)", output_file)
    log_message(f"  Test:  {te_pos}/{te_tot} ({100*te_rate:.3f}% positive)", output_file)

def run_single_experiment(alpha, gamma, config, train_loader, val_loader, device, output_file):
    """Run a single training experiment with given alpha and gamma"""
    log_message(f"\n{'='*60}", output_file)
    log_message(f"EXPERIMENT: alpha={alpha}, gamma={gamma}", output_file)
    log_message(f"{'='*60}", output_file)
    
    start_time = time.time()
    
    # Update config
    config.focal_alpha = alpha
    config.focal_gamma = gamma

    # Initialize model
    model = get_model(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"Model: {config.model_name}", output_file)
    log_message(f"  Total parameters: {total_params:,}", output_file)
    log_message(f"  Trainable parameters: {trainable_params:,}", output_file)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps
    )

    # Initialize loss function
    loss_fn = FocalLoss(
        alpha=config.focal_alpha,
        gamma=config.focal_gamma,
        reduction=config.loss_reduction
    )

    log_message(f"Optimizer: AdamW (lr={config.learning_rate}, wd={config.weight_decay})", output_file)
    log_message(f"Loss: FocalLoss (alpha={config.focal_alpha}, gamma={config.focal_gamma})", output_file)
    log_message(f"Training for {config.num_epochs} epochs...", output_file)

    # Train with best model selection
    try:
        history, best_info = train_full_supervision_with_selection(
            model, train_loader, val_loader, optimizer, device, loss_fn, config
        )
        
        training_time = time.time() - start_time
        
        # Log results
        log_message(f"Training completed successfully!", output_file)
        log_message(f"Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)", output_file)
        log_message(f"Best epoch: {best_info['best_epoch']}", output_file)
        log_message(f"Best {best_info['select_by']}: {best_info['best_metric']:.4f}", output_file)
        log_message(f"Best F1 (target): {best_info['best_f1_target']:.4f}", output_file)
        
        # Log final epoch metrics
        final_epoch = len(history['train_loss']) - 1
        log_message(f"Final epoch metrics:", output_file)
        log_message(f"  Train Loss: {history['train_loss'][final_epoch]:.4f}", output_file)
        log_message(f"  Train F1 (all): {history['train_f1_all'][final_epoch]:.4f}", output_file)
        log_message(f"  Train F1 (target): {history['train_f1_target'][final_epoch]:.4f}", output_file)
        log_message(f"  Val Loss: {history['val_loss'][final_epoch]:.4f}", output_file)
        log_message(f"  Val F1 (all): {history['val_f1_all'][final_epoch]:.4f}", output_file)
        log_message(f"  Val F1 (target): {history['val_f1_target'][final_epoch]:.4f}", output_file)
        log_message(f"  Val AUPRC: {history['val_auprc'][final_epoch]:.4f}", output_file)
        
        return {
            'alpha': alpha,
            'gamma': gamma,
            'success': True,
            'training_time': training_time,
            'best_epoch': best_info['best_epoch'],
            'best_val_auprc': best_info['best_metric'],
            'best_f1_target': best_info['best_f1_target'],
            'final_train_loss': history['train_loss'][final_epoch],
            'final_val_loss': history['val_loss'][final_epoch],
            'final_val_auprc': history['val_auprc'][final_epoch],
            'final_val_f1_target': history['val_f1_target'][final_epoch],
            'history': history
        }
        
    except Exception as e:
        training_time = time.time() - start_time
        error_msg = str(e)
        log_message(f"Training FAILED after {training_time:.1f} seconds", output_file)
        log_message(f"Error: {error_msg}", output_file)
        
        return {
            'alpha': alpha,
            'gamma': gamma,
            'success': False,
            'training_time': training_time,
            'error': error_msg
        }

def run_grid_search(output_file):
    """Main grid search execution"""
    log_message("="*80, output_file)
    log_message("FOCAL LOSS HYPERPARAMETER GRID SEARCH", output_file)
    log_message("="*80, output_file)
    log_message(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", output_file)
    
    # Create configuration
    config = Config()
    
    # Grid search parameters - you can modify these
    alpha_values = [0.3, 0.4, 0.5,0.6, 0.7, 0.8, 0.9]  # Extended range
    gamma_values = [1.5, 1.7, 1.9, 2.0, 2.1, 2.2, 2.5, 3.0]  # Extended range
    
    # Training configuration
    config.num_epochs = 30  # Reduced for faster grid search
    config.target_mode = "last"
    config.features_mode = "bands"  # Using bands as in notebook
    config.model_name = "SmallUNet1D"  # As in notebook
    config.dropout_rate = 0.3
    
    # Set random seed for reproducibility
    set_seed(config.seed)

    # Check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}", output_file)
    
    # Log configuration
    log_message(f"Configuration:", output_file)
    log_message(f"  Model: {config.model_name}", output_file)
    log_message(f"  Features: {config.features_mode} ({config.get_num_features()} features)", output_file)
    log_message(f"  Window size: {config.window_size}", output_file)
    log_message(f"  Batch size: {config.batch_size}", output_file)
    log_message(f"  Epochs per experiment: {config.num_epochs}", output_file)
    log_message(f"  Learning rate: {config.learning_rate}", output_file)
    log_message(f"  Target mode: {config.target_mode}", output_file)
    log_message(f"  Selection criterion: {config.select_by}", output_file)
    
    log_message(f"\nGrid search parameters:", output_file)
    log_message(f"  Alpha values: {alpha_values}", output_file)
    log_message(f"  Gamma values: {gamma_values}", output_file)
    log_message(f"  Total experiments: {len(alpha_values) * len(gamma_values)}", output_file)
    
    # Load and prepare data
    train_loader, val_loader, test_loader, n_features, features = load_and_prepare_data(config, output_file)
    
    # Check class balance
    check_class_balance(train_loader, val_loader, test_loader, config, device, output_file)
    
    # Track results
    grid_results = []
    total_experiments = len(alpha_values) * len(gamma_values)
    experiment_count = 0
    
    log_message(f"\n{'='*80}", output_file)
    log_message("STARTING GRID SEARCH EXPERIMENTS", output_file)
    log_message(f"{'='*80}", output_file)
    
    grid_start_time = time.time()
    
    # Run grid search
    for alpha in alpha_values:
        for gamma in gamma_values:
            experiment_count += 1
            log_message(f"\n>>> EXPERIMENT {experiment_count}/{total_experiments} <<<", output_file)
            
            result = run_single_experiment(alpha, gamma, config, train_loader, val_loader, device, output_file)
            grid_results.append(result)
            
            # Log progress
            elapsed_time = time.time() - grid_start_time
            avg_time_per_exp = elapsed_time / experiment_count
            remaining_exp = total_experiments - experiment_count
            estimated_remaining = avg_time_per_exp * remaining_exp
            
            log_message(f"Progress: {experiment_count}/{total_experiments} completed", output_file)
            log_message(f"Elapsed time: {elapsed_time/60:.1f} minutes", output_file)
            log_message(f"Estimated remaining time: {estimated_remaining/60:.1f} minutes", output_file)
    
    total_time = time.time() - grid_start_time
    
    # Analyze and log results
    log_message(f"\n{'='*80}", output_file)
    log_message("GRID SEARCH RESULTS SUMMARY", output_file)
    log_message(f"{'='*80}", output_file)
    log_message(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)", output_file)
    log_message(f"Average time per experiment: {total_time/total_experiments/60:.1f} minutes", output_file)
    
    # Filter successful experiments
    successful_results = [r for r in grid_results if r['success']]
    failed_results = [r for r in grid_results if not r['success']]
    
    log_message(f"\nExperiment outcomes:", output_file)
    log_message(f"  Successful: {len(successful_results)}/{total_experiments}", output_file)
    log_message(f"  Failed: {len(failed_results)}/{total_experiments}", output_file)
    
    if failed_results:
        log_message(f"\nFailed experiments:", output_file)
        for result in failed_results:
            log_message(f"  alpha={result['alpha']}, gamma={result['gamma']}: {result['error']}", output_file)
    
    if successful_results:
        # Sort by best validation AUPRC
        successful_results.sort(key=lambda x: x['best_val_auprc'], reverse=True)
        
        log_message(f"\n{'='*60}", output_file)
        log_message("TOP 10 RESULTS (by validation AUPRC)", output_file)
        log_message(f"{'='*60}", output_file)
        log_message(f"{'Rank':<4} {'Alpha':<6} {'Gamma':<6} {'AUPRC':<7} {'F1':<7} {'Epoch':<5} {'Time(min)':<9}", output_file)
        log_message(f"{'-'*50}", output_file)
        
        for i, result in enumerate(successful_results[:10]):
            log_message(f"{i+1:<4} {result['alpha']:<6} {result['gamma']:<6} "
                       f"{result['best_val_auprc']:<7.4f} {result['best_f1_target']:<7.4f} "
                       f"{result['best_epoch']:<5} {result['training_time']/60:<9.1f}", output_file)
        
        # Best result details
        best_result = successful_results[0]
        log_message(f"\n{'='*60}", output_file)
        log_message("BEST CONFIGURATION DETAILS", output_file)
        log_message(f"{'='*60}", output_file)
        log_message(f"Alpha: {best_result['alpha']}", output_file)
        log_message(f"Gamma: {best_result['gamma']}", output_file)
        log_message(f"Best epoch: {best_result['best_epoch']}", output_file)
        log_message(f"Best validation AUPRC: {best_result['best_val_auprc']:.4f}", output_file)
        log_message(f"Best F1 (target): {best_result['best_f1_target']:.4f}", output_file)
        log_message(f"Final validation AUPRC: {best_result['final_val_auprc']:.4f}", output_file)
        log_message(f"Final validation F1 (target): {best_result['final_val_f1_target']:.4f}", output_file)
        log_message(f"Training time: {best_result['training_time']/60:.1f} minutes", output_file)
        
        # Summary statistics
        log_message(f"\n{'='*60}", output_file)
        log_message("SUMMARY STATISTICS", output_file)
        log_message(f"{'='*60}", output_file)
        
        auprcs = [r['best_val_auprc'] for r in successful_results]
        f1s = [r['best_f1_target'] for r in successful_results]
        times = [r['training_time'] for r in successful_results]
        
        log_message(f"Validation AUPRC statistics:", output_file)
        log_message(f"  Mean: {sum(auprcs)/len(auprcs):.4f}", output_file)
        log_message(f"  Min:  {min(auprcs):.4f}", output_file)
        log_message(f"  Max:  {max(auprcs):.4f}", output_file)
        log_message(f"  Std:  {(sum([(x-sum(auprcs)/len(auprcs))**2 for x in auprcs])/len(auprcs))**0.5:.4f}", output_file)
        
        log_message(f"\nF1 (target) statistics:", output_file)
        log_message(f"  Mean: {sum(f1s)/len(f1s):.4f}", output_file)
        log_message(f"  Min:  {min(f1s):.4f}", output_file)
        log_message(f"  Max:  {max(f1s):.4f}", output_file)
        log_message(f"  Std:  {(sum([(x-sum(f1s)/len(f1s))**2 for x in f1s])/len(f1s))**0.5:.4f}", output_file)
        
        log_message(f"\nTraining time statistics:", output_file)
        log_message(f"  Mean: {sum(times)/len(times)/60:.1f} minutes", output_file)
        log_message(f"  Min:  {min(times)/60:.1f} minutes", output_file)
        log_message(f"  Max:  {max(times)/60:.1f} minutes", output_file)
    
    log_message(f"\n{'='*80}", output_file)
    log_message("GRID SEARCH COMPLETED", output_file)
    log_message(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", output_file)
    log_message(f"Results saved to: {output_file}", output_file)
    log_message(f"{'='*80}", output_file)

def main():
    """Main execution function"""
    print("Focal Loss Grid Search - Starting...")
    
    # Setup output file
    output_file = setup_output_file()
    
    try:
        # Run the grid search
        run_grid_search(output_file)
        
        print(f"\nGrid search completed successfully!")
        print(f"Results saved to: {output_file}")
        return 0
        
    except Exception as e:
        error_msg = f"Grid search failed with error: {str(e)}"
        log_message(error_msg, output_file)
        print(error_msg)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)