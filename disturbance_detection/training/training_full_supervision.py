import torch
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, accuracy_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_target_position(config, sequence_length):
    """
    Get the target position based on the config.
    """
    if config.target_mode == "last":
        target = -1
    elif config.target_mode == "second_last":
        target = -2
    elif config.target_mode == "center":
        target = sequence_length // 2
    else:
        target = -1
    return target


def train_epoch_full_supervision(model, train_loader, optimizer, device, loss_fn, config):
    """
    Train one epoch on all timesteps (full supervision).
    Computes training F1 for all positions and target timestep.
    """
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for x, y, _ in train_loader:  # ignore mask
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        y_hat = model(x)  # (B, T)
        loss = loss_fn(y_hat, y).mean()  # dense supervision
        loss.backward()
        if hasattr(config, "grad_clip") and config.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        all_probs.append(torch.sigmoid(y_hat).detach().cpu())
        all_labels.append(y.detach().cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Flatten for overall F1
    preds_all = (all_probs.flatten() > config.classification_threshold).astype(int)
    train_f1_all = f1_score(all_labels.flatten(), preds_all, average='binary', zero_division=0)
    train_acc = accuracy_score(all_labels.flatten(), preds_all)
    train_precision = precision_score(all_labels.flatten(), preds_all, zero_division=0)
    train_recall = recall_score(all_labels.flatten(), preds_all, zero_division=0)

    # Target timestep F1
    #target_pos = get_target_position(config, all_labels.shape[1])
    target_pos = get_target_position(config, config.window_size)

    train_f1_target = f1_score(
        all_labels[:, target_pos],
        (all_probs[:, target_pos] > config.classification_threshold).astype(int),
        zero_division=0
    )

    train_precision_target = precision_score(all_labels[:, target_pos], (all_probs[:, target_pos] > config.classification_threshold).astype(int), zero_division=0)
    train_recall_target = recall_score(all_labels[:, target_pos], (all_probs[:, target_pos] > config.classification_threshold).astype(int), zero_division=0)

    avg_loss = total_loss / len(train_loader)
    return avg_loss, train_acc, train_f1_all, train_f1_target, train_precision, train_recall


def validate_epoch_full_supervision(model, val_loader, device, loss_fn, config):
    """
    Validate one epoch.
    Computes:
        - loss at target timestep
        - F1 for all positions
        - F1 at target timestep
        - AUPRC at target timestep
    """
    model.eval()
    val_loss = 0.0
    val_probs_all, val_labels_all = [], []
    val_probs_target, val_labels_target = [], []
    eps = 1e-8

    # Determine target timestep index from first batch
    #sample_y = next(iter(val_loader))[1]
    #T = sample_y.shape[1]
    target_pos = get_target_position(config, config.window_size)

    with torch.no_grad():
        for x, y, _ in val_loader:  # ignore mask
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            per_elem_loss = loss_fn(y_hat, y)
            val_loss += per_elem_loss.mean().item()  # average over all positions

            probs = torch.sigmoid(y_hat).cpu()
            val_probs_all.append(probs)
            val_labels_all.append(y.cpu())

            # Select target timestep directly
            val_probs_target.append(probs[:, target_pos])
            val_labels_target.append(y.cpu()[:, target_pos])

    val_loss /= len(val_loader)

    # Overall F1 across all positions
    all_probs_flat = torch.cat(val_probs_all).numpy().flatten()
    all_labels_flat = torch.cat(val_labels_all).numpy().flatten()
    val_f1_all = f1_score(all_labels_flat, (all_probs_flat > config.classification_threshold).astype(int),
                           average='binary', zero_division=0)
    val_recall_all = recall_score(all_labels_flat, (all_probs_flat > config.classification_threshold).astype(int), zero_division=0)                       
    val_precision_all = precision_score(all_labels_flat, (all_probs_flat > config.classification_threshold).astype(int), zero_division=0)

    # Target timestep metrics
    y_true_target = torch.cat(val_labels_target).numpy().astype(int) if val_labels_target else np.array([])
    y_prob_target = torch.cat(val_probs_target).numpy() if val_probs_target else np.array([])

    val_recall = recall_score(y_true_target, (y_prob_target > config.classification_threshold).astype(int), zero_division=0)
    val_precision = precision_score(y_true_target, (y_prob_target > config.classification_threshold).astype(int), zero_division=0)

    if y_true_target.size and len(np.unique(y_true_target)) == 2:
        val_auprc = average_precision_score(y_true_target, y_prob_target)
        precision, recall, thresholds = precision_recall_curve(y_true_target, y_prob_target)
        f1_curve = 2 * precision * recall / (precision + recall + eps)
        val_f1_target_best = float(np.nanmax(f1_curve[1:])) if f1_curve.size > 1 else 0.0
    else:
        val_auprc = np.nan
        val_f1_target_best = 0.0

    return val_loss, val_f1_all, val_f1_target_best, val_auprc, val_precision, val_recall



def train_full_supervision_with_selection(model, train_loader, val_loader, optimizer, device, loss_fn, config):
    """
    Full training loop with model selection based on target timestep AUPRC.
    """

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor= config.scheduler_factor, patience = config.scheduler_patience, verbose=True)
    best_metric = -np.inf
    best_epoch = -1
    best_f1_target = 0.0
    history = {
        "train_loss": [], "train_f1_all": [], "train_f1_target": [],
        "val_loss": [], "val_f1_all": [], "val_f1_target": [], "val_auprc": [],
        "train_precision": [], "train_recall": [],
        "val_precision": [], "val_recall": []
    }

    for epoch in range(config.num_epochs):
        train_loss, train_acc, train_f1_all, train_f1_target, train_precision, train_recall = train_epoch_full_supervision(
            model, train_loader, optimizer, device, loss_fn, config
        )

        val_loss, val_f1_all, val_f1_target, val_auprc, val_precision, val_recall = validate_epoch_full_supervision(
            model, val_loader, device, loss_fn, config
        )

         # Step scheduler based on AUPRC
        if not np.isnan(val_auprc):
            scheduler.step(val_auprc)
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        history["train_loss"].append(train_loss)
        history["train_precision"].append(train_precision)
        history["train_recall"].append(train_recall)
        history["train_f1_all"].append(train_f1_all)
        history["train_f1_target"].append(train_f1_target)
        history["val_loss"].append(val_loss)
        history["val_f1_all"].append(val_f1_all)
        history["val_f1_target"].append(val_f1_target)
        history["val_auprc"].append(val_auprc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)

        print(f"Epoch {epoch+1:02d} | TrainLoss {train_loss:.4f} | "
              f"TrainF1(all) {train_f1_all:.4f} | TrainF1(target) {train_f1_target:.4f} | TrainPrecision {train_precision:.4f} | TrainRecall {train_recall:.4f} | " 
              f"ValLoss {val_loss:.4f} | ValF1(all) {val_f1_all:.4f} | ValF1* {val_f1_target:.4f} | ValAUPRC {val_auprc:.4f} | ValPrecision {val_precision:.4f} | ValRecall {val_recall:.4f}")

        # Model selection based on AUPRC at target timestep
        '''current_metric = val_auprc if config.select_by == "auprc" else -val_loss
        is_better = (config.select_by == "auprc" and (val_auprc == val_auprc) and current_metric > best_metric) or \
                    (config.select_by == "loss"  and current_metric > best_metric)'''


        current_metric = val_auprc if config.select_by == "auprc" else -val_loss

        is_valid_metric = not np.isnan(current_metric)
        is_better = is_valid_metric and current_metric > best_metric

        if is_better:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_f1_target = val_f1_target
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_auprc": val_auprc
            }, config.checkpoint_path)

    summary = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_f1_target": best_f1_target,
        "ckpt_path": config.checkpoint_path,
        "select_by": config.select_by
    }
            
    return history, summary



def focal_loss_grid_search(
    train_loader, 
    val_loader, 
    config, 
    device,
    alpha_values=None,
    gamma_values=None,
    num_epochs=10,
    verbose=True
):
    """
    Grid search for optimal focal loss hyperparameters.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader  
        config: Configuration object
        device: PyTorch device
        alpha_values: List of alpha values to test (default: [0.3, 0.5, 0.7])
        gamma_values: List of gamma values to test (default: [1.5, 2.0, 2.5])
        num_epochs: Number of epochs per experiment (default: 10)
        verbose: Whether to print progress (default: True)
        
    Returns:
        dict: Results containing best parameters and all experiment results
    """
    import torch
    from ..models import get_model
    from ..loss_fcts import FocalLoss
    
    # Default parameter ranges
    if alpha_values is None:
        alpha_values = [0.3, 0.5, 0.7]
    if gamma_values is None:
        gamma_values = [1.5, 2.0, 2.5]
    
    # Store original config values
    original_alpha = config.focal_alpha
    original_gamma = config.focal_gamma
    original_epochs = config.num_epochs
    
    # Update config for grid search
    config.num_epochs = num_epochs
    
    # Track results
    grid_results = []
    best_result = None
    best_metric = 0.0
    
    if verbose:
        print("Starting Grid Search for Focal Loss Hyperparameters...")
        print("-" * 80)
        print(f"Testing {len(alpha_values)} alpha × {len(gamma_values)} gamma = {len(alpha_values) * len(gamma_values)} combinations")
        print(f"Alpha values: {alpha_values}")
        print(f"Gamma values: {gamma_values}")
        print(f"Epochs per experiment: {num_epochs}")
        print("-" * 80)
    
    total_experiments = len(alpha_values) * len(gamma_values)
    experiment_count = 0
    
    for alpha in alpha_values:
        for gamma in gamma_values:
            experiment_count += 1
            
            if verbose:
                print(f"\nExperiment {experiment_count}/{total_experiments}: alpha={alpha}, gamma={gamma}")
                print("-" * 40)
            
            # Update config
            config.focal_alpha = alpha
            config.focal_gamma = gamma
            
            # Reinitialize model
            model = get_model(config).to(device)
            
            # Reinitialize optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=config.betas,
                eps=config.eps
            )
            
            # Reinitialize loss function
            loss_fn = FocalLoss(
                alpha=config.focal_alpha,
                gamma=config.focal_gamma,
                reduction=config.loss_reduction
            )
            
            try:
                # Train with best model selection
                history, best_info = train_full_supervision_with_selection(
                    model, train_loader, val_loader, optimizer, device, loss_fn, config
                )
                
                # Store results
                result = {
                    'alpha': alpha,
                    'gamma': gamma,
                    'best_epoch': best_info['best_epoch'],
                    'best_val_auprc': best_info['best_metric'],
                    'best_f1_target': best_info['best_f1_target'],
                    'history': history,
                    'success': True
                }
                
                # Check if this is the best result
                if best_info['best_metric'] > best_metric:
                    best_metric = best_info['best_metric']
                    best_result = result.copy()
                
                if verbose:
                    print(f"✓ Success: AUPRC={best_info['best_metric']:.4f}, F1(target)={best_info['best_f1_target']:.4f}")
                
            except Exception as e:
                result = {
                    'alpha': alpha,
                    'gamma': gamma,
                    'error': str(e),
                    'success': False
                }
                if verbose:
                    print(f"✗ Failed: {str(e)}")
            
            grid_results.append(result)
    
    # Restore original config values
    config.focal_alpha = original_alpha
    config.focal_gamma = original_gamma
    config.num_epochs = original_epochs
    
    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print("GRID SEARCH RESULTS SUMMARY")
        print("=" * 80)
        
        successful_results = [r for r in grid_results if r.get('success', False)]
        if successful_results:
            print(f"Successful experiments: {len(successful_results)}/{total_experiments}")
            print(f"\nBest result:")
            print(f"  Alpha: {best_result['alpha']}")
            print(f"  Gamma: {best_result['gamma']}")
            print(f"  Best AUPRC: {best_result['best_val_auprc']:.4f}")
            print(f"  Best F1(target): {best_result['best_f1_target']:.4f}")
            print(f"  Best epoch: {best_result['best_epoch']}")
            
            # Show top 3 results
            sorted_results = sorted(successful_results, key=lambda x: x['best_val_auprc'], reverse=True)
            print(f"\nTop 3 combinations:")
            for i, result in enumerate(sorted_results[:3]):
                print(f"  {i+1}. α={result['alpha']}, γ={result['gamma']} → AUPRC={result['best_val_auprc']:.4f}")
        else:
            print("No successful experiments!")
    
    return {
        'best_result': best_result,
        'all_results': grid_results,
        'best_alpha': best_result['alpha'] if best_result else None,
        'best_gamma': best_result['gamma'] if best_result else None,
        'best_auprc': best_metric
    }
