"""
Training functions for forest disturbance detection.
Includes target masking, training loops, and model selection.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve


def make_target_mask(m: torch.Tensor, mode: str = "last") -> torch.Tensor:
    """
    Build a 1-hot mask over time per sample using the valid length inferred from m (1=valid, 0=pad).
    
    Supported modes:
      - "center": floor(center) within the valid part
      - "last": last valid step
      - "second_last": second-to-last valid step (clamped to 0 if length==1)
      - "idx:<int>": fixed index; negatives mean from the end (e.g., idx:-1 == last)
    
    Args:
        m: Mask tensor (B, T) where 1=valid timestep, 0=padding
        mode: Target position mode
        
    Returns:
        Target mask (B, T) with 1 at supervised position, 0 elsewhere
    """
    B, T = m.shape
    lens = m.sum(dim=1).long().clamp(min=1)  # valid lengths per sample

    if mode == "center":
        idx = lens // 2
    elif mode == "last":
        idx = lens - 1
    elif mode == "second_last":
        idx = (lens - 2).clamp(min=0)
    elif mode.startswith("idx:"):
        k = int(mode.split(":", 1)[1])
        # support negative indexing from the end
        idx = torch.where(torch.tensor(k >= 0, device=m.device), torch.full_like(lens, k), lens + k)
        idx = idx.clamp(min=0, max=lens - 1)
    else:
        raise ValueError("mode must be 'center', 'last', 'second_last', or 'idx:<int>'")

    tm = torch.zeros_like(m)
    tm[torch.arange(B, device=m.device), idx] = 1.0
    return tm


def safe_to_device(module, device):
    """
    Safely move module to device with fallback to CPU on CUDA errors.
    
    Args:
        module: PyTorch module to move
        device: Target device
        
    Returns:
        Module on device (or CPU if device failed)
    """
    try:
        return module.to(device)
    except RuntimeError as e:
        print("[WARN] CUDA issue, falling back to CPU:", e)
        torch.cuda.empty_cache()
        return module.to("cpu")


def train_model(model, train_loader, val_loader, device, loss_fn, optimizer, config):
    """
    Basic training loop with validation.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        loss_fn: Loss function (should return per-element loss with reduction='none')
        optimizer: Optimizer
        config: Config object containing training parameters
    """
    for epoch in range(config.num_epochs):
        model.train()
        train_loss, train_f1 = 0.0, 0.0
        
        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad()

            y_hat = model(x)                   # (B, T)
            per_elem = loss_fn(y_hat, y)       # (B, T)

            # Apply target mask to focus on supervised position
            tm = make_target_mask(m, mode=config.target_mode)
            loss = (per_elem * tm).sum() / (tm.sum() + 1e-8)

            loss.backward()
            
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()

            # Compute F1 on supervised positions only
            preds = (torch.sigmoid(y_hat) > 0.5).float()
            train_f1 += f1_score(
                y[tm.bool()].cpu(), preds[tm.bool()].cpu(), zero_division=0
            )
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_f1   /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss, val_f1 = 0.0, 0.0
        
        with torch.no_grad():
            for x, y, m in val_loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                y_hat = model(x)
                per_elem = loss_fn(y_hat, y)
                tm = make_target_mask(m, mode=config.target_mode)
                loss = (per_elem * tm).sum() / (tm.sum() + 1e-8)

                preds = (torch.sigmoid(y_hat) > 0.5).float()
                val_f1 += f1_score(
                    y[tm.bool()].cpu(), preds[tm.bool()].cpu(), zero_division=0
                )
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_f1   /= len(val_loader)
        
        print(f"Epoch {epoch+1:02d} | TrainLoss {train_loss:.4f} | TrainF1 {train_f1:.4f} | "
              f"ValLoss {val_loss:.4f} | ValF1 {val_f1:.4f}")


def train_and_select_best(model, train_loader, val_loader, device, loss_fn, optimizer, config):
    """
    Training loop with best model selection by validation AUPRC or loss.
    Saves the best epoch checkpoint.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        loss_fn: Loss function (should return per-element loss with reduction='none')
        optimizer: Optimizer
        config: Config object containing training parameters
        
    Returns:
        tuple: (history dict, best_info dict)
    """
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "val_auprc": []}
    best_metric = -np.inf if config.select_by == "auprc" else np.inf
    best_epoch = -1

    for epoch in range(config.num_epochs):
        # -------- TRAIN --------
        model.train()
        train_loss, train_f1 = 0.0, 0.0
        
        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad()

            y_hat = model(x)
            per_elem = loss_fn(y_hat, y)
            tm = make_target_mask(m, mode=config.target_mode)
            loss = (per_elem * tm).sum() / (tm.sum() + 1e-8)

            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            preds = (torch.sigmoid(y_hat) > 0.5).float()
            train_f1 += f1_score(y[tm.bool()].cpu(), preds[tm.bool()].cpu(), zero_division=0)
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        train_f1   /= max(1, len(train_loader))

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0
        val_probs_all, val_labels_all = [], []
        
        with torch.no_grad():
            for x, y, m in val_loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                y_hat = model(x)
                per_elem = loss_fn(y_hat, y)
                tm = make_target_mask(m, mode=config.target_mode)
                loss = (per_elem * tm).sum() / (tm.sum() + 1e-8)
                val_loss += loss.item()

                probs = torch.sigmoid(y_hat)
                val_probs_all.append(probs[tm.bool()].detach().cpu())
                val_labels_all.append(y[tm.bool()].detach().cpu())

        val_loss /= max(1, len(val_loader))
        y_true_va = torch.cat(val_labels_all).numpy().astype(int) if val_labels_all else np.array([])
        y_prob_va = torch.cat(val_probs_all).numpy() if val_probs_all else np.array([])

        # Compute AUPRC and best F1
        if y_true_va.size and len(np.unique(y_true_va)) == 2:
            ap_va = average_precision_score(y_true_va, y_prob_va)
            P, R, th = precision_recall_curve(y_true_va, y_prob_va)
            f1_curve = 2 * P * R / (P + R + 1e-8)
            f1_va_best = float(np.nanmax(f1_curve[1:])) if f1_curve.size > 1 else 0.0
        else:
            ap_va = np.nan
            f1_va_best = 0.0

        # Log history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(f1_va_best)
        history["val_auprc"].append(float(ap_va) if ap_va == ap_va else np.nan)

        print(f"Epoch {epoch+1:02d} | "
              f"TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
              f"TrainF1 {train_f1:.4f} | ValF1* {f1_va_best:.4f} | ValAUPRC {ap_va:.4f}")

        # Select best epoch
        current_metric = ap_va if config.select_by == "auprc" else -val_loss
        is_better = (config.select_by == "auprc" and (ap_va == ap_va) and current_metric > best_metric) or \
                    (config.select_by == "loss"  and current_metric > best_metric)
        
        if is_better:
            best_metric = current_metric
            best_epoch = epoch + 1
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_auprc": ap_va
            }, config.checkpoint_path)

    return history, {"best_epoch": best_epoch, "best_metric": best_metric,
                     "ckpt_path": config.checkpoint_path, "select_by": config.select_by}