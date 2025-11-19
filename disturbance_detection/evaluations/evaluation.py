"""
Evaluation and metrics functions for forest disturbance detection.
Includes threshold selection, metrics computation, and visualization.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, 
                            precision_recall_curve, average_precision_score)


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


@torch.no_grad()
def count_supervised_positives(loader, target_mode="last", device="cpu"):
    """
    Count positive samples at the supervised target position.
    Useful for understanding class balance at the position you're predicting.
    
    Args:
        loader: DataLoader
        target_mode: Target position mode
        device: Device to use
        
    Returns:
        tuple: (num_positives, total_samples, positive_rate)
    """
    pos = 0
    tot = 0
    for x, y, m in loader:
        m = m.to(device)
        y = y.to(device)
        tm = make_target_mask(m, mode=target_mode)
        tgt = y[tm.bool()]
        pos += int(tgt.sum().item())
        tot += int(tgt.numel())
    return pos, tot, (pos / max(tot, 1))


@torch.no_grad()
def collect_probs(model, data_loader, device, target_mode="last"):
    """
    Collect model predictions and ground truth labels at target position.
    
    Args:
        model: Trained model
        data_loader: DataLoader
        device: Device to use
        target_mode: Target position mode
        
    Returns:
        tuple: (y_true, y_prob) as numpy arrays
    """
    model.eval()
    probs_all, labels_all = [], []
    
    for x, y, m in data_loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        logits = model(x)                    # (B, T)
        probs  = torch.sigmoid(logits)       # (B, T)
        tm = make_target_mask(m, mode=target_mode)  # (B, T) one-hot per window
        
        # Pick the supervised timestep
        probs_t = probs[tm.bool()]           # (B,)
        labels_t = y[tm.bool()]              # (B,)
        
        probs_all.append(probs_t.detach().cpu())
        labels_all.append(labels_t.detach().cpu())
    
    y_true = torch.cat(labels_all).numpy().astype(int)
    y_prob = torch.cat(probs_all).numpy()
    return y_true, y_prob


def pick_threshold_by_f1(y_true, y_prob):
    """
    Select optimal classification threshold by maximizing F1 score on validation set.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_prob: Predicted probabilities
        
    Returns:
        tuple: (best_threshold, best_f1, precision, recall, (P, R, th, f1_curve))
    """
    P, R, th = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(th)+1 points; align F1 to P,R
    f1 = 2 * P * R / (P + R + 1e-8)
    # the first P,R correspond to threshold = +inf (no positives); skip that for argmax
    best_idx = np.nanargmax(f1[1:]) + 1
    best_thr = th[best_idx - 1] if len(th) else 0.5
    return float(best_thr), float(f1[best_idx]), float(P[best_idx]), float(R[best_idx]), (P, R, th, f1)


def eval_at_threshold(y_true, y_prob, thr):
    """
    Compute classification metrics at a given threshold.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        thr: Classification threshold
        
    Returns:
        dict: Metrics including precision, recall, f1, confusion matrix
    """
    y_pred = (y_prob >= thr).astype(int)
    return dict(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        cm=confusion_matrix(y_true, y_pred, labels=[0, 1])
    )


def final_eval_with_val_threshold(model, device, ckpt_path, val_loader, test_loader, target_mode="last"):
    """
    Complete evaluation pipeline:
    1. Load best model checkpoint
    2. Calibrate threshold on validation set (maximize F1)
    3. Apply frozen threshold to test set
    4. Report comprehensive metrics
    
    Args:
        model: Model architecture (state will be loaded from checkpoint)
        device: Device to use
        ckpt_path: Path to best model checkpoint
        val_loader: Validation data loader
        test_loader: Test data loader
        target_mode: Target position mode
        
    Returns:
        tuple: (best_threshold, test_metrics_dict)
    """
    # Load best model
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded best model from epoch {ckpt['epoch']}.")

    # 1) Calibrate threshold ONCE on validation
    y_true_val, y_prob_val = collect_probs(model, val_loader, device, target_mode=target_mode)
    best_thr, best_f1_val, best_P_val, best_R_val, _ = pick_threshold_by_f1(y_true_val, y_prob_val)
    ap_val = average_precision_score(y_true_val, y_prob_val) if len(np.unique(y_true_val)) == 2 else np.nan
    print(f"[VAL] thr*={best_thr:.3f} | F1*={best_f1_val:.3f} | AUPRC={ap_val:.3f}")

    # 2) Apply frozen threshold to TEST (once)
    y_true_test, y_prob_test = collect_probs(model, test_loader, device, target_mode=target_mode)
    y_pred_test = (y_prob_test >= best_thr).astype(int)
    
    test_metrics = {
        "precision": precision_score(y_true_test, y_pred_test, zero_division=0),
        "recall":    recall_score(y_true_test, y_pred_test, zero_division=0),
        "f1":        f1_score(y_true_test, y_pred_test, zero_division=0),
        "accuracy":  accuracy_score(y_true_test, y_pred_test),
        "cm":        confusion_matrix(y_true_test, y_pred_test, labels=[0, 1]),
        "auprc":     average_precision_score(y_true_test, y_prob_test) if len(np.unique(y_true_test)) == 2 else np.nan
    }
    
    print(f"[TEST] thr={best_thr:.3f} | "
          f"P={test_metrics['precision']:.3f} R={test_metrics['recall']:.3f} "
          f"F1={test_metrics['f1']:.3f} Acc={test_metrics['accuracy']:.3f} | "
          f"AUPRC={test_metrics['auprc']:.3f}")
    print("Confusion matrix (TEST) [rows=true 0/1, cols=pred 0/1]:\n", test_metrics["cm"])
    
    return best_thr, test_metrics


def print_classification_report(y_true, y_prob, threshold, target_names=["Undisturbed (0)", "Disturbed (1)"]):
    """
    Print detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        threshold: Classification threshold
        target_names: Class names for report
    """
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0, digits=3))


def plot_precision_recall_curve(y_true, y_prob, title="Precision-Recall Curve"):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        title: Plot title
    """
    P, R, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    
    plt.figure(figsize=(6, 5))
    plt.plot(R, P, lw=2, label=f"AUPRC={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_f1_vs_threshold(y_true, y_prob, best_threshold=None, title="F1 Score vs Threshold"):
    """
    Plot F1 score as a function of classification threshold.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        best_threshold: Optimal threshold to highlight (optional)
        title: Plot title
    """
    P, R, th = precision_recall_curve(y_true, y_prob)
    f1 = 2 * P[1:] * R[1:] / (P[1:] + R[1:] + 1e-8)
    
    plt.figure(figsize=(6, 5))
    plt.plot(th, f1, lw=2)
    if best_threshold is not None:
        plt.axvline(best_threshold, ls="--", color='r', alpha=0.7, 
                   label=f"Best thr={best_threshold:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if best_threshold is not None:
        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_history(history, title_suffix=""):
    """
    Plot training history curves (loss, F1, AUPRC over epochs).
    
    Args:
        history: Dictionary with training history
        title_suffix: Optional suffix for plot titles
    """
    ep = np.arange(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(ep, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(ep, history["val_loss"], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss over epochs{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # F1
    plt.subplot(1, 3, 2)
    plt.plot(ep, history["train_f1"], marker="o", label="Train F1")
    plt.plot(ep, history["val_f1"], marker="o", label="Val F1 (best-per-epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title(f"F1 over epochs{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # AUPRC (if available)
    if "val_auprc" in history:
        plt.subplot(1, 3, 3)
        plt.plot(ep, history["val_auprc"], marker="o", label="Val AUPRC", color='green')
        plt.xlabel("Epoch")
        plt.ylabel("AUPRC")
        plt.title(f"AUPRC over epochs{title_suffix}")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, labels=["Undisturbed", "Disturbed"], title="Confusion Matrix"):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix (2x2 array)
        labels: Class labels
        title: Plot title
    """
    import seaborn as sns
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def evaluate_position_wise_metrics(model, val_loader, device, config):

    ''' Evaluate F1 score for each position in the sequence'''
    
    eps = 1e-8
    model.eval()
    val_probs_all, val_labels_all = [], []
    with torch.no_grad():
        for x, y, _ in val_loader:
            x,y = x.to(device), y.to(device)
            y_hat = model(x)
            probs = torch.sigmoid(y_hat).cpu()
            val_probs_all.append(probs)
            val_labels_all.append(y.cpu())
    all_probs = torch.cat(val_probs_all).numpy()
    all_labels = torch.cat(val_labels_all).numpy()

    T = all_labels.shape[1]

    # Compute metrics for each position
    position_metrics = []

    for pos in range(T):
        y_true_pos = all_labels[:, pos]
        y_prob_pos = all_probs[:, pos]
        
        # Compute best F1 (like in training validation)
        if len(np.unique(y_true_pos)) == 2:
            from sklearn.metrics import precision_recall_curve
            precision, recall, thresholds = precision_recall_curve(y_true_pos, y_prob_pos)
            f1_curve = 2 * precision * recall / (precision + recall + eps)
            f1_pos = float(np.nanmax(f1_curve[1:])) if f1_curve.size > 1 else 0.0
            
            # Also compute at best threshold
            best_idx = np.nanargmax(f1_curve[1:]) + 1
            precision_pos = float(precision[best_idx])
            recall_pos = float(recall[best_idx])
        else:
            f1_pos = 0.0
            precision_pos = 0.0
            recall_pos = 0.0
        
        position_metrics.append({
            'position': pos,
            'f1': f1_pos,
            'precision': precision_pos,
            'recall': recall_pos
        })
    return position_metrics

def load_best_model_and_evaluate_position_wise_metrics(model, device, ckpt_path, val_loader, config):
    '''Load the best model and evaluate position-wise metrics'''

    # Load the best checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}.")
    print(f"Evaluating position-wise metrics for the best model...")
    position_metrics = evaluate_position_wise_metrics(model, val_loader, device, config)
    print(f"Position-wise metrics: {position_metrics}")

    for pos, metrics in position_metrics:
        print(f"Position {pos}: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
    return position_metrics