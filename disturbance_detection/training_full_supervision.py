import torch
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, accuracy_score


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
    preds_all = (all_probs.flatten() > 0.5).astype(int)
    train_f1_all = f1_score(all_labels.flatten(), preds_all, average='binary', zero_division=0)
    train_acc = accuracy_score(all_labels.flatten(), preds_all)

    # Target timestep F1
    if config.target_mode == "last":
        target_pos = -1
    elif config.target_mode == "second_last":
        target_pos = -2
    elif config.target_mode == "center":
        target_pos = all_labels.shape[1] // 2
    else:
        target_pos = -1

    train_f1_target = f1_score(
        all_labels[:, target_pos],
        (all_probs[:, target_pos] > 0.5).astype(int),
        zero_division=0
    )

    avg_loss = total_loss / len(train_loader)
    return avg_loss, train_acc, train_f1_all, train_f1_target


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

    # Determine target timestep index from first batch
    sample_y = next(iter(val_loader))[1]
    T = sample_y.shape[1]
    if config.target_mode == "last":
        target_pos = -1
    elif config.target_mode == "second_last":
        target_pos = -2
    elif config.target_mode == "center":
        target_pos = T // 2
    else:
        target_pos = -1

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
    val_f1_all = f1_score(all_labels_flat, (all_probs_flat > 0.5).astype(int),
                           average='binary', zero_division=0)

    # Target timestep metrics
    y_true_target = torch.cat(val_labels_target).numpy().astype(int) if val_labels_target else np.array([])
    y_prob_target = torch.cat(val_probs_target).numpy() if val_probs_target else np.array([])

    if y_true_target.size and len(np.unique(y_true_target)) == 2:
        val_auprc = average_precision_score(y_true_target, y_prob_target)
        precision, recall, thresholds = precision_recall_curve(y_true_target, y_prob_target)
        f1_curve = 2 * precision * recall / (precision + recall + 1e-8)
        val_f1_target_best = float(np.nanmax(f1_curve[1:])) if f1_curve.size > 1 else 0.0
    else:
        val_auprc = np.nan
        val_f1_target_best = 0.0

    return val_loss, val_f1_all, val_f1_target_best, val_auprc



def train_full_supervision_with_selection(model, train_loader, val_loader, optimizer, device, loss_fn, config):
    """
    Full training loop with model selection based on target timestep AUPRC.
    """
    best_metric = -np.inf
    best_epoch = -1
    history = {
        "train_loss": [], "train_f1_all": [], "train_f1_target": [],
        "val_loss": [], "val_f1_all": [], "val_f1_target": [], "val_auprc": []
    }

    for epoch in range(config.num_epochs):
        train_loss, train_acc, train_f1_all, train_f1_target = train_epoch_full_supervision(
            model, train_loader, optimizer, device, loss_fn, config
        )

        val_loss, val_f1_all, val_f1_target, val_auprc = validate_epoch_full_supervision(
            model, val_loader, device, loss_fn, config
        )

        # Log metrics
        history["train_loss"].append(train_loss)
        history["train_f1_all"].append(train_f1_all)
        history["train_f1_target"].append(train_f1_target)
        history["val_loss"].append(val_loss)
        history["val_f1_all"].append(val_f1_all)
        history["val_f1_target"].append(val_f1_target)
        history["val_auprc"].append(val_auprc)

        print(f"Epoch {epoch+1:02d} | TrainLoss {train_loss:.4f} | "
              f"TrainF1(all) {train_f1_all:.4f} | TrainF1(target) {train_f1_target:.4f} | "
              f"ValLoss {val_loss:.4f} | ValF1(all) {val_f1_all:.4f} | ValF1* {val_f1_target:.4f} | ValAUPRC {val_auprc:.4f}")

        # Model selection based on AUPRC at target timestep
        current_metric = val_auprc if config.select_by == "auprc" else -val_loss
