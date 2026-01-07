############ for  last position
###### Band permutation
###### Temporal oclusion

import numpy as np, torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

# ---------- small utilities ----------
def _ensure_pos_index(t_idx, T):
    j = t_idx if t_idx >= 0 else T + t_idx
    if not (0 <= j < T):
        raise IndexError(f"target_index {t_idx} invalid for sequence length T={T}")
    return j

@torch.no_grad()
@torch.no_grad()
def _stack_loader(loader):
    Xs, ys = [], []
    for x, y in loader:  # Only expect 2 values from data loader
        Xs.append(x.cpu().numpy())
        ys.append(y.cpu().numpy())
    
    X = np.concatenate(Xs, 0)  # Shape: (N, C, T)
    y = np.concatenate(ys, 0)  # Shape: (N, T)
    
    # Create a mask of all ones (all timesteps are valid)
    m = np.ones_like(y, dtype=bool)
    
    return X, y, m  # X:(N,C,T), y/m:(N,T)

    
def _target_mask_at(m, target_index=-1):
    """one-hot at target step (supports negative index like -1, -2, etc.)."""
    T = m.shape[1]
    j = _ensure_pos_index(target_index, T)
    tm = np.zeros_like(m, dtype=bool)
    tm[:, j] = True
    return tm

@torch.no_grad()
def _predict_logit_at(model, X_np, device, target_index=-1, batch=2048):
    """X_np: (N,C,T) -> logits at target timestep as (N,)"""
    model.eval()
    T = X_np.shape[-1]
    j = _ensure_pos_index(target_index, T)
    out = []
    for i in range(0, X_np.shape[0], batch):
        xb = torch.from_numpy(X_np[i:i+batch]).to(device).float()
        logits = model(xb)         # expect (B,T) or (B,T,1)
        if logits.ndim == 3 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        out.append(logits[:, j].detach().cpu().numpy())
    return np.concatenate(out, 0)

def _metrics_imbalance(y_true, y_prob, thr=None):
    """AUPRC + Lift (AUPRC / prevalence) + optional thresholded F1 (secondary)"""
    prevalence = float(np.mean(y_true))
    auprc = average_precision_score(y_true, y_prob)
    lift = auprc / max(prevalence, 1e-8)

    if thr is None:
        P, R, th = precision_recall_curve(y_true, y_prob)
        f1 = 2*P*R/(P+R+1e-8)
        best = np.nanargmax(f1[1:]) + 1 if len(f1)>1 else 0
        thr = th[best-1] if len(th) else 0.5

    y_pred = (y_prob >= thr).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score
    return dict(thr=float(thr), prevalence=prevalence, auprc=auprc, lift=lift,
                f1=f1_score(y_true, y_pred, zero_division=0),
                precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0))

# ---------- BAND PERMUTATION (generalized target) ----------
@torch.no_grad()
def band_permutation_importance(model, loader, device, n_repeats=30,
                                base_threshold=None, seed=42, target_index=-1):
    """
    Permute one channel across samples (keep each sample's time intact), measure AUPRC/Lift drop
    at the specified target_index (e.g., -2 for second-to-last).
    Returns:
      base: baseline metrics dict
      table: list of dicts per band with mean±std ΔAUPRC and ΔLift
      drops_raw: raw arrays per band for custom CIs
    """
    rng = np.random.RandomState(seed)
    X, y, m = _stack_loader(loader)              # X:(N,C,T)
    N, C, T = X.shape
    j = _ensure_pos_index(target_index, T)
    tm = _target_mask_at(m, target_index=j)
    y_true = y[tm]                               # (N,)

    # Baseline
    base_logits = _predict_logit_at(model, X, device, target_index=j)
    base_prob   = 1/(1+np.exp(-base_logits))
    base = _metrics_imbalance(y_true, base_prob, thr=base_threshold)
    thr = base["thr"] if base_threshold is None else base_threshold

    # Collect drops
    drops = {b: {"auprc": [], "lift": []} for b in range(C)}
    for _ in range(n_repeats):
        perm = rng.permutation(N)
        for b in range(C):
            Xp = X.copy()
            Xp[:, b, :] = X[perm, b, :]          # permute band b across samples
            logits_p = _predict_logit_at(model, Xp, device, target_index=j)
            prob_p   = 1/(1+np.exp(-logits_p))
            met_p    = _metrics_imbalance(y_true, prob_p, thr=thr)
            drops[b]["auprc"].append(base["auprc"] - met_p["auprc"])
            drops[b]["lift"].append(base["lift"] - met_p["lift"])

    # Summarize
    table = []
    for b in range(C):
        auprc_d = np.array(drops[b]["auprc"]); lift_d = np.array(drops[b]["lift"])
        table.append(dict(
            band_index=b,
            delta_auprc_mean=float(auprc_d.mean()), delta_auprc_std=float(auprc_d.std()),
            delta_lift_mean=float(lift_d.mean()),   delta_lift_std=float(lift_d.std()),
        ))
    return base, table, drops

# ---------- TEMPORAL OCCLUSION (generalized target) ----------
@torch.no_grad()
def _baseline_per_band_time_undisturbed(X, y, m, target_index=-1):
    """Mean (per band,time) over windows whose label at target step is 0; fallback=global mean."""
    T = X.shape[-1]
    j = _ensure_pos_index(target_index, T)
    tm = _target_mask_at(m, target_index=j)
    und = (y[tm] == 0)
    return (X[und].mean(axis=0, keepdims=True) if und.any() else X.mean(axis=0, keepdims=True))  # (1,C,T)

@torch.no_grad()
def temporal_occlusion(model, loader, device, per_band=True, normalize_by_abs_logit=True, target_index=-1):
    """
    For disturbed windows (label==1 at target step), occlude time t by baseline and
    measure Δlogit at the target step.
    Returns:
      rel: np.arange(-j, T-j) where j is the absolute target index
      d_all: (T,) mean Δ (normalized if normalize_by_abs_logit)
      d_bands: (C,T) or None
    """
    X, y, m = _stack_loader(loader)
    N, C, T = X.shape
    j = _ensure_pos_index(target_index, T)

    tm = _target_mask_at(m, target_index=j)
    pos_idx = (y[tm] == 1)
    if pos_idx.sum() == 0:
        raise RuntimeError("No positive samples at target step in this split.")
    Xp = X[pos_idx]

    baseline = _baseline_per_band_time_undisturbed(X, y, m, target_index=j)  # (1,C,T)
    base_logits = _predict_logit_at(model, Xp, device, target_index=j)       # (Np,)

    d_all = np.zeros(T, float); n_all = np.zeros(T, int)
    d_b   = np.zeros((C,T), float) if per_band else None
    n_b   = np.zeros((C,T), int)   if per_band else None

    for t in range(T):
        Xocc = Xp.copy(); Xocc[:, :, t] = baseline[0, :, t]
        lo = _predict_logit_at(model, Xocc, device, target_index=j)
        d  = base_logits - lo
        if normalize_by_abs_logit:
            d = d / (np.abs(base_logits) + 1e-6)
        d_all[t] += d.mean(); n_all[t] += 1

        if per_band:
            for b in range(C):
                Xb = Xp.copy(); Xb[:, b, t] = baseline[0, b, t]
                lob = _predict_logit_at(model, Xb, device, target_index=j)
                db  = base_logits - lob
                if normalize_by_abs_logit:
                    db = db / (np.abs(base_logits) + 1e-6)
                d_b[b, t] += db.mean(); n_b[b, t] += 1

    d_all = d_all / np.maximum(n_all, 1)
    if per_band: d_b = d_b / np.maximum(n_b, 1)
    rel = np.arange(-j, T - j)                       # e.g., T=5, j=3 -> [-3,-2,-1,0,1]
    return rel, d_all, d_b