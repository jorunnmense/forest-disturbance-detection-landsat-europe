"""
Data loading and preprocessing utilities for forest disturbance detection.
Handles data splits, windowing, normalization, and DataLoader creation.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os


def make_or_load_uid_splits(df, split_path, seed=42):
    """
    Returns (train_uids, val_uids, test_uids) as np arrays.
    If split_path exists, loads it; else creates stratified 70/10/20 by uniqueid and saves.
    Stratification is by whether a uniqueid contains any positive (new_class2_v5 == 1).
    Prints class balance per split for verification.
    
    Args:
        df: DataFrame with 'uniqueid' and 'new_class2_v5' columns
        split_path: Path to save/load the split .npz file
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_uids, val_uids, test_uids) as numpy arrays
    """
    if os.path.exists(split_path):
        z = np.load(split_path, allow_pickle=True)
        print(f"[INFO] Loaded existing split file: {split_path}")
        return z["train_uids"], z["val_uids"], z["test_uids"]

    uid_groups = df.groupby("uniqueid")["new_class2_v5"]
    uids = uid_groups.count().index.values
    has_pos = (uid_groups.max().values > 0).astype(int)

    train_u, temp_u = train_test_split(
        uids, test_size=0.30, random_state=seed, stratify=has_pos
    )

    pos_map = dict(zip(uids, has_pos))
    temp_labels = np.array([pos_map[u] for u in temp_u])

    val_u, test_u = train_test_split(
        temp_u, test_size=0.66, random_state=seed, stratify=temp_labels
    )

    np.savez(split_path, train_uids=train_u, val_uids=val_u, test_uids=test_u, seed=seed)
    print(f"[INFO] Saved new split file: {split_path}")

    def count_pos_neg(uids_subset):
        sub = df[df["uniqueid"].isin(uids_subset)]
        total = len(sub)
        pos = int(sub["new_class2_v5"].sum())
        neg = total - pos
        return total, pos, neg, pos / max(total, 1)

    tr_total, tr_pos, tr_neg, tr_ratio = count_pos_neg(train_u)
    va_total, va_pos, va_neg, va_ratio = count_pos_neg(val_u)
    te_total, te_pos, te_neg, te_ratio = count_pos_neg(test_u)

    print("\n[Split summary by pixel-level samples]")
    print(f"Train: total={tr_total:,} | pos={tr_pos:,} | neg={tr_neg:,} | pos%={100*tr_ratio:.3f}")
    print(f"Val:   total={va_total:,} | pos={va_pos:,} | neg={va_neg:,} | pos%={100*va_ratio:.3f}")
    print(f"Test:  total={te_total:,} | pos={te_pos:,} | neg={te_neg:,} | pos%={100*te_ratio:.3f}")
    print("-----------------------------------------------------------")

    def uid_pos_ratio(uids_subset):
        labels = np.array([pos_map[u] for u in uids_subset])
        return labels.sum(), len(labels) - labels.sum(), labels.mean()

    tr_uid_pos, tr_uid_neg, tr_uid_ratio = uid_pos_ratio(train_u)
    va_uid_pos, va_uid_neg, va_uid_ratio = uid_pos_ratio(val_u)
    te_uid_pos, te_uid_neg, te_uid_ratio = uid_pos_ratio(test_u)

    print("[Split summary by uniqueid]")
    print(f"Train: total={len(train_u):,} | disturbed_uids={tr_uid_pos:,} | pos%={100*tr_uid_ratio:.3f}")
    print(f"Val:   total={len(val_u):,} | disturbed_uids={va_uid_pos:,} | pos%={100*va_uid_ratio:.3f}")
    print(f"Test:  total={len(test_u):,} | disturbed_uids={te_uid_pos:,} | pos%={100*te_uid_ratio:.3f}")
    print("===========================================================")

    return train_u, val_u, test_u


def prepare_data(df, config):
    """
    Main data preparation function: windowing, normalization, and DataLoader creation.
    
    Don't fillna(0.0) (keep real missing), pad with NaN (not 0),
    compute min–max on train valid timesteps only, ignoring NaN and −9999,
    after scaling, turn remaining NaN/−9999 into 0, and zero-out padded timesteps with the timestep mask.
    
    Args:
        df: DataFrame with features and labels
        config: Config object containing all parameters
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, n_features, used_features)
    """
    # --- Get features from config ---
    features = config.get_features()
    n_features = config.get_num_features()
    
    # --- fixed uniqueid splits (create once, then always load) ---
    train_uids, val_uids, test_uids = make_or_load_uid_splits(
        df, config.split_path, config.seed
    )
    train_uids_set, val_uids_set, test_uids_set = set(train_uids), set(val_uids), set(test_uids)

    # buckets
    X_tr_list, y_tr_list, m_tr_list = [], [], []
    X_va_list, y_va_list, m_va_list = [], [], []
    X_te_list, y_te_list, m_te_list = [], [], []

    def add_sample(uid, X, y, m):
        if uid in train_uids_set:
            X_tr_list.append(X); y_tr_list.append(y); m_tr_list.append(m)
        elif uid in val_uids_set:
            X_va_list.append(X); y_va_list.append(y); m_va_list.append(m)
        elif uid in test_uids_set:
            X_te_list.append(X); y_te_list.append(y); m_te_list.append(m)

    # --- build windows per uniqueid, routed into the right split ---
    for uid, group in df.groupby('uniqueid'):
        if uid not in train_uids_set and uid not in val_uids_set and uid not in test_uids_set:
            continue
        group = group.sort_values('year')
        if len(group) < config.window_size:
            continue

        # keep true missing values; DO NOT fill here
        data  = group[features].to_numpy(dtype=float)          # shape [T_uid, F]; may contain NaN or -9999
        label = group['new_class2_v5'].to_numpy(dtype=float)   # shape [T_uid]

        for i in range(len(data) - config.window_size + 1):
            seq_x = data[i:i+config.window_size]                      # [W, F]
            seq_y = label[i:i+config.window_size]                     # [W]
            mask  = np.ones(config.window_size, dtype=np.float32)     # timestep mask (1=real)

            # pad to max_window with NaN for features, 0 for labels, 0 for mask
            pad_len = config.max_window - config.window_size
            if pad_len > 0:
                seq_x = np.pad(seq_x, ((0,pad_len),(0,0)), mode='constant', constant_values=np.nan)
                seq_y = np.pad(seq_y, (0,pad_len),          mode='constant', constant_values=0.0)
                mask  = np.pad(mask,  (0,pad_len),          mode='constant', constant_values=0.0)

            add_sample(uid, seq_x, seq_y, mask)

    # stack per split
    def stack_or_empty(XL, yL, mL):
        if len(XL) == 0:
            return (np.empty((0, config.max_window, len(features)), dtype=float),
                    np.empty((0, config.max_window), dtype=float),
                    np.empty((0, config.max_window), dtype=float))
        return np.stack(XL), np.stack(yL), np.stack(mL)

    X_train, y_train, m_train = stack_or_empty(X_tr_list, y_tr_list, m_tr_list)
    X_val,   y_val,   m_val   = stack_or_empty(X_va_list, y_va_list, m_va_list)
    X_test,  y_test,  m_test  = stack_or_empty(X_te_list, y_te_list, m_te_list)

    # --- normalization: per-feature min–max using TRAIN valid timesteps only ---
    if X_train.shape[0] == 0:
        raise RuntimeError("Empty training split after windowing. Check filters/window_size.")

    def to_nan(x):
        x = x.copy()
        x[x == -9999.0] = np.nan
        return x

    X_train_nan = to_nan(X_train)
    X_val_nan   = to_nan(X_val)
    X_test_nan  = to_nan(X_test)

    N_tr, T, F = X_train_nan.shape
    Xt_tr = X_train_nan.reshape(-1, F)
    mt_tr = m_train.reshape(-1).astype(bool)
    Xt_tr_valid = Xt_tr[mt_tr]

    feat_min = np.nanmin(Xt_tr_valid, axis=0, keepdims=True)
    feat_max = np.nanmax(Xt_tr_valid, axis=0, keepdims=True)
    feat_rng = np.clip(feat_max - feat_min, 1e-8, None)

    def norm_per_feature(X_nan, M):
        Xn = (X_nan - feat_min.reshape(1,1,F)) / feat_rng.reshape(1,1,F)
        Xn = np.where(np.isnan(Xn), 0.0, Xn)
        Xn = Xn * M[..., None]
        return np.clip(Xn, 0.0, 1.0)

    X_train = norm_per_feature(X_train_nan, m_train)
    X_val   = norm_per_feature(X_val_nan,   m_val)
    X_test  = norm_per_feature(X_test_nan,  m_test)

    # Save normalization stats
    np.save(config.min_train_path, feat_min)
    np.save(config.max_train_path, feat_max)

    # --- tensors in (B, C, T) for Conv1d/U-Net ---
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0,2,1)
    X_val_tensor   = torch.tensor(X_val,   dtype=torch.float32).permute(0,2,1)
    X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32).permute(0,2,1)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

    m_train_tensor = torch.tensor(m_train, dtype=torch.float32)
    m_val_tensor   = torch.tensor(m_val,   dtype=torch.float32)
    m_test_tensor  = torch.tensor(m_test,  dtype=torch.float32)

    # datasets/loaders
    train_ds = TensorDataset(X_train_tensor, y_train_tensor, m_train_tensor)
    val_ds   = TensorDataset(X_val_tensor,   y_val_tensor,   m_val_tensor)
    test_ds  = TensorDataset(X_test_tensor,  y_test_tensor,  m_test_tensor)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  
                              num_workers=config.num_workers, pin_memory=config.pin_memory, 
                              persistent_workers=config.persistent_workers)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, 
                              num_workers=config.num_workers, pin_memory=config.pin_memory, 
                              persistent_workers=config.persistent_workers)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False, 
                              num_workers=config.num_workers, pin_memory=config.pin_memory, 
                              persistent_workers=config.persistent_workers)

    return train_loader, val_loader, test_loader, n_features, features