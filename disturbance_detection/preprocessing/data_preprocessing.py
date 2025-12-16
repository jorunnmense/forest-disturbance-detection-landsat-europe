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

    train_total, train_pos, train_neg, train_ratio = count_pos_neg(train_u)
    val_total, val_pos, val_neg, val_ratio = count_pos_neg(val_u)
    test_total, test_pos, test_neg, test_ratio = count_pos_neg(test_u)

    print("\n[Split summary by pixel-level samples]")
    print(f"Train: total={train_total:,} | pos={train_pos:,} | neg={train_neg:,} | pos%={100*train_ratio:.3f}")
    print(f"Val:   total={val_total:,} | pos={val_pos:,} | neg={val_neg:,} | pos%={100*val_ratio:.3f}")
    print(f"Test:  total={test_total:,} | pos={test_pos:,} | neg={test_neg:,} | pos%={100*test_ratio:.3f}")
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

def to_nan(x):
    x = x.copy()
    x[x == -9999.0] = np.nan
    return x

def norm_per_feature(X_nan, feat_min, feat_rng, F):
    Xn = (X_nan - feat_min.reshape(1,1,F)) / feat_rng.reshape(1,1,F)
    Xn = np.where(np.isnan(Xn), 0.0, Xn)
    return np.clip(Xn, 0.0, 1.0)

    # stack per split
def stack_or_empty(XL, yL, features, window_size):
    if len(XL) == 0:
        X_result = np.empty((0, window_size, len(features)), dtype=float)
        y_result = np.empty((0, window_size), dtype=float)
    else:
        X_result = np.stack(XL)
        y_result = np.stack(yL)
    return X_result, y_result


def get_features_and_splits(df, config):
    """Extract features and create data splits"""
    # --- Get features from config ---
    features = config.get_features()
    n_features = config.get_num_features()
    
    # --- fixed uniqueid splits (create once, then always load) ---
    train_uids, val_uids, test_uids = make_or_load_uid_splits(
        df, config.split_path, config.seed
    )
    train_uids_set, val_uids_set, test_uids_set = set(train_uids), set(val_uids), set(test_uids)
    
    return features, n_features, train_uids_set, val_uids_set, test_uids_set

def create_windowed_data(df, features, config, train_uids_set, val_uids_set, test_uids_set):
    """Create windowed data """
    # buckets
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    def add_sample(uid, X, y):
        if uid in train_uids_set:
            X_train_list.append(X); y_train_list.append(y)
        elif uid in val_uids_set:
            X_val_list.append(X); y_val_list.append(y)
        elif uid in test_uids_set:
            X_test_list.append(X); y_test_list.append(y)

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
            add_sample(uid, seq_x, seq_y)

    return X_train_list, y_train_list, X_val_list, y_val_list, X_test_list, y_test_list

def stack_and_normalize_data(X_train_list, y_train_list, X_val_list, y_val_list, X_test_list, y_test_list, n_features, config):
    """Stack arrays and normalize"""
    X_train, y_train = stack_or_empty(X_train_list, y_train_list, n_features, config.window_size)
    X_val,   y_val  = stack_or_empty(X_val_list, y_val_list, n_features, config.window_size)
    X_test,  y_test  = stack_or_empty(X_test_list, y_test_list, n_features, config.window_size)

    # --- normalization: per-feature min–max using TRAIN valid timesteps only ---
    if X_train.shape[0] == 0:
        raise RuntimeError("Empty training split after windowing. Check filters/window_size.")


    X_train_nan = to_nan(X_train)
    X_val_nan   = to_nan(X_val)
    X_test_nan  = to_nan(X_test)

    n_train, n_timesteps, n_features = X_train_nan.shape
    X_train_nan_flattened = X_train_nan.reshape(-1, n_features)

    feat_min = np.nanmin(X_train_nan_flattened, axis=0, keepdims=True)
    feat_max = np.nanmax(X_train_nan_flattened, axis=0, keepdims=True)
    feat_rng = np.clip(feat_max - feat_min, 1e-8, None)

    X_train = norm_per_feature(X_train_nan, feat_min, feat_rng, n_features	)
    X_val   = norm_per_feature(X_val_nan, feat_min, feat_rng, n_features)
    X_test  = norm_per_feature(X_test_nan, feat_min, feat_rng, n_features)

    # Save normalization stats
    np.save(config.min_train_path, feat_min)
    np.save(config.max_train_path, feat_max)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_tensors_and_loaders(X_train, y_train, X_val, y_val, X_test, y_test, config):
    """Create tensors and DataLoaders"""
    # --- tensors in (B, C, T) for Conv1d/U-Net ---
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0,2,1)
    X_val_tensor   = torch.tensor(X_val,   dtype=torch.float32).permute(0,2,1)
    X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32).permute(0,2,1)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

    # datasets/loaders
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    val_ds   = TensorDataset(X_val_tensor,   y_val_tensor)
    test_ds  = TensorDataset(X_test_tensor,  y_test_tensor)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  
                              num_workers=config.num_workers, pin_memory=config.pin_memory, 
                              persistent_workers=config.persistent_workers)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, 
                              num_workers=config.num_workers, pin_memory=config.pin_memory, 
                              persistent_workers=config.persistent_workers)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False, 
                              num_workers=config.num_workers, pin_memory=config.pin_memory, 
                              persistent_workers=config.persistent_workers)
    
    return train_loader, val_loader, test_loader

def prepare_data(df, config):
    """
    Main data preparation function: windowing, normalization, and DataLoader creation.
    
    Don't fillna(0.0) (keep real missing)
    compute min–max on train valid timesteps only, ignoring NaN and −9999,
    after scaling, turn remaining NaN/−9999 into 0
    
    Args:
        df: DataFrame with features and labels
        config: Config object containing all parameters
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, n_features, used_features)
    """
    # Step 1: Get features and splits
    features, n_features, train_uids_set, val_uids_set, test_uids_set = get_features_and_splits(df, config)
    
    # Step 2: Create windowed data
    X_train_list, y_train_list, X_val_list, y_val_list, X_test_list, y_test_list = create_windowed_data(
        df, features, config, train_uids_set, val_uids_set, test_uids_set
    )
    
    # Step 3: Stack and normalize
    X_train, y_train, X_val, y_val, X_test, y_test = stack_and_normalize_data(
        X_train_list, y_train_list, X_val_list, y_val_list, X_test_list, y_test_list, n_features, config
    )
    
    # Step 4: Create tensors and loaders
    train_loader, val_loader, test_loader = create_tensors_and_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, config
    )

    return train_loader, val_loader, test_loader, n_features, features

