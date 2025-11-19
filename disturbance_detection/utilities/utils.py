"""
Utility functions for reproducibility and dataset inspection
"""
import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """
    Set seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): Seed value to use. Default is 42.
    """
    # Set Python seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set PyTorch seed
    torch.manual_seed(seed)
    
    # Set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        # Additional CUDA settings for reproducibility
        torch.backends.cudnn.deterministic = True
        
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")


def print_split_balances(df, train_uids, val_uids, test_uids):
    """
    Print detailed statistics about train/val/test split balance.
    Shows both pixel-level and uniqueid-level class distribution.
    
    Args:
        df: DataFrame with columns 'uniqueid' and 'new_class2_v5'
        train_uids: Array of training unique IDs
        val_uids: Array of validation unique IDs
        test_uids: Array of test unique IDs
    """
    def count_pos_neg(uids_subset):
        sub = df[df["uniqueid"].isin(uids_subset)]
        total = len(sub)
        pos = int(sub["new_class2_v5"].sum())
        neg = total - pos
        return total, pos, neg, pos / max(total, 1)

    tr_total, tr_pos, tr_neg, tr_ratio = count_pos_neg(train_uids)
    va_total, va_pos, va_neg, va_ratio = count_pos_neg(val_uids)
    te_total, te_pos, te_neg, te_ratio = count_pos_neg(test_uids)

    print("\n[Split summary by pixel-level samples]")
    print(f"Train: total={tr_total:,} | pos={tr_pos:,} | neg={tr_neg:,} | pos%={100*tr_ratio:.3f}")
    print(f"Val:   total={va_total:,} | pos={va_pos:,} | neg={va_neg:,} | pos%={100*va_ratio:.3f}")
    print(f"Test:  total={te_total:,} | pos={te_pos:,} | neg={te_neg:,} | pos%={100*te_ratio:.3f}")
    print("-----------------------------------------------------------")

    # uniqueid-level balance
    uid_groups = df.groupby("uniqueid")["new_class2_v5"]
    uids = uid_groups.count().index.values
    has_pos = (uid_groups.max().values > 0).astype(int)
    pos_map = dict(zip(uids, has_pos))

    def uid_pos_ratio(uids_subset):
        labels = np.array([pos_map[u] for u in uids_subset])
        return labels.sum(), len(labels) - labels.sum(), labels.mean()

    tr_uid_pos, _, tr_uid_ratio = uid_pos_ratio(train_uids)
    va_uid_pos, _, va_uid_ratio = uid_pos_ratio(val_uids)
    te_uid_pos, _, te_uid_ratio = uid_pos_ratio(test_uids)

    print("[Split summary by uniqueid]")
    print(f"Train: total={len(train_uids):,} | disturbed_uids={tr_uid_pos:,} | pos%={100*tr_uid_ratio:.3f}")
    print(f"Val:   total={len(val_uids):,} | disturbed_uids={va_uid_pos:,} | pos%={100*va_uid_ratio:.3f}")
    print(f"Test:  total={len(test_uids):,} | disturbed_uids={te_uid_pos:,} | pos%={100*te_uid_ratio:.3f}")
    print("===========================================================")