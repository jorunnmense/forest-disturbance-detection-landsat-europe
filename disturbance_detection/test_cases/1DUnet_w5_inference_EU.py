####################################################################################################3
# inference using Landsat BAP across all europe, using tile system
# 1985 to 2024
# inference only within forest land areas to speed up 
# @aviana 
# modified 02/11/2025

import os
import time
import torch
import numpy as np
import rasterio
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.warp import reproject, Resampling  ### in case forest mask is non aligned, match CRS

# --- Config ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TemporalDropout(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
    def forward(self, x):
        if not self.training or self.dropout_rate == 0.0:
            return x
        B, C, T = x.shape
        mask = (torch.rand(B, T, device=x.device) > self.dropout_rate).float().unsqueeze(1)
        return x * mask

# 2) MultiKernelConv1d with selectable norm + kernel set
# norm='bn' for BatchNorm (your current default), norm='ln' to get LayerNorm-like via GroupNorm(1, C), or norm='gn8' for GroupNorm(8).
# kernel_sizes selectable (use (1,3,5) for small T).
class MultiKernelConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop=0.2,
                 kernel_sizes=(1,3,5), norm='bn'):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            layers = [nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2)]
            if norm == 'bn':
                layers += [nn.BatchNorm1d(out_channels)]
            elif norm == 'ln':
                layers += [nn.GroupNorm(1, out_channels)]     # LN-like over channels
            elif norm.startswith('gn'):
                g = int(norm[2:]) if norm[2:].isdigit() else 8
                layers += [nn.GroupNorm(g, out_channels)]
            else:
                raise ValueError("norm must be 'bn', 'ln', or 'gnK'")
            layers += [nn.ReLU(inplace=True)]
            self.branches.append(nn.Sequential(*layers))
        self.dropout = nn.Dropout1d(p_drop)

    def forward(self, x):
        y = torch.cat([b(x) for b in self.branches], dim=1)  # (B, len(ks)*out_channels, T)
        return self.dropout(y)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2, reduction='none'):   ### or 2.5
        super().__init__()
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits, same shape as targets (e.g., B x T)
        # targets: {0,1} float tensor
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt  = torch.exp(-bce)  # prob of the true class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce

        if self.reduction == 'none':
            return loss                 # per-element (B x T)
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        # x: (batch, channels, time)
        x = x.permute(0, 2, 1)  # -> (batch, time, channels)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_weights = torch.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        attended = attn_weights @ V

        return attended.permute(0, 2, 1)  # -> (batch, channels, time)
        
# SmallUNet1D updated
# Adds TemporalDropout at the input.
# Lets you pick norm and kernel set once, then reuses everywhere.
# Still uses interpolate upsampling (good for tiny/odd T).

class SmallUNet1D(nn.Module):
    def __init__(self, in_channels, base=16, p_drop=0.2,
                 norm='ln',                 # ← default to LN-like for small batches
                 kernel_sizes_small=(1,3,5),
                 kernel_sizes_big=(3,5,7),
                 tdrop_rate=0.0):
        super().__init__()
        self.tdrop = TemporalDropout(tdrop_rate)  ### set to 0 because: On small temporal windows, it throws away too much 
                                                    #information and hurts performance more than it regularizes.

        # choose kernels once (you can also make this conditional on expected T)
        ks1 = kernel_sizes_small   # for first two levels with small T setups
        ks2 = kernel_sizes_small

        # ---- Encoder ----
        self.enc1 = MultiKernelConv1d(in_channels, base, p_drop=p_drop, kernel_sizes=ks1, norm=norm)   # -> (B, 3*base, T)
        self.pool1 = nn.AvgPool1d(2, ceil_mode=True)

        self.enc2 = MultiKernelConv1d(3*base, 2*base, p_drop=p_drop, kernel_sizes=ks2, norm=norm)      # -> (B, 3*2*base, T/2)
        self.pool2 = nn.AvgPool1d(2, ceil_mode=True)

        # ---- Bottleneck ----
        bottleneck_ch = 4*base
        self.bn_conv = nn.Sequential(
            nn.Conv1d(6*base, bottleneck_ch, 1),
            nn.GroupNorm(1, bottleneck_ch) if norm=='ln' else
            (nn.GroupNorm(8, bottleneck_ch) if norm.startswith('gn') else nn.BatchNorm1d(bottleneck_ch)),
            nn.ReLU(inplace=True),
        )
        self.attn = TemporalSelfAttention(embed_dim=bottleneck_ch)

        # ---- Decoder ----
        self.dec2_reduce = nn.Sequential(
            nn.Conv1d(bottleneck_ch + 6*base, 6*base, 1),
            nn.GroupNorm(1, 6*base) if norm=='ln' else
            (nn.GroupNorm(8, 6*base) if norm.startswith('gn') else nn.BatchNorm1d(6*base)),
            nn.ReLU(inplace=True),
        )
        self.dec1_reduce = nn.Sequential(
            nn.Conv1d(6*base + 3*base, 3*base, 1),
            nn.GroupNorm(1, 3*base) if norm=='ln' else
            (nn.GroupNorm(8, 3*base) if norm.startswith('gn') else nn.BatchNorm1d(3*base)),
            nn.ReLU(inplace=True),
        )

        # ---- Head ----
        self.out_conv = nn.Conv1d(3*base, 1, 1)

    def forward(self, x):
        x = self.tdrop(x)  # temporal dropout at input
        x1f = self.enc1(x);  x1  = self.pool1(x1f)
        x2f = self.enc2(x1); x2  = self.pool2(x2f)

        xb = self.bn_conv(x2)
        xb = self.attn(xb)

        y  = F.interpolate(xb, size=x2f.size(2), mode='linear', align_corners=False)
        y  = self.dec2_reduce(torch.cat([y, x2f], dim=1))

        y  = F.interpolate(y, size=x1f.size(2), mode='linear', align_corners=False)
        y  = self.dec1_reduce(torch.cat([y, x1f], dim=1))

        return self.out_conv(y).squeeze(1)  # (B, T)

# --- Paths ---
model_path = '/mnt/dss_project/.../AI4Forest/models_unet/1dunet_w5_sectolast_bands_v6.pth'
dir_path = '/mnt/dss_europe/level3_interpolated/'
txt_file_path = '/mnt/dss_europe/param/level4/level4_disturbmapping_queue.txt'
output_dir = '/mnt/dss_europe/level4_disturbance/AI_disturbance/all/w5_v6/.../'
mask_root = '/mnt/dss_europe/level4_forest'

# --- Features / window config ---
ibap_bands_indices = [0,1,2,3,4,5]
n_features = len(ibap_bands_indices)  # 6  #+ len(features_sep)  # 12
window_size = 5
max_window = 8  # MUST match training ? check that

# --- Model ---
model = SmallUNet1D(in_channels=n_features)
state = torch.load(model_path, map_location="cpu")
model.load_state_dict(state)
model.to(device).eval()
# (skip TorchScript for now; add later if desired AFTER load)

# --- Queue ---
with open(txt_file_path, 'r') as f:
    subfolders_to_process = [line.strip() for line in f.readlines()]

existing_subfolders = [sub for sub in subfolders_to_process if os.path.isdir(os.path.join(dir_path, sub))]
missing_subfolders = set(subfolders_to_process) - set(existing_subfolders)
if missing_subfolders:
    print("Warning: missing subfolders under dir_path:")
    for m in missing_subfolders: print(" -", m)

dummy_array = None
common_int = "0801_LEVEL3_LNDLG"

for subdir in existing_subfolders:
    tilepath = os.path.join(dir_path, subdir)
    print(f"\n>>> Processing folder: {tilepath}")
        
    # Sample raster to get metadata – use any IBAP file
    sample_raster_path = None

    # try 1984 IBAP first
    candidate = os.path.join(tilepath, f"1984{common_int}_IBAP.tif")
    if os.path.exists(candidate):
        sample_raster_path = candidate
    else:
        # fall back: search any year with an IBAP file
        for year in range(1984, 2025):
            candidate = os.path.join(tilepath, f"{year}{common_int}_IBAP.tif")
            if os.path.exists(candidate):
                sample_raster_path = candidate
                break

    if sample_raster_path is None:
        print(f"No IBAP sample raster found for {subdir}, skipping.")
        continue


    with rasterio.open(sample_raster_path) as src:
        height, width = src.height, src.width
        crs = src.crs; transform = src.transform; dtype = src.dtypes[0]

    if dummy_array is None:
        dummy_array = np.full((height, width), -9999, dtype=dtype)

    # Forest mask (reproject if needed)
    mask_path = os.path.join(mask_root, subdir, "forest_landuse_mask.tif")
    if not os.path.exists(mask_path):
        print(f"Forest mask not found: {mask_path}. Skipping tile."); continue

    with rasterio.open(mask_path) as msrc:
        if (msrc.height == height) and (msrc.width == width) and (msrc.crs == crs) and (msrc.transform == transform):
            forest_mask = msrc.read(1)
        else:
            src_mask = msrc.read(1)
            dest = np.zeros((height, width), dtype=src_mask.dtype)
            reproject(source=src_mask, destination=dest,
                      src_transform=msrc.transform, src_crs=msrc.crs,
                      dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.nearest)
            forest_mask = dest
    forest_mask = (forest_mask > 0).astype(np.uint8)
    num_forest = int(forest_mask.sum())
    if num_forest == 0:
        print(f"No forest pixels in {subdir}. Skipping tile."); continue
    forest_idx_flat = np.where(forest_mask.ravel() == 1)[0]

    for target_year in range(1985, 2025):
        print(f"Processing year {target_year}")
        out_file = os.path.join(output_dir, subdir, f"{target_year}_disturbed_undisturbed_pred_unet_w5_prob48_v6.tif")
        if os.path.exists(out_file):
            print(f"Output exists for {target_year}, skipping."); continue

        # Build (N_forest, F, T=window_size)
        X = np.zeros((num_forest, n_features, window_size), dtype=np.float32)

            
        for w in range(window_size):
            year = target_year - (window_size - 1 - w)
            year_features = []

            # IBAP bands only
            if year < 1984:
                # no data before 1984: fill all IBAP bands with nodata
                for _ in ibap_bands_indices:
                    year_features.append(dummy_array.ravel()[forest_idx_flat])
            else:
                ibap_path = os.path.join(tilepath, f"{year}{common_int}_IBAP.tif")
                if os.path.exists(ibap_path):
                    with rasterio.open(ibap_path) as src_ibap:
                        for band_idx in ibap_bands_indices:
                            try:
                                band_data = src_ibap.read(band_idx + 1)
                            except Exception:
                                band_data = dummy_array
                            year_features.append(band_data.ravel()[forest_idx_flat])
                else:
                    print(f"Missing IBAP for {year}, filling IBAP bands with nodata")
                    for _ in ibap_bands_indices:
                        year_features.append(dummy_array.ravel()[forest_idx_flat])

            # year_features length must equal n_features
            X[:, :, w] = np.stack(year_features, axis=1)


        # Replace NaNs/Infs
        X = np.nan_to_num(X, nan=-9999, posinf=-9999, neginf=-9999)

        # ---- PAD time to max_window (right-pad with zeros) ----
        if window_size < max_window:
            pad_t = max_window - window_size
            X = np.pad(X, pad_width=((0,0),(0,0),(0,pad_t)),
                       mode='constant', constant_values=0.0)
        # X is now (N_forest, F, max_window)

        # ---- Normalize with training stats (broadcast-safe) ----
        min_train = np.load("/mnt/dss_project/aviana/AI4Forest/models_unet/min_train_w5_sectolast_bands_v6_results.npy")
        max_train = np.load("/mnt/dss_project/aviana/AI4Forest/models_unet/max_train_w5_sectolast_bands_v6_results.npy")
        rng_train = np.clip(max_train - min_train, 1e-8, None)
        # reshape to (1, F, 1) for broadcasting over (N, F, T)
        min_train = min_train.reshape(1, n_features, 1)
        rng_train = rng_train.reshape(1, n_features, 1)

        input_data = (X - min_train) / rng_train
        input_data = np.clip(input_data, 0, 1)

        input_tensor = torch.tensor(input_data, dtype=torch.float32, device=device).contiguous()

        batch_size = 50000  # adjust for RAM memory
        preds_masked = []
        best_thr = 0.48  # TODO: replace with your tuned threshold from validation

        with torch.no_grad():
            for i in range(0, num_forest, batch_size):
                batch = input_tensor[i:i+batch_size]              # (B, F, max_window)
                logits = model(batch)                              # (B, max_window)
                probs_last_valid = torch.sigmoid(logits[:, window_size - 1])  # (B,)
                preds_np = (probs_last_valid > best_thr).cpu().numpy().astype(np.uint8)
                preds_masked.append(preds_np)

        preds_masked = np.concatenate(preds_masked, axis=0)

        # scatter back to full raster
        predictions_raster = np.zeros((height * width,), dtype=np.uint8)
        predictions_raster[forest_idx_flat] = preds_masked
        predictions_raster = predictions_raster.reshape(height, width)

        # Save
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        with rasterio.open(
            out_file, 'w', driver='GTiff',
            height=height, width=width, count=1, dtype=rasterio.uint8,
            crs=crs, transform=transform, compress='lzw'
        ) as dst:
            dst.write(predictions_raster, 1)

        print(f"Saved prediction for year {target_year} in {out_file}")

