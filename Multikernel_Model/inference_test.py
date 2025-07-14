import os
import numpy as np
import torch
import rasterio
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader

from multikernel_model import AlbasUNet

# --- Config ---
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path    = '/home/ubuntu/work/saved_data/landsat_disturbance_detection/1D_U_Net/Multikernel_Model/best_model_2.pth'
dir_path      = '/home/ubuntu/work/saved_data/landsat_disturbance_detection/1D_U_Net/tiles/raw_data'
output_dir    = '/home/ubuntu/work/saved_data/landsat_disturbance_detection/1D_U_Net/Multikernel_Model/predictions'

band_indices  = [0, 1, 2, 3, 4, 5]
n_features    = len(band_indices)
window_size   = 8

# --- Load model ---
model = AlbasUNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# --- Read metadata & load all BAP into memory + normalize ---
years_all = list(range(1984, 2024))
n_years   = len(years_all)

sample_file = os.path.join(dir_path, f"{years_all[0]}0801_LEVEL3_LNDLG_BAP.tif")
with rasterio.open(sample_file) as src:
    H, W      = src.height, src.width
    crs       = src.crs
    transform = src.transform

bap_data = np.full((n_years, H, W, n_features), -9999, dtype=np.float32)
for i, year in enumerate(years_all):
    path = os.path.join(dir_path, f"{year}0801_LEVEL3_LNDLG_BAP.tif")
    if os.path.exists(path):
        with rasterio.open(path) as src:
            b = src.read([bi+1 for bi in band_indices])
            bap_data[i] = b.transpose(1,2,0)
    else:
        print(f"[!] Missing year {year}, filled nodata")
    arr = bap_data[i]
    arr = np.nan_to_num(arr, nan=-9999, posinf=-9999, neginf=-9999)
    for f in range(n_features):
        band = arr[...,f]
        mask = band != -9999
        if mask.any():
            p2,p98 = np.percentile(band[mask],[2,98])
            band = np.clip(band,p2,p98)
            band = np.where(mask,(band-p2)/(p98-p2+1e-8),0)
        arr[...,f] = band
    bap_data[i] = np.clip(arr, 0, 1)
    print(f"Year {year} loaded & normalized")

# --- Dataset that slides only temporally ---
class TemporalWindowDataset(Dataset):
    def __init__(self, bap_data, window_size):
        self.bap = bap_data
        self.w   = window_size
        self.ny, self.H, self.W, _ = bap_data.shape

    def __len__(self):
        return self.ny - self.w + 1

    def __getitem__(self, idx):
        cube = self.bap[idx:idx+self.w]  # (T, H, W, F)
        # reorder to (H*W, F, T)
        cube = np.transpose(cube, (1,2,3,0)).reshape(self.H*self.W, n_features, self.w)
        return torch.from_numpy(cube).float(), idx

dataset = TemporalWindowDataset(bap_data, window_size)
loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Prepare output arrays ---
pred_res = np.zeros((n_years, H, W), dtype=np.float32)
cnt_res  = np.zeros((n_years, H, W), dtype=np.float32)

# --- Inference over temporal windows with pixel‐chunking ---
pixel_batch = 65536  # adjust to fit GPU memory

with torch.no_grad():
    for cubes, start_idx in tqdm(loader, desc="Temporal windows"):
        _, P, F, T = cubes.shape  # P = H*W
        flat = cubes.view(P, F, T)

        out_all = np.zeros((P, T), dtype=np.float32)
        for i in range(0, P, pixel_batch):
            sub = flat[i:i+pixel_batch].to(device)     # (sub_P, F, T)
            pred = model(sub)                          # (sub_P, T)
            pred = torch.sigmoid(pred).cpu().numpy()
            out_all[i:i+pixel_batch] = pred

        out = out_all.reshape(H, W, window_size)
        si = start_idx.item()
        for t in range(window_size):
            year_idx = si + t
            pred_res[year_idx] += out[:,:,t]
            cnt_res [year_idx] += 1

# --- Finalize predictions ---
mask = cnt_res > 0
prob = np.zeros_like(pred_res)
prob[mask] = pred_res[mask] / cnt_res[mask]
binr = (prob > 0.5).astype(np.uint8)

# --- Save per‐year TIFFs ---
os.makedirs(output_dir, exist_ok=True)
for idx, year in enumerate(years_all):
    outp = os.path.join(output_dir, f"{year}_pred_w{window_size}.tif")
    meta = {
        'driver':      'GTiff',
        'height':       H,
        'width':        W,
        'count':        1,
        'dtype':       rasterio.uint8,
        'crs':          crs,
        'transform':    transform,
        'compress':     'lzw',
        'tiled':        True,
        'blockxsize':   256,
        'blockysize':   256,
        'interleave':   'band',
    }
    with rasterio.open(outp, 'w', **meta) as dst:
        dst.write(binr[idx], 1)
        dst.set_band_description(1, f"Prediction {year}")
    print(f"Saved {outp}")
