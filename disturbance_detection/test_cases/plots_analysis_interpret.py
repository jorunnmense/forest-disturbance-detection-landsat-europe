def plot_band_importance(table, names):
    tab = sorted(table, key=lambda r: -r["delta_auprc_mean"])
    labels = [names[r["band_index"]] for r in tab]
    vals   = [r["delta_auprc_mean"] for r in tab]
    errs   = [r["delta_auprc_std"]  for r in tab]
    plt.figure(figsize=(5,4))
    plt.bar(labels, vals, yerr=errs)
    plt.ylabel("ΔAUPRC (permute band)")
    plt.title("Band permutation importance")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.show()

def plot_temporal_occlusion(rel, d_all, d_bands=None, band_names=None):
    plt.figure(figsize=(4,3.5))
    plt.plot(rel, d_all, marker="o")
    plt.axhline(0, ls="--", alpha=0.5)
    plt.xlabel("Relative time (target=0)")
    plt.ylabel("Δ logit (base − occluded)")
    plt.title("Temporal sensitivity (occlude all bands)")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    if d_bands is not None:
        plt.figure(figsize=(5, 0.35*(d_bands.shape[0]) + 2))
        im = plt.imshow(d_bands, aspect="auto")
        plt.colorbar(im, shrink=0.8)
        plt.xticks(range(len(rel)), rel)
        if band_names:
            plt.yticks(range(len(band_names)), band_names)
        else:
            plt.yticks(range(d_bands.shape[0]), [f"B{b}" for b in range(d_bands.shape[0])])
        plt.xlabel("Relative time (target=0)")
        plt.title("Per-band temporal sensitivity (Δ logit)")
        plt.tight_layout(); plt.show()

''' note: Window size = 5 → timesteps [0,1,2,3,4]
Target =  last → target_index = 4 (or -1)
Relative years = [-4 , -3, -2, -1, 0]'''
# 1) BAND PERMUTATION on VALIDATION
base_val, table_val, drops_val = band_permutation_importance(
    model, val_loader, device,
    n_repeats=100, base_threshold=None, seed=42, target_index=-1   ## n_repeats=100, 50, 30
)
print("Baseline VAL:", base_val)
plot_band_importance(table_val, used_feats)

# Optional: export table
df_imp = pd.DataFrame(table_val)
df_imp["band"] = [used_feats[i] for i in df_imp["band_index"]]
df_imp.sort_values("delta_auprc_mean", ascending=False).to_csv(
    "band_importance_val_w5_unet_last.csv", index=False
)

# 2) TEMPORAL OCCLUSION on VALIDATION
rel, d_all, d_bands = temporal_occlusion(
    model, val_loader, device, per_band=True, normalize_by_abs_logit=True, target_index=-1
)
plot_temporal_occlusion(rel, d_all, d_bands, band_names=used_feats)

print("Relative indices:", rel)                     # expect [-3, -2, -1, 0, 1]
print("Mean Δlogit (all bands):", np.round(d_all, 4))