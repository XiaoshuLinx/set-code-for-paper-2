#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:57:22 2026

@author: linx
Unified Time-resolved MVPA + Topomap-ready coefficients + optional GAT
Per-subject: time-resolved AUC + GAT in one figure
Group-level: mean time AUC + GAT
Epoch: [-1500, 0] ms, 4 ms per sample
Date: 2026-03-19
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore
from scipy.ndimage import gaussian_filter
from mne.decoding import LinearModel
from mne.decoding import get_coef



# ---------------------- PARAMETERS ----------------------
freq = 'alpha' #epoch
data_original = 'sensor'
if data_original == 'sensor':
    data_dir = "/nashome2/linx/Desktop/mindeye/ml_data/sensor/grad"
    subject_list = ['04','05','06','07','08','09','10',
                    '11','12','13','14','15','16','18','19','20',
                    '21','22','23','24','25','26','27','28','29','30',
                    '31','33','34','35','36','37','38','40',
                    '41','42','43','44','45','46','01','02','18','39']

t_keep = 376       # -1500 to 0 ms
sfreq = 250
n_splits = 5
clf_C = 2.0
max_iter = 1000


do_smooth = True
smooth_win = 5
do_gat = True
gat_smooth_sigma = 1.0  # sigma for gaussian_filter

# ---------------------- SAVE PATHS ----------------------
save_root = f"/nashome2/linx/Desktop/mindeye/ml_data/time_resolved_mvpa_{freq}_{data_original}_coef_topomap_a"
os.makedirs(save_root, exist_ok=True)
per_subj_dir = os.path.join(save_root, "per_subject")
os.makedirs(per_subj_dir, exist_ok=True)

# ---------------------- HELPER FUNCTIONS ----------------------
def create_dataset_epoch(subject):
    f_dim = os.path.join(data_dir, f"{subject}_dimension_{freq}_{data_original}.npz")
    f_obj = os.path.join(data_dir, f"{subject}_object_{freq}_{data_original}.npz")
    x0 = np.load(f_dim)["data"][:, :, :t_keep]
    x1 = np.load(f_obj)["data"][:, :, :t_keep]
    y0 = np.zeros(x0.shape[0], dtype=int)
    y1 = np.ones(x1.shape[0], dtype=int)
    X = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([y0, y1], axis=0)
    print(f"  Loaded {subject}: dim {x0.shape}, obj {x1.shape}")
    return X, y

def shuffle_dataset(X, y, subject_seed):
    rng = np.random.RandomState(int(subject_seed))  # ensure subject string -> int
    idx = rng.permutation(y.size)
    return X[idx], y[idx]

def smooth_time_series_edge(arr, window=smooth_win):
    if window < 2:
        return arr
    kernel = np.ones(window)/window
    pad = window // 2
    if arr.ndim == 1:
        padded = np.pad(arr, (pad, pad), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
    else:
        smoothed = np.zeros_like(arr, dtype=float)
        for i, trial in enumerate(arr):
            padded = np.pad(trial, (pad, pad), mode='edge')
            smoothed[i] = np.convolve(padded, kernel, mode='valid')
    return smoothed

# ---------------------- TIME VECTOR ----------------------
times = np.linspace(-1500, 0, t_keep)  # ms

# ---------------------- READ SUBJECTS ----------------------
data_files = os.listdir(data_dir)
subjects = sorted(list({f.split("_")[0] for f in data_files if f.endswith(".npz")}))
def has_epoch_files(subj):
    f_dim = os.path.join(data_dir, f"{subj}_dimension_{freq}_{data_original}.npz")
    f_obj = os.path.join(data_dir, f"{subj}_object_{freq}_{data_original}.npz")
    return os.path.isfile(f_dim) and os.path.isfile(f_obj)
subjects = [s for s in subjects if has_epoch_files(s)]
if subject_list:
    subjects = [s for s in subjects if s in subject_list]
print(f"Subjects to run: {len(subjects)} -> {subjects}\n")

# ---------------------- MAIN LOOP ----------------------
all_subject_auc = []
all_subject_gat = []

for subject in subjects:
    try:
        X, y = create_dataset_epoch(subject)
    except Exception as e:
        print(f"Skipping {subject}: failed to load data ({e})")
        continue

    if np.isnan(X).any():
        print(f"⚠️ NaNs found in {subject}, replacing with 0")
        X = np.nan_to_num(X)

   # X, y = shuffle_dataset(X, y, subject)
    n_trials, n_features, n_times = X.shape
    print(f"Subject {subject}: trials={n_trials}, features={n_features}, times={n_times}")

    pipeline = make_pipeline(
        StandardScaler(),
        LinearModel(LogisticRegression(C=clf_C, solver="liblinear", max_iter=max_iter))
)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # --------- SlidingEstimator for CV-average coefficients ---------
    auc_scores = []
    all_patterns = []
    decision_all = np.zeros((n_trials, n_times))

    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        est = clone(pipeline)
        time_estimator = SlidingEstimator(est, scoring='roc_auc', n_jobs=-1)
        time_estimator.fit(X_train, y_train)

        patterns = get_coef(time_estimator, 'patterns_', inverse_transform=True)
        all_patterns.append(patterns)
 
        scores_fold = time_estimator.score(X_test, y_test)
        auc_scores.append(scores_fold)        

    mean_auc = np.mean(np.stack(auc_scores, axis=0), axis=0)

    # save per-subject
    all_patterns = np.stack(all_patterns, axis=0)  # shape: (n_folds, n_times, n_features)
    np.save(os.path.join(per_subj_dir, f"{subject}_{freq}_all_patterns.npy"), all_patterns)
    np.save(os.path.join(per_subj_dir, f"{subject}_{freq}_time_auc.npy"), mean_auc)
    all_subject_auc.append(mean_auc)
    
    # =========================
    # 2. FULL DATA: pattern
    # =========================
    print("  Fitting full-data model for patterns...")
    est_full = clone(pipeline)
    time_estimator_full = SlidingEstimator(est_full, scoring='roc_auc', n_jobs=-1)
    time_estimator_full.fit(X, y)

    patterns = get_coef(time_estimator_full, 'patterns_', inverse_transform=True)
    np.save(os.path.join(per_subj_dir, f"{subject}_{freq}_patterns_full.npy"), patterns)
    # --------- Optional GAT ---------
    if do_gat:
        gat = GeneralizingEstimator(pipeline, scoring='roc_auc', n_jobs=-1)
        scores_gat = cross_val_multiscore(gat, X, y, cv=skf, n_jobs=-1)
        mean_gat = np.mean(scores_gat, axis=0)
        all_subject_gat.append(mean_gat)
        np.save(os.path.join(per_subj_dir, f"{subject}_{freq}_gat.npy"), mean_gat)

    # --------- Per-subject plotting: AUC + GAT ---------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Time-resolved AUC
    plot_auc = mean_auc.copy()
    if do_smooth and smooth_win > 1:
        plot_auc = smooth_time_series_edge(plot_auc, smooth_win)
    ax[0].plot(times, plot_auc, lw=2)
    ax[0].axhline(0.5, color='gray', ls='--')
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("ROC-AUC")
    ax[0].set_title(f"{subject} - Time-resolved AUC (smoothed={do_smooth})")

    # GAT heatmap
    if do_gat:
        gat_plot = mean_gat.copy()
        if gat_smooth_sigma > 0:
            gat_plot = gaussian_filter(gat_plot, sigma=gat_smooth_sigma)
        center = 0.5
        max_dev = np.max(np.abs(mean_gat - center))
        vmin_val = center - max_dev
        vmax_val = center + max_dev
        im = ax[1].imshow(
            gat_plot,
            origin="lower",
            aspect="auto",
            extent=[times[0], times[-1], times[0], times[-1]],
            vmin=vmin_val,
            vmax=vmax_val,
            cmap="RdBu_r"
        )
        ax[1].set_xlabel("Test time (ms)")
        ax[1].set_ylabel("Train time (ms)")
        ax[1].set_title(f"{subject} - Temporal generalization (AUC)")
        plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)

    plt.tight_layout()
    fig_path = os.path.join(per_subj_dir, f"{subject}_{freq}_time_and_gat.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Saved figure -> {fig_path}")

print("\nALL DONE. CV-average coefficients, AUC, and GAT figures saved to:", save_root)
