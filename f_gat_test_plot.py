#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:27:15 2026

@author: linx
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 20
})

# ---------------- PATH ----------------
freq = "epoch" #'epoch'
data_original = 'sensor'
root = f"/nashome2/linx/Desktop/mindeye/ml_data/time_resolved_mvpa_{freq}_{data_original}_coef_topomap_a"
gat_dir = os.path.join(root, "per_subject")

# ---------------- PARAMETERS ----------------
t_keep = 376
times = np.linspace(-1500, 0, t_keep)

baseline = (-1500, -1000)

do_smooth = True
smooth_sigma = 1.0

do_stats = True

stat_window_chance = (-1500, 0)
stat_window_baseline = (-1000, 0)

correction_method = "fdr"
alpha = 0.05
chance_level = 0.5

# ---------------- LOAD DATA ----------------
files = sorted([f for f in os.listdir(gat_dir) if f.endswith("_gat.npy")])
print("Found subjects:", len(files))

all_gat = np.stack([np.load(os.path.join(gat_dir, f)) for f in files], axis=0)
print("Data shape:", all_gat.shape)

# ---------------- BASELINE INDEX ----------------
baseline_idx = np.where((times >= baseline[0]) & (times <= baseline[1]))[0]

# ---------------- RAW GROUP MEAN ----------------
group_mean_gat = all_gat.mean(axis=0)

gat_plot = group_mean_gat.copy()
if do_smooth:
    gat_plot = gaussian_filter(gat_plot, sigma=smooth_sigma)

# ---------------- COLOR SCALE (CENTERED AT BASELINE) ----------------
baseline_mean = group_mean_gat[np.ix_(baseline_idx, baseline_idx)].mean()

max_dev = np.max(np.abs(group_mean_gat - baseline_mean))
vmin_val = baseline_mean - max_dev
vmax_val = baseline_mean + max_dev

print("Baseline mean:", baseline_mean)
print("Color scale:", vmin_val, vmax_val)

# ---------------- BASELINE-CORRECTED DATA (FOR STATS ONLY) ----------------
baseline_gat = all_gat[:, baseline_idx][:, :, baseline_idx]
subject_baselines = baseline_gat.mean(axis=(1,2), keepdims=True)
all_gat_baselined = all_gat - subject_baselines

# ---------------- STATISTICS ----------------
mask_chance = None
mask_baseline = None

if do_stats:
    # indices
    TW_start_c = np.searchsorted(times, stat_window_chance[0])
    TW_end_c = np.searchsorted(times, stat_window_chance[1], side='right')

    TW_start_b = np.searchsorted(times, stat_window_baseline[0])
    TW_end_b = np.searchsorted(times, stat_window_baseline[1], side='right')

    # ======================================================
    # 1) vs chance (raw AUC)
    # ======================================================
    t_stat_c, p_c = ttest_1samp(
        all_gat[:, TW_start_c:TW_end_c, TW_start_c:TW_end_c],
        popmean=chance_level,
        axis=0
    )

    if correction_method == "fdr":
        p_flat = p_c.ravel()
        _, p_corr = fdrcorrection(p_flat, alpha=alpha)
        p_c = p_corr.reshape(p_c.shape)

    mask_chance = p_c < alpha

    # ======================================================
    # 2) vs baseline (baseline-corrected)
    # ======================================================
    t_stat_b, p_two = ttest_1samp(
        all_gat_baselined[:, TW_start_b:TW_end_b, TW_start_b:TW_end_b],
        popmean=0.0,
        axis=0
    )

    # one-tailed (greater than baseline)
    p_b = np.where(
        t_stat_b > 0,
        p_two / 2,
        1 - (p_two / 2)
    )

    if correction_method == "fdr":
        p_flat = p_b.ravel()
        _, p_corr = fdrcorrection(p_flat, alpha=alpha)
        p_b = p_corr.reshape(p_b.shape)

    mask_baseline = p_b < alpha

# ============================
# SUMMARY PRINT
# ============================

print("\n============================")
print("GAT STATISTICS SUMMARY")
print("============================")

# --- chance ---
sig_chance = np.sum(mask_chance) if mask_chance is not None else 0
print(f"\n[VS CHANCE]")
print(f"Significant time-time points: {sig_chance}")
print(f"FDR alpha: {alpha}")

# extract coordinates of significant pixels
if mask_chance is not None:
    coords = np.where(mask_chance)
    if len(coords[0]) > 0:
        t_train_min = times[TW_start_c + coords[0].min()]
        t_train_max = times[TW_start_c + coords[0].max()]
        t_test_min  = times[TW_start_c + coords[1].min()]
        t_test_max  = times[TW_start_c + coords[1].max()]

        print(f"Training time range: {t_train_min:.0f} to {t_train_max:.0f} ms")
        print(f"Testing time range: {t_test_min:.0f} to {t_test_max:.0f} ms")

# --- baseline ---
sig_baseline = np.sum(mask_baseline) if mask_baseline is not None else 0
print("\n[VS BASELINE]")
print(f"Significant time-time points: {sig_baseline}")
print(f"FDR alpha: {alpha}")

if mask_baseline is not None:
    coords = np.where(mask_baseline)
    if len(coords[0]) > 0:
        t_train_min = times[TW_start_b + coords[0].min()]
        t_train_max = times[TW_start_b + coords[0].max()]
        t_test_min  = times[TW_start_b + coords[1].min()]
        t_test_max  = times[TW_start_b + coords[1].max()]

        print(f"Training time range: {t_train_min:.0f} to {t_train_max:.0f} ms")
        print(f"Testing time range: {t_test_min:.0f} to {t_test_max:.0f} ms")

# --- peak decoding ---
peak_idx = np.unravel_index(np.argmax(group_mean_gat), group_mean_gat.shape)
peak_value = group_mean_gat[peak_idx]

print("\n[PEAK DECODING]")
print(f"Peak AUC = {peak_value:.4f}")
print(f"Occurs at train={times[peak_idx[0]]:.0f} ms, test={times[peak_idx[1]]:.0f} ms")


# ---------------- PLOT ----------------
plt.figure(figsize=(6,5))

im = plt.imshow(
    gat_plot,
    origin='lower',
    aspect='auto',
    extent=[times[0], times[-1], times[0], times[-1]],
    vmin=vmin_val,
    vmax=vmax_val,
    cmap='RdBu_r'
)

plt.xlabel("Test time (ms)")
plt.ylabel("Train time (ms)")
plt.title("Temporal Generalization Matrix\n(Baseline-centered)", pad=15)

ticks = np.arange(-1500, 1, 250)
plt.xticks(ticks)
plt.yticks(ticks)

# ---------------- CONTOURS ----------------
# vs chance (black)
if mask_chance is not None:
    full_mask_c = np.zeros_like(group_mean_gat, dtype=bool)
    full_mask_c[TW_start_c:TW_end_c, TW_start_c:TW_end_c] = mask_chance

    plt.contour(
        times, times, full_mask_c,
        levels=[0.5],
        colors='black',
        linewidths=1.5,
        alpha=0.5
    )

# vs baseline (green)
if mask_baseline is not None:
    full_mask_b = np.zeros_like(group_mean_gat, dtype=bool)
    full_mask_b[TW_start_b:TW_end_b, TW_start_b:TW_end_b] = mask_baseline

    plt.contour(
        times, times, full_mask_b,
        levels=[0.5],
        colors='green',
        linewidths=1.5,
        alpha=0.5
    )

# ---------------- COLORBAR ----------------
cbar = plt.colorbar(im, fraction=0.046, pad=0.03)
cbar.set_label("AUC")

# ---------------- SAVE ----------------
save_path = os.path.join(root, "group_gat_raw_baseline_centered_dual_stats.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print("Saved to:", save_path)
