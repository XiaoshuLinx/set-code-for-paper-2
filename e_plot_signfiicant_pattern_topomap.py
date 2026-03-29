#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:07:20 2026

@author: linx
"""

import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from mne.stats import fdr_correction
# ---------------------- PATH ----------------------
freq = 'alpha' #"epoch"
data_original = 'sensor'

per_subj_dir = f"/nashome2/linx/Desktop/mindeye/ml_data/time_resolved_mvpa_{freq}_{data_original}_coef_topomap_a/per_subject"
save_dir = f"/nashome2/linx/Desktop/mindeye/ml_data/time_resolved_mvpa_{freq}_{data_original}_coef_topomap_a"
os.makedirs(save_dir, exist_ok=True)

raw_fif = "/projects/mindeye/MRI/wiica/04_ica_di_raw.fif"

# ---------------------- LOAD INFO ----------------------
raw = mne.io.read_raw_fif(raw_fif, preload=False)
picks = mne.pick_types(raw.info, meg='grad')
info = mne.pick_info(raw.info, picks)

# ---------------------- TIME ----------------------
tmin, tmax = -1500, 0
n_times = 376
times = np.linspace(tmin, tmax, n_times)

# ---------------------- LOAD PATTERN ----------------------
subject_files = sorted([f for f in os.listdir(per_subj_dir) if f.endswith('patterns_full.npy')])

all_subject_patterns = []
for f in subject_files:
    p = np.load(os.path.join(per_subj_dir, f))  # (channels, times)
    all_subject_patterns.append(p)

X = np.stack(all_subject_patterns, axis=0)  # (subjects, channels, times)
X = np.transpose(X, (0, 2, 1))  # -> (subjects, times, channels)



# ---------------------- YOUR TIME WINDOWS ----------------------
time_windows = [
    (-480, 0),#alpha
   # (-932, -200),#epoch
]

# ---------------------- LOOP WINDOWS ----------------------
for i_win, (t_start, t_end) in enumerate(time_windows):

    time_inds = np.where((times >= t_start) & (times <= t_end))[0]

    # 平均时间
    data_win = X[:, time_inds, :].mean(axis=1)  # (subjects, channels)

    # ---------------------- T-TEST ----------------------
    t_vals, p_vals = ttest_1samp(data_win, 0, axis=0)
    reject, p_fdr = fdr_correction(p_vals, alpha=0.05)
    sig_mask = reject

    #vmax = np.percentile(np.abs(t_vals), 95)
    # mask 非显著
    t_plot = t_vals.copy()
    t_plot[~sig_mask] = 0

    # center（可选）
    #t_plot = t_plot - np.mean(t_plot)

    vmax = np.max(np.abs(t_plot))

    # ---------------------- PLOT ----------------------
    fig, ax = plt.subplots(figsize=(8,8))
    im, _=mne.viz.plot_topomap(
        t_plot,
        info,
        axes=ax,
        cmap='RdBu_r',
        vlim=(-vmax, vmax),
        #contours=0,
        sphere=0.09,
        show=False
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title("t-value topography", fontsize=20)

    fname = os.path.join(save_dir, f"{i_win}_{int(t_start)}_{int(t_end)}ms.png")
    plt.savefig(fname, dpi=300)
    plt.close()

    print(f"Saved: {fname}")
