#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:36:40 2026

@author: linx
Group-level full-pattern topomap with global color scale
For patterns shape: (n_features, n_times)
Date: 2026-03-20
"""
import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# ----------------------setting ----------------------
freq = 'alpha' #"epoch"
data_original = 'sensor'
per_subj_dir = f"/nashome2/linx/Desktop/mindeye/ml_data/time_resolved_mvpa_{freq}_{data_original}_coef_topomap_a/per_subject"
save_dir = f"/nashome2/linx/Desktop/mindeye/ml_data/time_resolved_mvpa_{freq}_{data_original}_coef_topomap_a/pattern"

raw_fif = "/projects/mindeye/MRI/wiica/04_ica_di_raw.fif"  # 任意一个subject的raw info即可
os.makedirs(save_dir, exist_ok=True)

# ---------------------- load raw info ----------------------
print("Loading raw info from:", raw_fif)
raw = mne.io.read_raw_fif(raw_fif, preload=False)
raw.pick_types(meg=True, eeg=False, eog=False, stim=False)
raw.pick_types(meg='grad')
info = raw.info
n_features = len(info['ch_names'])

# ---------------------- time paramether ----------------------
tmin = -1500
tmax = 0
n_times = 376
times = np.linspace(tmin, tmax, n_times)

# ----------------------load subject full patterns ----------------------
subject_files = sorted([f for f in os.listdir(per_subj_dir) if f.endswith('patterns_full.npy')])
all_subject_patterns = []
for f in subject_files:
    p = np.load(os.path.join(per_subj_dir, f))  # shape (n_features, n_times)
    assert p.shape[0] == n_features, f"{f} feature number mismatch"
    all_subject_patterns.append(p)
all_subject_patterns = np.stack(all_subject_patterns, axis=0)  # shape (n_subjects, n_features, n_times)
print(f"Loaded patterns: {all_subject_patterns.shape}")

# ----------------------  group mean ----------------------
group_mean_patterns = all_subject_patterns.mean(axis=0)  # shape (n_features, n_times)

# ---------------------- time window for topomap ----------------------
time_windows = [
    (-1500,-1250), (-1250,-1000), (-1000,-750), (-750,-500), 
    (-500,-250), (-250,0)
]

# ---------------------- claculate max ----------------------
all_patterns_windowed = []
for t_start, t_end in time_windows:
    idx_start = np.argmin(np.abs(times - t_start))
    idx_end = np.argmin(np.abs(times - t_end)) + 1
    pattern_window = group_mean_patterns[:, idx_start:idx_end].mean(axis=1)
    pattern_window_centered = pattern_window - np.mean(pattern_window)  # center
    all_patterns_windowed.append(pattern_window_centered)
all_patterns_windowed = np.array(all_patterns_windowed)  # shape (n_windows, n_features)
global_vmax = np.max(np.abs(all_patterns_windowed))  #  color scale
print("Global max abs value for color scale:", global_vmax)

# ---------------------- save each time window ----------------------
for (t_start, t_end), pattern_window_centered in zip(time_windows, all_patterns_windowed):
    fig, ax = plt.subplots()
    mne.viz.plot_topomap(
        pattern_window_centered, info, axes=ax, show=False,
        cmap='RdBu_r', vlim=(-global_vmax, global_vmax)
    )
    ax.set_title(f"{t_start}-{t_end} ms", fontsize=16, pad=-200)
    fname_fig = os.path.join(save_dir, f"group_topomap_{t_start}_{t_end}ms.png")
    plt.savefig(fname_fig, dpi=300)
    plt.close()
    print(f"Saved topomap -> {fname_fig}")

# ---------------------- stack pics----------------------
n_cols = len(time_windows)
fig, axes = plt.subplots(1, n_cols, figsize=(3*n_cols, 4))
if n_cols == 1:
    axes = [axes]

for ax, pattern_window_centered, (t_start, t_end) in zip(axes, all_patterns_windowed, time_windows):
    im, _ = mne.viz.plot_topomap(
        pattern_window_centered, info, axes=ax, show=False,
        cmap='RdBu_r', vlim=(-global_vmax, global_vmax)
    )
    ax.set_title(f"{t_start}-{t_end} ms", fontsize=14, pad=-200)


#add colorbar
cbar_ax = fig.add_axes([0.90, 0.20, 0.007, 0.5])  # [left, bottom, width, height]，调整位置大小
fig.colorbar(im, cax=cbar_ax, label='Pattern ') #epoch
#fig.colorbar(im, cax=cbar_ax, label='Feature - Object (×1e3)') #alpha

plt.tight_layout(rect=[0, 0, 0.9, 1])  # save place for colorbar
fname_big = os.path.join(save_dir, "group_topomap_all_windows.png")
plt.savefig(fname_big, dpi=300)
plt.close()
print(f"\n✅ Combined topomap saved -> {fname_big}")
