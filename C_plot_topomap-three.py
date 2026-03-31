#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:41:45 2026

@author: linx
Topomap over time (3 conditions: object, feature, difference) with global color scale
"""
import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ----------------
typea = 'sensor'
freq = "alpha"#epoch

data_dir = f"/nashome2/linx/Desktop/mindeye/ml_data/{typea}/grad"
save_dir = f"/nashome2/linx/Desktop/mindeye/ml_data/time_resolved_mvpa_{freq}_{typea}_coef_topomap_a/topomap_250ms_3conds"
os.makedirs(save_dir, exist_ok=True)

raw_fif = "/projects/mindeye/MRI/wiica/04_ica_di_raw.fif"

subject_list = [
'01','02','04','05','06','07','08','09','10',
'11','12','13','14','15','16','18','19','20',
'21','22','23','24','25','26','27','28','29',
'30','31','33','34','35','36','37','38','39',
'40','41','42','43','44','45','46'
]

# ---------------- LOAD INFO ----------------
raw = mne.io.read_raw_fif(raw_fif, preload=False)
picks = mne.pick_types(raw.info, meg='grad')
info = mne.pick_info(raw.info, picks)

# ---------------- TIME WINDOWS ----------------
time_windows = [
    (-1500,-1250),
    (-1250,-1000),
    (-1000,-750),
    (-750,-500),
    (-500,-250),
    (-250,0),
]

# ---------------- LOAD AND COMPUTE PATTERNS ----------------
def load_average_pattern(sub_list, cond_name):
    all_patterns = []
    for sub in sub_list:
        f_path = os.path.join(data_dir, f"{sub}_{cond_name}_{freq}_{typea}.npz")
        if not os.path.exists(f_path):
            continue
        x = np.load(f_path)["data"]  # (trials, sensors, times)
        x_avg = x.mean(axis=0)       # average trials -> (sensors, times)
        all_patterns.append(x_avg)
    X = np.stack(all_patterns, axis=0)  # (subjects, sensors, times)
    return X.mean(axis=0)  # group mean

# 计算三种模式
group_obj  = load_average_pattern(subject_list, 'object')     # (sensors, times)
group_feat = load_average_pattern(subject_list, 'dimension')  # (sensors, times)
group_diff = group_feat - group_obj                             # (sensors, times)

# ---------------- COMPUTE TIME-WINDOW AVERAGES ----------------
def window_patterns(group_data, time_windows, times, scale=1e3):
    patterns = []
    for t_start, t_end in time_windows:
        inds = np.where((times >= t_start) & (times < t_end))[0]
        pattern = group_data[:, inds].mean(axis=1) 
        pattern = pattern * scale
        patterns.append(pattern)
    return np.array(patterns)  # shape: (n_windows, n_sensors)

n_times = group_obj.shape[1]
times = np.linspace(-1500, 0, n_times)

all_obj_patterns  = window_patterns(group_obj, time_windows, times)
all_feat_patterns = window_patterns(group_feat, time_windows, times)
all_diff_patterns = window_patterns(group_diff, time_windows, times)

# ---------------- GLOBAL COLOR SCALE ----------------
global_vmax = max(
    np.max(np.abs(all_obj_patterns)),
    np.max(np.abs(all_feat_patterns)),
    np.max(np.abs(all_diff_patterns))
)
print("Global vmax for color scale:", global_vmax)

# ---------------- PLOT ----------------
conditions = ['Object', 'Feature', 'Feature - Object']
all_patterns_list = [all_obj_patterns, all_feat_patterns, all_diff_patterns]

n_rows = 3
n_cols = len(time_windows)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols+1, 2.9*n_rows))
if n_rows == 1:
    axes = [axes]
if n_cols == 1:
    axes = [[ax] for ax in axes]

for row_idx, (cond_name, patterns) in enumerate(zip(conditions, all_patterns_list)):
    for col_idx, (ax, pattern, (t_start, t_end)) in enumerate(
            zip(axes[row_idx], patterns, time_windows)):
        im, _ = mne.viz.plot_topomap(
            pattern, info, axes=ax,
            cmap='RdBu_r',
            vlim=(-global_vmax, global_vmax),
            show=False
        )
        if row_idx == 0:
            ax.set_title(f"{t_start}-{t_end} ms", fontsize=20)
        if col_idx == 0:
            ax.set_ylabel(cond_name, fontsize=20)

for ax_row in axes:
    for ax in ax_row:
        ax.set_xticks([])
        ax.set_yticks([])
        
plt.subplots_adjust(
    left=0.05,
    right=0.90,   
    top=0.92,
    bottom=0.05,
    wspace=0,  
    hspace=0  
)

# 右侧统一 colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Amplitude (×1e3)')
#fig.colorbar(im, cax=cbar_ax, label='Amplitude')
#plt.tight_layout(rect=[0, 0, 0.9, 1])
fname = os.path.join(save_dir, "topomap_timecourse_3conds.png")
plt.savefig(fname, dpi=300)
plt.close()
print("✅ Saved:", fname)
