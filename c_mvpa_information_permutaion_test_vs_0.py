#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:26:05 2026

@author: linx

Compute descriptive decoding metrics for predefined time windows.

Outputs:
- Mean AUC within time window
- Enhancement vs baseline
"""

import os
import numpy as np

# ============================
# PATH
# ============================

freq = "alpha" #'epoch'
data_original='sensor'
save_root = f"/nashome2/linx/Desktop/mindeye/ml_data/time_resolved_mvpa_{freq}_{data_original}_coef_topomap_a"

per_subj_dir = os.path.join(save_root, "per_subject")

# ============================
# PARAMETERS
# ============================

n_times = 376
times = np.linspace(-1500, 0, n_times)

baseline_idx = np.arange(0, 125)

# 👉 你要的两个时间窗（可改）
time_windows = {
    "epoch": (-932, -200),
    #"alpha":(-480, 0),
   "encoding": (-1000, 0),
   
   
}

# ============================
# LOAD DATA
# ============================

files = sorted([f for f in os.listdir(per_subj_dir) if f.endswith("_time_auc.npy")])

all_subject_auc = []
for f in files:
    data = np.load(os.path.join(per_subj_dir, f))
    if len(data) == n_times:
        all_subject_auc.append(data)

all_subject_auc = np.stack(all_subject_auc, axis=0)
n_subjects = all_subject_auc.shape[0]

print(f"\nLoaded {n_subjects} subjects.")

# ============================
# BASELINE
# ============================

baseline_mean = all_subject_auc[:, baseline_idx].mean(axis=1)
group_baseline = baseline_mean.mean()

print(f"\nGroup baseline mean AUC = {group_baseline:.3f}")

# baseline-corrected data
X_diff = all_subject_auc - baseline_mean[:, None]

# group mean curve
group_mean = all_subject_auc.mean(axis=0)
print(f'full time mean auc ={group_mean.mean()}')
# ============================
# DESCRIPTIVE OUTPUT
# ============================

print("\n============================")
print("DESCRIPTIVE RESULTS")
print("============================")

for name, (t_start, t_end) in time_windows.items():
    
    win_idx = np.where((times >= t_start) & (times <= t_end))[0]
    
    if len(win_idx) == 0:
        print(f"\n{name}: No timepoints found")
        continue
    
    # Mean AUC（group level）
    window_auc = group_mean[win_idx].mean()
    
    # Enhancement vs baseline
    window_enh = X_diff[:, win_idx].mean()
    
    print(f"\n{name.upper()} WINDOW")
    print(f"Time = {t_start:.0f} – {t_end:.0f} ms")
    print(f"Mean AUC = {window_auc:.3f}")
    print(f"Enhancement vs baseline = {window_enh:.3f}")


print("\nDone.")


from mne.stats import permutation_cluster_1samp_test

# ============================
# CLUSTER-BASED TEST vs CHANCE
# ============================

print("\n============================")
print("CLUSTER PERMUTATION TEST vs CHANCE (0.5)")
print("============================")

# subtract chance level
X = all_subject_auc - 0.5  # shape: (subjects, time)

# run cluster test
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
    X,
    n_permutations=1000,
    tail=1,              # > chance
    threshold=None,      # default t-threshold
    out_type='mask'
)

# ============================
# PRINT RESULTS
# ============================

significant = cluster_p_values < 0.05

if not np.any(significant):
    print("\nNo significant clusters found.")
else:
    print(f"\nFound {significant.sum()} significant clusters:\n")
    
    for i, sig in enumerate(significant):
        if sig:
            cluster_mask = clusters[i]
            cluster_inds = np.where(cluster_mask)[0]
            cluster_times = times[cluster_inds]
        
            t_start = cluster_times[0]
            t_end = cluster_times[-1]
            p_val = cluster_p_values[i]

        # ============================
        # cluster-level statistic
        # ============================
            cluster_stat = T_obs[cluster_inds].sum()

        # ============================
        # Cohen's d (vs chance)
        # ============================
            cluster_values = (all_subject_auc[:, cluster_inds] - 0.5).mean(axis=1)

            mean_diff = cluster_values.mean()
            std_diff = cluster_values.std(ddof=1)
            cohens_d_chance = mean_diff / std_diff

            print(f"Cluster {i+1}:")
            print(f"Time = {t_start:.0f} – {t_end:.0f} ms")
            print(f"p = {p_val:.5f}")
            print(f"cluster-level t-sum = {cluster_stat:.2f}")
            print(f"Cohen's d (vs chance) = {cohens_d_chance:.3f}")
            print("-----")


