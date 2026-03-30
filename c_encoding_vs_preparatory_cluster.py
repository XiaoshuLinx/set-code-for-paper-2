#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 10:22:41 2026

@author: linx

Cluster permutation test:
Save all significant clusters (p<0.05) to Excel for later correlation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import t



# ============================
# PATH
# ============================

freq = "alpha" #"epoch"
data_original='sensor'
save_root = f"/nashome2/linx/Desktop/mindeye/ml_data/time_resolved_mvpa_{freq}_{data_original}_coef_topomap_a"

#save_root = f"/nashome2/linx/Desktop/mindeye/ml_data/time_resolved_mvpa_{freq})_nobaseline"
per_subj_dir = os.path.join(save_root, "per_subject")
excel_out = os.path.join(save_root, "significant_clusters.xlsx")

n_times = 376
baseline_idx = np.arange(0, 125)
encoding_idx = np.arange(125, 376)

times = np.linspace(-1500, 0, n_times)
dt = times[1] - times[0]

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
# BASELINE MEAN
# ============================


baseline_mean = all_subject_auc[:, baseline_idx].mean(axis=1)
group_baseline = baseline_mean.mean()
X_diff = all_subject_auc - baseline_mean[:, None]

print(f"Group baseline mean AUC: {group_baseline:.4f}")

df = n_subjects - 1
threshold = t.ppf(1 - 0.05, df)  # 单尾 p=0.05
# ============================
# CLUSTER PERMUTATION TEST
# ============================

T_obs, clusters, p_values, _ = permutation_cluster_1samp_test(
    X_diff,
    n_permutations=5000,
    threshold=None,
    tail=1,
    n_jobs=1
)

print("\nCluster test finished.")
print(f"Total clusters found: {len(clusters)}")

# ============================
# EXTRACT SIGNIFICANT CLUSTERS
# ============================

group_mean = all_subject_auc.mean(axis=0)
global_mean_auc = group_mean.mean()

sig_clusters_info = []

min_duration_ms = 0  # 比如最少40ms
min_points = int(min_duration_ms / dt)

for i_cluster, p_val in enumerate(p_values):
    if p_val >= 0.05:
        continue  # skip non-significant

    cluster_inds = clusters[i_cluster][0]
    t_values = T_obs[cluster_inds]        # 每个点的 t 值
    cluster_t_sum = t_values.sum()       # ⭐ cluster mass（最标准）
    if len(cluster_inds) < min_points:
       continue 
    t_start = times[cluster_inds[0]]
    t_end = times[cluster_inds[-1]]
    duration = len(cluster_inds) * dt
    cluster_mean_auc = group_mean[cluster_inds].mean()
    cluster_mean_diff = X_diff[:, cluster_inds].mean()
    cluster_values = X_diff[:, cluster_inds].mean(axis=1)

    mean_diff = cluster_values.mean()
    std_diff = cluster_values.std(ddof=1)

    cohens_d = mean_diff / std_diff
    peak_auc = group_mean[cluster_inds].max()
    #cluster_stat = T_obs[clusters[i]].sum()


    if np.all(cluster_inds < 125):
        region = "Baseline"
    elif np.all(cluster_inds >= 125):
        region = "Encoding"
    else:
        region = "Mixed"


    sig_clusters_info.append({
    "Cluster_ID": i_cluster + 1,
    "p_value": p_val,
    "Time_start_ms": t_start,
    "Time_end_ms": t_end,
    "Duration_ms": duration,
    "Baseline_mean_AUC": group_baseline,
    "Global_mean_AUC": global_mean_auc,
    "Mean_AUC": cluster_mean_auc,
    "Mean_enhancement_vs_baseline": cluster_mean_diff,
    "Cohens_d": cohens_d,
    "Peak_AUC": peak_auc,
    "Region": region,
    "Cluster_t_sum": cluster_t_sum,
})

    print(
    f"\nCluster {i_cluster + 1}"
    f"\np = {p_val:.4f}"
    f"\nTime = {t_start:.1f} – {t_end:.1f} ms"
    f"\nDuration = {duration:.1f} ms"
    f"\nBaseline mean AUC = {group_baseline:.3f}"
    f"\nGlobal mean AUC = {global_mean_auc:.3f}"
    f"\nCluster mean AUC = {cluster_mean_auc:.3f}"
    f"\nEnhancement vs baseline = {cluster_mean_diff:.3f}"
    f"\nPeak AUC = {peak_auc:.3f}"
    f"\nCohen's d = {cohens_d:.3f}"
    f"\nRegion = {region}\n"
    f"\nCluster t-sum = {cluster_t_sum:.3f}"
)



# ============================
# SAVE TO EXCEL
# ============================

if sig_clusters_info:
    df_sig = pd.DataFrame(sig_clusters_info)
    df_sig.to_excel(excel_out, index=False)
    print(f"\nSignificant clusters saved to {excel_out}")
else:
    print("\nNo significant clusters found (p<0.05).")

# ============================
# PLOT
# ============================

plt.figure(figsize=(10,4))
plt.plot(times, group_mean, lw=2)
plt.axhline(group_baseline, linestyle="--", label="Group baseline mean")

for cluster in sig_clusters_info:
    plt.axvspan(cluster["Time_start_ms"], cluster["Time_end_ms"], alpha=0.3)

plt.legend()

ax = plt.gca()


# x ticks
ax.tick_params(
    axis='x',
    which='both',
    direction='in',   # tick：'in', 'out', 'inout'
    length=6,            # tick length
    labeltop=False,
    labelbottom=True,
    pad=-15,
    labelsize=10               
)

# y轴：刻度文字小一点
ax.yaxis.set_tick_params(labelsize=10)

# 轴标签
ax.set_xlabel("Time (ms)", fontsize=12)  # xlabel
ax.set_ylabel("AUC", fontsize=12)        # ylabel

# 标题和 legend 字体
ax.set_title("Timepoints significantly > baseline mean (cluster p<0.05)", fontsize=12)
ax.legend(fontsize=10)
ax.set_xlabel("Time (ms)", fontsize=12)  # xlabel
ax.set_ylabel("AUC", fontsize=12)        # ylabel
plt.tight_layout()
plt.savefig(os.path.join(save_root, "cluster_vs_baseline.png"), dpi=300)
plt.close()

print("Figure saved as cluster_vs_baseline.png")
