#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:59:48 2026

@author: linx
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem
from mne.stats import permutation_cluster_1samp_test

# ---------------- PARAMETERS ----------------
typea='sensor'
freq = "alpha"#epoch
save_dir = "/nashome2/linx/Desktop/mindeye/figures"
subject_list = [
'01','02','04','05','06','07','08','09','10',
'11','12','13','14','15','16','18','19','20',
'21','22','23','24','25','26','27','28','29',
'30','33','34','35','36','37','38','39',
'40','41','42','43','44','45','46'
]

data_dir = f"/nashome2/linx/Desktop/mindeye/ml_data/{typea}/grad"

smooth_sigma = 2
cluster_alpha = 0.05
t_keeps=376
os.makedirs(save_dir, exist_ok=True)

# ---------------- LOAD DATA ----------------

feature_all = []
object_all = []

for sub in subject_list:

    f_dim = os.path.join(data_dir, f"{sub}_dimension_{freq}_{typea}.npz")
    f_obj = os.path.join(data_dir, f"{sub}_object_{freq}_{typea}.npz")

    if not (os.path.exists(f_dim) and os.path.exists(f_obj)):
        print("skip", sub)
        continue

    x_dim = np.load(f_dim)["data"][:,:,:t_keeps]
    x_obj = np.load(f_obj)["data"][:,:,:t_keeps]

    # average trials + sensors
    dim_ts = x_dim.mean(axis=(0,1))
    obj_ts = x_obj.mean(axis=(0,1))

    feature_all.append(dim_ts)
    object_all.append(obj_ts)

feature_all = np.array(feature_all)
object_all = np.array(object_all)

n_subjects, n_times = feature_all.shape

print("Subjects:", n_subjects)

# ---------------- TIME ----------------

times = np.linspace(-1500,0,n_times)

# ---------------- SMOOTHING ----------------

feature_s = gaussian_filter1d(feature_all, smooth_sigma, axis=1)
object_s = gaussian_filter1d(object_all, smooth_sigma, axis=1)

# ---------------- GROUP STATS ----------------

feature_mean = feature_s.mean(axis=0)
object_mean = object_s.mean(axis=0)

feature_sem = sem(feature_s, axis=0)
object_sem = sem(object_s, axis=0)

# difference
diff_subj = feature_s - object_s
diff_mean = diff_subj.mean(axis=0)
diff_sem = sem(diff_subj, axis=0)

# ---------------- CLUSTER TEST in full time----------------

T_obs, clusters, p_vals, _ = permutation_cluster_1samp_test(
    diff_subj,
    n_permutations=1000,
    tail=0,
    out_type="mask"
)

sig_mask = np.zeros(n_times)

for c,p in zip(clusters,p_vals):
    if p < cluster_alpha:
        sig_mask[c] = 1
        
# ---------------- CLUSTER TEST: encoding vs baseline ----------------

# define time windows
baseline_mask = (times >= -1500) & (times <= -1000)
encoding_mask = (times > -1000) & (times <= 0)

# compute baseline mean per subject
baseline_mean = diff_subj[:, baseline_mask].mean(axis=1, keepdims=True)

# baseline-corrected difference (each subject)
diff_bc = diff_subj - baseline_mean  # shape: (n_subj, n_times)

# only test in encoding window
diff_bc_encoding = diff_bc[:, encoding_mask]

# cluster test
T_obs_enc, clusters_enc, p_vals_enc, _ = permutation_cluster_1samp_test(
    diff_bc_encoding,
    n_permutations=1000,
    tail=0,
    out_type="mask"
)

# create full-length mask (same length as times)
sig_mask_enc = np.zeros(n_times)

for c, p in zip(clusters_enc, p_vals_enc):
    if p < cluster_alpha:
        # map back to full time indices
        sig_mask_enc[np.where(encoding_mask)[0][c]] = 1

# ---------------- PLOT ----------------

fig, axes = plt.subplots(
    2,1,
    figsize=(10,4),
    sharex=True,
    gridspec_kw={"height_ratios":[1,1]}
)

# -------- Panel A: Feature & Object signals --------
ax = axes[0]
ax.plot(times, object_mean, color="tab:red", label="Object")
ax.plot(times, feature_mean, color="tab:blue", label="Feature")

ax.fill_between(times,
                object_mean-object_sem,
                object_mean+object_sem,
                color="tab:red", alpha=0.25)
ax.fill_between(times,
                feature_mean-feature_sem,
                feature_mean+feature_sem,
                color="tab:blue", alpha=0.25)

ax.set_ylabel("Signal", fontsize=12)
ax.set_title("Group Time Series: Feature vs Object", fontsize=12)
ax.tick_params(axis='both', labelsize=10)
ax.legend(fontsize=10, frameon=False)

# -------- Panel B: Difference & Cluster --------
ax = axes[1]

min_sem = 1e-24
diff_sem_plot = np.maximum(diff_sem, min_sem)

ax.plot(times, diff_mean, color="black")
ax.fill_between(times,
                diff_mean-diff_sem_plot,
                diff_mean+diff_sem_plot,
                color="black", alpha=0.2)

ax.axhline(0, color="gray", linestyle="--")

ymin, ymax = ax.get_ylim()
cluster_height = 0.15 * (ymax - ymin)

# -------- Full time vs 0 (yellow) --------
ax.fill_between(times,
                ymin,
                ymin + cluster_height,
                where=sig_mask.astype(bool),
                color="yellow",
                label="full time vs 0 ")

# -------- Encoding vs baseline (green) --------
ax.fill_between(times,
                ymin + cluster_height,
                ymin + 2*cluster_height,
                where=sig_mask_enc.astype(bool),
                color="green",
                label="encoding vs early prepartory")

ax.set_xlabel("Time (ms)", fontsize=12)
ax.set_ylabel("Feature − Object", fontsize=12)
ax.tick_params(axis='both', labelsize=10)

ax.legend(
    loc="lower left",
    frameon=False,
    fontsize=9
)



# Style tweaks
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(times[0], times[-1])
    ax.margins(x=0)

plt.tight_layout()
save_path = os.path.join(save_dir, f"group_TSE_{freq}_{typea}_feature_vs_object.png")
plt.savefig(save_path, dpi=300)
plt.show()
print("Figure saved:", save_path)
