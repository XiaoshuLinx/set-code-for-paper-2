#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:05:05 2026

@author: linx
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Butterfly plot with RMS and mean, spatiotemporal clusters
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from mne.stats import spatio_temporal_cluster_1samp_test
import mne

# ---------------- PARAMETERS ----------------
typea = 'sensor'
freq = "alpha" #"epoch"
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
t_keeps = 376
os.makedirs(save_dir, exist_ok=True)

# ---------------- LOAD DATA ----------------
feature_all, object_all = [], []

for sub in subject_list:
    f_dim = os.path.join(data_dir, f"{sub}_dimension_{freq}_{typea}.npz")
    f_obj = os.path.join(data_dir, f"{sub}_object_{freq}_{typea}.npz")
    if not (os.path.exists(f_dim) and os.path.exists(f_obj)):
        print("skip", sub)
        continue
    x_dim = np.load(f_dim)["data"][:, :, :t_keeps]
    x_obj = np.load(f_obj)["data"][:, :, :t_keeps]
    feature_all.append(x_dim.mean(axis=0))
    object_all.append(x_obj.mean(axis=0))

feature_all = np.array(feature_all)  # subjects x sensors x times
object_all = np.array(object_all)
n_subjects, n_sensors, n_times = feature_all.shape
print("Subjects:", n_subjects, "| Sensors:", n_sensors)

# ---------------- TIME ----------------
times = np.linspace(-1500, 0, n_times)

# ---------------- SMOOTHING ----------------
feature_s = gaussian_filter1d(feature_all, smooth_sigma, axis=2)
object_s = gaussian_filter1d(object_all, smooth_sigma, axis=2)
diff = feature_s - object_s  # subjects x sensors x times

# ---------------- LOAD INFO ----------------
raw_fif = "/projects/mindeye/MRI/wiica/04_ica_di_raw.fif"
raw = mne.io.read_raw_fif(raw_fif, preload=False)
raw.pick_types(meg='grad')
info = raw.info

# ---------------- BUILD ADJACENCY ----------------
adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type='grad')

# ---------------- RESHAPE FOR CLUSTER ----------------
X = np.transpose(diff, (0, 2, 1))  # subjects x times x sensors

# ---------------- SPATIOTEMPORAL CLUSTER ----------------
T_obs, clusters, p_vals, _ = spatio_temporal_cluster_1samp_test(
    X, adjacency=adjacency, n_permutations=1000, threshold=None, tail=0, n_jobs=1
)

sig_mask_time = np.zeros(n_times)
for c, p in zip(clusters, p_vals):
    if p < cluster_alpha:
        t_inds = c[0]
        sig_mask_time[t_inds] = 1

# ---------------- GROUP MEAN ----------------
feature_mean = feature_s.mean(axis=0)
object_mean = object_s.mean(axis=0)
diff_mean = diff.mean(axis=0)

# ---------------- BUILD EVOKED ----------------
tmin_sec = times[0] / 1000  
evoked_feat = mne.EvokedArray(feature_mean, info, tmin=tmin_sec )
evoked_obj = mne.EvokedArray(object_mean, info, tmin=tmin_sec )
evoked_diff = mne.EvokedArray(diff_mean, info, tmin=tmin_sec )

# ---------------- FIGURE ----------------
fig, axes = plt.subplots(3, 1, figsize=(10, 6),
                         gridspec_kw={'height_ratios':[2,2,2]}, sharex=True)

def plot_evoked_butterfly(evoked, ax, title, show_rms=True):
    # butterfly plot
    evoked.plot(picks='grad', axes=ax, show=False, spatial_colors=True, gfp=True)

    # RMS (可选)
    if show_rms:
        rms = np.sqrt(np.mean(evoked.data**2, axis=0))
        ax.plot(times, rms, color='blue', linewidth=1.5)

    # mean line（保留）
    mean_data = evoked.data.mean(axis=0)
    scale_factor = np.max(np.abs(evoked.data)) / np.max(np.abs(mean_data))
    ax.plot(times, mean_data * scale_factor, color='black', linewidth=2.5)

    # clusters
    for c, p in zip(clusters, p_vals):
        if p < cluster_alpha:
            t_inds = c[0]
            ax.axvspan(times[t_inds[0]], times[t_inds[-1]],
                       color='red', alpha=0.15)

    ax.set_title(title, fontsize=16, y=1)

    # 只保留左+下边框
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    #ax.set_xticks(np.arange(-1500, 1, 500))
    #ax.set_xticklabels([f"{int(t)}" for t in np.arange(-1500, 1, 500)])
# -------- Feature --------
plot_evoked_butterfly(evoked_feat, axes[0], "Feature", show_rms=False)
axes[0].tick_params(axis='x', labelbottom=False)
axes[0].set_xlabel("")
axes[0].set_title("Feature", fontsize=16, y=1)
# -------- Object --------
plot_evoked_butterfly(evoked_obj, axes[1], "Object", show_rms=False)
axes[1].tick_params(axis='x', labelbottom=False)
axes[1].set_xlabel("")
axes[1].set_title("Object", fontsize=16, y=1)
# -------- Difference --------
plot_evoked_butterfly(evoked_diff, axes[2], "Feature − Object", show_rms=True)
axes[2].set_xlabel("Time (ms)")
axes[2].set_title("Feature − Object", fontsize=16, y=1)

plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, hspace=0.35)

# ---------------- SAVE ----------------
save_path = os.path.join(save_dir, f"{typea}_{freq}_Evoked_diff_butterfly.png")
plt.savefig(save_path, dpi=300)
plt.show()
print("✅ Saved:", save_path)
