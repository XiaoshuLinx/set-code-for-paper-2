#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 16:36:04 2025

@author: linx
"""

import mne
import numpy as np
from os.path import join
import gc

bands = {"alpha": (np.arange(8, 15), 6),}

sample_path = '/projects/mindeye/MRI'
condition = 'dimension'
save_dir = '/projects/mindeye/ml_data/'

for subject_num in range(1, 47):
    if subject_num in [3,17, 32]:
        continue

    subject = f"{subject_num:02d}"
    subjects_dir = join(sample_path, subject)
    fname_fwd = join(subjects_dir, 'bem', f"{subject}-ob-fwd.fif")
    raw = mne.io.read_raw_fif(join(sample_path, 'wiica','clean', f"{subject}_ica_di_raw.fif"), preload=True)
    events = mne.find_events(raw, shortest_event=1)


    # Fix event codes for specific subjects
    if subject_num == 18:
        events[:, 2] = events[:, 2] - events[:, 1]
        true_event = []
        for i in events[:, 2]:
            if i == 8192:
                true_event.append(256)
            elif i == 4096:
                true_event.append(512)
            else:
                true_event.append(i)
        events[:, 2] = true_event

    if subject_num in {2, 4, 7, 11, 12}:
        events[:, 2] = events[:, 2] - 3072

    # Merge event codes as per protocol
    events = mne.merge_events(events, [18, 14], 18)
    events = mne.merge_events(events, [16, 12], 12)
    events = mne.merge_events(events, [11, 13, 15, 17], 11)
    events = mne.merge_events(events, [11, 12, 18], 20)

    reject_criteria = dict(mag=4000e-15, grad=3000e-13)
    flat_criteria = dict(mag=1e-15, grad=1e-13)

    event_id = {'dimension': 20}
    picked_events = mne.pick_events(events, include=list(event_id.values()))
 
    # Remove events too close together
    min_interval = 250
    keep_mask = np.ones(len(picked_events), dtype=bool)
    for i in range(1, len(picked_events)):
        if picked_events[i, 0] - picked_events[i - 1, 0] < min_interval:
            keep_mask[i] = False
    picked_events = picked_events[keep_mask]    
    
    
    epochs = mne.Epochs(raw, picked_events, event_id=event_id, tmin=-2, tmax=0.5, baseline=None,
                        reject_tmax=0, flat=flat_criteria, reject=reject_criteria, preload=True)
    # Extract 
    epochs = mne.Epochs.subtract_evoked(epochs, evoked=None)
    epochs= epochs.copy().pick_types(meg=True)
    epochs= epochs.copy().pick_types(meg="grad")
     
    # ------- Now extract behavioral labels only for the picked events -------
    # First, find out which events from the original "events" these picked ones correspond to
    picked_event_samples = picked_events[:, 0]
    all_event_samples = events[:, 0]

    # Match indices between picked events and all events (find closest match)
    picked_indices = [np.where(all_event_samples == samp)[0][0] for samp in picked_event_samples]
    epochs = mne.Epochs.subtract_evoked(epochs, evoked=None)

    for band_name, (freqs, n_cycles) in bands.items():
        print(f"Running {band_name} band: {freqs[0]}–{freqs[-1]} Hz")

        # Get dimensions from a dummy frequency
        tmp_freq = np.array([freqs[0]])
        tmp_tfr = mne.time_frequency.tfr_morlet(epochs, freqs=tmp_freq, n_cycles=n_cycles,
                                                return_itc=False, output="complex", average=False).crop(tmin=-1.5, tmax=0)     
        # -------------- Sensor Level --------------
        # Compute sensor power TFR across all freqs (per epoch)
        print(f"  Computing sensor TFR power for {band_name} band")
        epochs_tfr = mne.time_frequency.tfr_morlet(
            epochs, freqs=freqs, n_cycles=n_cycles,
            return_itc=False, output="power", average=False).crop(tmin=-1.5, tmax=0)
        sensor_data = epochs_tfr.data  # shape: (n_epochs, n_freqs, n_channels, n_times)      
        # Baseline sensor power (same baseline window)
        print(f"  Computing sensor baseline power for {band_name} band")

        # Baseline correction 
        sensor_data_avg = sensor_data.mean(axis=2).astype(np.float32)  # average over freq dim

    # Save source baseline-corrected data
        save_dir = '/projects/mindeye/ml_data/sensor/grad'
        output_fname = join(save_dir, f'{subject}_{condition}_{band_name}_sensor.npz')
        np.savez(output_fname, data=sensor_data_avg ,  times=epochs.times, channel=epochs.ch_names,)

        # Cleanup
        del epochs_tfr, sensor_data,  sensor_data_avg,   
        gc.collect()

    del epochs, raw, events
    gc.collect()
