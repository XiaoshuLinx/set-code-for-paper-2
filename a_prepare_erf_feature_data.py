#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 22:36:38 2025

@author: linx
"""



import mne
import numpy as np
from os.path import join
import gc



sample_path = '/projects/mindeye/MRI'
condition = 'dimension'


for subject_num in range(1, 47):
    if subject_num in [ 3,17, 32]:
        continue

    subject = f"{subject_num:02d}"
    subjects_dir = join(sample_path, subject)
    fname_fwd = join(subjects_dir, 'bem', f"{subject}-ob-fwd.fif")
    raw = mne.io.read_raw_fif(join(sample_path, 'wiica','clean', f"{subject}_ica_di_raw.fif"), preload=True)    
    events = mne.find_events(raw, shortest_event=1)

#labels    

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
    epochs= epochs.copy().pick_types(meg=True)
    # Extract 
    epochs= epochs.copy().pick_types(meg='grad')
 
    n_epochs, n_sensors, n_times = epochs._data.shape
   
    # Sum power 
    sum_data = np.zeros((n_epochs, n_sensors, n_times), dtype=np.float32)
    epochs= epochs.crop(tmin=-1.5, tmax=0)
    sum_data=epochs._data.astype(np.float32)  
    gc.collect()

        # Save 
    save_dir = '/projects/mindeye/ml_data/sensor/grad/'
    output_fname = join(save_dir, f'{subject}_{condition}_epoch_sensor.npz')
    np.savez(output_fname, data=sum_data,  times=epochs.times, channel=epochs.ch_names,
                )
        

    gc.collect()

