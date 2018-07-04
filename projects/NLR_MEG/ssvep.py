#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:34:05 2018

@author: sjjoo
"""

import mne
import os
import plotlib as plt
import nympy as np

#%%
data_path = '/mnt/scratch/r21/joo_sung_session'
raw_fname = data_path + '/sss_fif/joo_sung_session_1_raw_sss.fif'
#proj_fname = data_path + '/MEG/sample/sample_audvis_eog-proj.fif'

tmin, tmax = 12, 52  #

# Setup for reading the raw data (to save memory, crop before loading)
raw = mne.io.read_raw_fif(raw_fname,allow_maxshield=True,preload=True).crop(tmin, tmax).load_data()
#raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

order = np.arange(raw.info['nchan'])
order[9] = 306  # We exchange the plotting order of two channels
order[306] = 9  # to show the trigger channel as the 10th channel.
raw.plot(n_channels=10, order=order, block=True)

# Add SSP projection vectors to reduce EOG and ECG artifacts
#projs = read_proj(proj_fname)
#raw.add_proj(projs, remove_existing=True)


fmin, fmax = 0, 50  # look at frequencies between 2 and 300Hz
n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2

#%%
layout = mne.channels.read_layout('Vectorview-mag')
layout.plot()
#raw.plot_psd_topo(tmax=30., fmin=5., fmax=60., n_fft=1024, layout=layout)

#%%
raw.plot_psd(area_mode='range', tmax=10.0, show=False, average=True)

raw.filter(0.15, 40., fir_design='firwin')

raw.plot_psd(area_mode='range', tmax=10.0, show=False, average=True)

#%%
reject = dict(grad=5000e-13, mag=4e-12, eog = np.inf)

projs, events = mne.preprocessing.compute_proj_ecg(raw, n_grad=1, n_mag=1, n_eeg=0, average=False)
print(projs)

ecg_projs = projs[-2:]
mne.viz.plot_projs_topomap(ecg_projs)

projs, events = mne.preprocessing.compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=0, average=False)
print(projs)

eog_projs = projs[-3:]
mne.viz.plot_projs_topomap(eog_projs, info=raw.info)

#%%
raw.info['projs'] += ecg_projs
events = mne.find_events(raw, stim_channel='STI001')
events = np.array([events[0]])

# this can be highly data dependent
event_id = {'tag': 5}

epochs_no_proj = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=40,
                            proj=False, baseline=(None, 0), reject=reject)
epochs_no_proj.average().plot(spatial_colors=True)


epochs_proj = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=40, proj=True,
                         baseline=(None, 0), reject=reject)
epochs_proj.average().plot(spatial_colors=True)

evoked = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=40.,
                    proj='delayed', baseline=(None, 0),
                    reject=reject).average()

#times = np.arange(0.05, 40, 5)
#
#fig = evoked.plot_topomap(times, proj='interactive')

#%%
temp_selection = mne.read_selection('Left-occipital')
selection = []
for i in np.arange(0,len(temp_selection)):
    selection.append(temp_selection[i][:3]+temp_selection[i][4:])
    
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       selection=selection)

#['MEG1641','MEG1642','MEG1643','MEG1941','MEG1942','MEG1943','MEG1731','MEG1732','MEG1733']

# Let's just look at the first few channels for demonstration purposes
#picks = picks[:4]

plt.figure()
ax = plt.axes()
raw.plot_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, n_fft=n_fft,
             n_jobs=1, proj=True, ax=ax, color=(0, 0, 1),  picks=picks,
             show=True, average=True)

raw.plot_psd_topo(tmin=0.0, tmax=tmax, fmin=0, fmax=fmax, proj=False, n_fft=n_fft, 
                  n_overlap=0, layout=None, color='w', fig_facecolor='k', axis_facecolor='k', dB=True, show=True, block=False, n_jobs=1, axes=None, verbose=None)

#%%
#raw.plot_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, n_fft=n_fft,
#             n_jobs=1, proj=True, ax=ax, color=(0, 1, 0), picks=picks,
#             show=False, average=True)

# And now do the same with SSP + notch filtering
# Pick all channels for notch since the SSP projection mixes channels together
raw.notch_filter(np.arange(60, 241, 60), n_jobs=1, fir_design='firwin')
raw.plot_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, n_fft=n_fft,
             n_jobs=1, proj=True, ax=ax, color=(1, 0, 0), picks=picks,
             show=False, average=True)

ax.set_title('Four left-temporal magnetometers')
plt.legend(ax.lines[::3], ['Without SSP', 'With SSP', 'SSP + Notch'])