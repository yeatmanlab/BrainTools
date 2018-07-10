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
import time

#%%
data_path = '/mnt/scratch/r21/180702/'
raw_fname = data_path + 'video_timing_01_raw.fif'
raw = mne.io.Raw(raw_fname,allow_maxshield=True,preload=True)

order = np.arange(raw.info['nchan'])
order[0] = 306  # We exchange the plotting order of two channels
order[1] = 307
order[2] = 308  # to show the trigger channel as the 10th channel.
raw.plot(n_channels=3, order=order, block=True, scalings='auto')

#picks = mne.pick_types(raw.info, meg=False, eeg=False, eog=False, stim=True, misc=True)
temp_data = raw.get_data(picks=[306,307])

plt.figure(1)
plt.clf()
plt.hold(True)
plt.plot(temp_data[0])
plt.plot(temp_data[1])

events = mne.find_events(raw, stim_channel='STI101')
trigger = events[:,0]
trigger = trigger - raw.first_samp

y1 = temp_data[0]
tt = 100
test = 1
onset = []
onsetdetect = 0
for i in np.arange(0,len(y1)):
    if y1[i] >= 0.1 and test:
        onset.append(i)
        onsetdetect = i
    if i < onsetdetect + tt:
        test = 0
    else:
        test = 1
onset = np.array(onset[:len(trigger)])
#%%
tDur = 10
video_delay = .070
lowpass = 40.
highpass = 0.5

data_path = '/mnt/scratch/r21/pmd'
raw_fname = data_path + '/sss_fif/pmd_1_raw_sss.fif'
#proj_fname = data_path + '/MEG/sample/sample_audvis_eog-proj.fif'

# Setup for reading the raw data (to save memory, crop before loading)
raw = mne.io.Raw(raw_fname,allow_maxshield=True,preload=True)
#raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

#raw.crop(tmin=12.0, tmax=13.0)
order = np.arange(raw.info['nchan'])
order[9] = 314 # We exchange the plotting order of two channels
#order[10] = 307
#order[306] = 9  # to show the trigger channel as the 10th channel.
raw.plot(n_channels=10, order=order, block=True)

# Add SSP projection vectors to reduce EOG and ECG artifacts
#projs = read_proj(proj_fname)
#raw.add_proj(projs, remove_existing=True)

layout = mne.channels.read_layout('Vectorview-mag')
layout.plot()
#raw.plot_psd_topo(tmax=30., fmin=5., fmax=60., n_fft=1024, layout=layout)

events = mne.find_events(raw, stim_channel='STI001')
events = np.array([events[0]])

tmin = (events[0][0]-raw.first_samp)/raw.info['sfreq'] + video_delay
tmax = tmin + tDur
# this can be highly data dependent
event_id = {'tag': 5}

raw.crop(tmin,tmax)

raw.filter(highpass, lowpass, fir_design='firwin')

#%%
#raw.plot_psd(area_mode='range', tmax=40, show=False, average=True)
#
#raw.filter(None, 40., fir_design='firwin')
#
#raw.plot_psd(area_mode='range', tmax=40, show=False, average=True)

#%%
reject = dict(grad=3000e-13, mag=4e-12, eog = np.inf)

projs, events = mne.preprocessing.compute_proj_ecg(raw, n_grad=1, n_mag=1, n_eeg=0, average=False)
print(projs)

ecg_projs = projs[-2:]
mne.viz.plot_projs_topomap(ecg_projs)

projs, events = mne.preprocessing.compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=0, average=False)
print(projs)

eog_projs = projs[-3:]
mne.viz.plot_projs_topomap(eog_projs, info=raw.info)

raw.info['projs'] += ecg_projs

evoked = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=40.,
                    proj='delayed', baseline=(None, 0),
                    reject=reject).average()

times = np.arange(0.05, 40, 5)

fig = evoked.plot_topomap(times, proj='interactive')

#%%
data_path = '/mnt/scratch/r21/pmd'
raw_fname = data_path + '/sss_fif/pmd_2_raw_sss.fif'
#proj_fname = data_path + '/MEG/sample/sample_audvis_eog-proj.fif'

# Setup for reading the raw data (to save memory, crop before loading)
raw2 = mne.io.Raw(raw_fname,allow_maxshield=True,preload=True)
#raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

events = mne.find_events(raw2, stim_channel='STI001')
events = np.array([events[0]])

tmin = (events[0][0]-raw2.first_samp)/raw2.info['sfreq'] + video_delay
tmax = tmin + tDur
# this can be highly data dependent
event_id = {'tag': 5}

raw2.crop(tmin,tmax)

#raw2.plot_psd(area_mode='range', tmax=40, show=False, average=True)

raw2.filter(highpass, lowpass, fir_design='firwin')

#raw2.plot_psd(area_mode='range', tmax=40, show=False, average=True)

#%%
temp_selection = mne.read_selection('Left-occipital')
selection = []
for i in np.arange(0,len(temp_selection)):
    selection.append(temp_selection[i][:3]+temp_selection[i][4:])
    
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       selection=selection)

#['MEG1641','MEG1642','MEG1643','MEG1941','MEG1942','MEG1943','MEG1731','MEG1732','MEG1733']

#%%
# Let's just look at the first few channels for demonstration purposes
#picks = picks[:4]

fmin, fmax = 0, 60  # look at frequencies between 2 and 300Hz
n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2

plt.figure()
ax = plt.axes()
raw.plot_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, n_fft=n_fft,
             n_jobs=1, proj=True, ax=ax, color=(0, 0, 1),  picks=picks,
             show=True, average=True)

raw.plot_psd_topo(tmin=0.0, tmax=tmax, fmin=0, fmax=fmax, proj=False, n_fft=n_fft, 
                  n_overlap=0, layout=None, color='w', fig_facecolor='k', axis_facecolor='k', dB=True, show=True, block=False, n_jobs=1, axes=None, verbose=None)

#%%

Fs = raw.info['sfreq'];  # sampling rate
Ts = 1.0/Fs; # sampling interval

temp_y1 = np.stack((raw.get_data(picks=picks)))
temp_y2 = np.stack((raw2.get_data(picks=picks)))
temp_y = np.stack((temp_y1,temp_y2))
y = np.mean(np.mean(temp_y, axis = 0),axis=0)

t = np.linspace(0.,tDur,tDur*Fs+1) # time vector


n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(n/2)] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(n/2)]

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')

ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

#%%
y1 = np.mean(temp_y1, axis = 0)
Y = np.fft.fft(y1)/n # fft computing and normalization
Y = Y[range(n/2)]

fig, ax = plt.subplots(3, 1)
ax[0].plot(frq,abs(Y),'r') # plotting the spectrum
ax[0].set_xlabel('Freq (Hz)')
ax[0].set_ylabel('|Y(freq)|')

y2 = np.mean(temp_y2, axis = 0)
Y = np.fft.fft(y2)/n # fft computing and normalization
Y = Y[range(n/2)]

ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

y = np.mean(np.mean(temp_y, axis = 1),axis=0)
Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(n/2)]

ax[2].plot(frq,abs(Y),'r') # plotting the spectrum
ax[2].set_xlabel('Freq (Hz)')
ax[2].set_ylabel('|Y(freq)|')

#%%
Fs = raw.info['sfreq'];  # sampling rate
Ts = 1.0/Fs; # sampling interval

temp_y1 = np.stack((raw.get_data(picks=np.arange(0,306))))

sig = np.empty((len(temp_y1),3))
for i in np.arange(0,len(temp_y1)):
    Y = np.fft.fft(temp_y1[i,:])/n # fft computing and normalization
    Y = Y[range(n/2)]
    temp_sig = abs(Y)
    sig[i,:] = temp_sig[47:50]

plt.figure(1)
plt.clf()
plt.hold(True)
plt.plot(sig)

chan = np.where(sig[:,2] > 7e-13)

#%%
temp_y = np.stack((raw.get_data(picks=chan)))
y = np.mean(np.mean(temp_y, axis = 0), axis = 0)

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(n/2)]

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')

ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')