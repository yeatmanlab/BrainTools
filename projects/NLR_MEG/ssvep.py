#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:34:05 2018

@author: sjjoo
"""

import mne
import os
import matplotlib.pyplot as plt
import numpy as np
import time

#%%
""" Timing test """
#data_path = '/mnt/scratch/r21/180712/'
#raw_fname = data_path + 'timing_test_01_raw.fif'
#raw = mne.io.Raw(raw_fname,allow_maxshield=True,preload=True)
#
#order = np.arange(raw.info['nchan'])
#order[0] = 306  # We exchange the plotting order of two channels
#order[1] = 308
#order[2] = 320 # to show the trigger channel as the 10th channel.
#raw.plot(n_channels=3, order=order, block=True, scalings='auto')
#
##picks = mne.pick_types(raw.info, meg=False, eeg=False, eog=False, stim=True, misc=True)
#temp_data = raw.get_data(picks=[306,320])
#
#plt.figure(1)
#plt.clf()
#plt.hold(True)
#plt.plot(temp_data[0])
#plt.plot(temp_data[1])
#
#events = mne.find_events(raw)
#trigger = events[:,0]
#trigger = trigger - raw.first_samp
#
#y1 = temp_data[1]
#tt = 100
#test = 1
#onset = []
#onsetdetect = 0
#for i in np.arange(trigger[0],len(y1)):
#    if y1[i] >= 0.1 and test:
#        onset.append(i)
#        onsetdetect = i
#    if i < onsetdetect + tt:
#        test = 0
#    else:
#        test = 1
#onset = np.array(onset[:len(trigger)])
#
#de = onset-trigger

#%%
tDur = 20.
video_delay = .066
lowpass = 40.
highpass = 0.5

data_path = '/mnt/scratch/r21/ek_short'
raw_fname1 = data_path + '/sss_fif/ek_short_1_raw_sss.fif'
raw_fname2 = data_path + '/sss_fif/ek_short_2_raw_sss.fif'
raw_fname3 = data_path + '/sss_fif/ek_short_3_raw_sss.fif'
raw_fname4 = data_path + '/sss_fif/ek_short_4_raw_sss.fif'

# Setup for reading the raw data (to save memory, crop before loading)
raw1 = mne.io.Raw(raw_fname1,allow_maxshield=True,preload=True)
raw2 = mne.io.Raw(raw_fname2,allow_maxshield=True,preload=True)
raw3 = mne.io.Raw(raw_fname3,allow_maxshield=True,preload=True)
raw4 = mne.io.Raw(raw_fname4,allow_maxshield=True,preload=True)

raws = mne.concatenate_raws([raw1,raw2,raw3,raw4])
#raw.resample(600, npad='auto', n_jobs=12)

#raw.crop(tmin=12.0, tmax=13.0)
#order = np.arange(raw.info['nchan'])
#order[9] = 306 # We exchange the plotting order of two channels
#order[10] = 307
#order[306] = 9  # to show the trigger channel as the 10th channel.
#raw.plot(n_channels=10, order=order, block=True)

# Add SSP projection vectors to reduce EOG and ECG artifacts
#projs = read_proj(proj_fname)
#raw.add_proj(projs, remove_existing=True)

#layout = mne.channels.read_layout('Vectorview-mag')
#layout.plot()
#raw.plot_psd_topo(tmax=30., fmin=5., fmax=60., n_fft=1024, layout=layout)

events = mne.find_events(raws, stim_channel=['STI001','STI002','STI003','STI004'])
#start = np.array([events[0]])
#
#tmin = (start[0][0]-raw1.first_samp)/raw1.info['sfreq'] + video_delay
#tmax = tmin + tDur
# this can be highly data dependent
event_id = {'tag': 5}

raws.filter(highpass, lowpass, fir_design='firwin')

#raw1.crop(tmin,tmax)

#%%
#raw.plot_psd(area_mode='range', tmax=40, show=False, average=True)
#
#raw.filter(None, 40., fir_design='firwin')
#
#raw.plot_psd(area_mode='range', tmax=40, show=False, average=True)

#%%
reject = dict(grad=4000e-13, mag=4e-12, eog = np.inf, ecg = np.inf)

projs, tmp = mne.preprocessing.compute_proj_ecg(raws, n_grad=1, n_mag=1, n_eeg=0, average=False, reject=reject)
print(projs)

ecg_projs = projs[-4:]
mne.viz.plot_projs_topomap(ecg_projs)

projs, tmp = mne.preprocessing.compute_proj_eog(raws, n_grad=1, n_mag=1, n_eeg=0, average=False, reject=reject)
print(projs)

eog_projs = projs[-4:]
mne.viz.plot_projs_topomap(eog_projs, info=raw1.info)

#%%
raws.info['projs'] += ecg_projs  + eog_projs

sessions1 = mne.Epochs(raws, events, event_id=None, tmin=0., tmax=tDur,
                    proj=False, baseline=None, reject=None)
yyy = sessions1.get_data()

plt.figure(1)
plt.clf()
plt.hold(True)
plt.plot(np.mean(yyy[:,201,:],axis=0),'k-')

sessions2 = mne.Epochs(raws, events, event_id=None, tmin=0., tmax=tDur,
                    proj=True, baseline=None, reject=None, detrend = 1)
yyy = sessions2.get_data()

plt.plot(np.mean(yyy[:,201,:],axis=0),'r-')

#%%
temp_selection = mne.read_selection('Left-occipital')
selection = []
for i in np.arange(0,len(temp_selection)):
    selection.append(temp_selection[i][:3]+temp_selection[i][4:])
    
picks = mne.pick_types(raw1.info, meg='mag', eeg=False, eog=False,
                       selection=['MEG1941','MEG1942','MEG1943'])

#['MEG1641','MEG1642','MEG1643','MEG1941','MEG1942','MEG1943','MEG1731','MEG1732','MEG1733']

#%%
# Let's just look at the first few channels for demonstration purposes
#picks = picks[:4]

#fmin, fmax = 0, 60  # look at frequencies between 2 and 300Hz
#n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2
#
#plt.figure()
#ax = plt.axes()
#raw.plot_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, n_fft=n_fft,
#             n_jobs=1, proj=True, ax=ax, color=(0, 0, 1),  picks=picks,
#             show=True, average=True)
#
#raw.plot_psd_topo(tmin=0.0, tmax=tmax, fmin=0, fmax=fmax, proj=False, n_fft=n_fft, 
#                  n_overlap=0, layout=None, color='w', fig_facecolor='k', axis_facecolor='k', dB=True, show=True, block=False, n_jobs=1, axes=None, verbose=None)


#%%
""" Let's plot the topomap """
Fs = sessions2.info['sfreq'];  # sampling rate
Ts = 1.0/Fs; # sampling interval

temp_y1 = sessions2.get_data()
n = temp_y1.shape[2]

sig = np.empty((temp_y1.shape[2],3))
for i in np.arange(0,temp_y1.shape[1]):
    y = temp_y1[0,i,:]
    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]
    temp_sig = abs(Y)
    sig[i,:] = temp_sig[24] # 48, 240

plt.figure(1)
plt.clf()
plt.hold(True)
plt.plot(np.mean(sig,axis=1))

chan = np.where(np.mean(sig,axis=1) > 0.6e-12)

ch_names = np.array(raw1.info['ch_names'])

plt.figure()
picks = mne.pick_types(sessions2.info, meg='mag', eeg=False, eog=False)
pick_info = mne.pick_info(sessions2.info, sel=picks)
data = np.mean(sig,axis=1)
mne.viz.plot_topomap(data[picks], pick_info, names = ch_names[picks], show_names=True)

plt.figure()
picks = mne.pick_types(sessions2.info, meg='grad', eeg=False, eog=False)
pick_info = mne.pick_info(sessions2.info, sel=picks)
data = np.mean(sig,axis=1)
#picks = mne.pick_types(sessions2.info, meg='mag', eeg=False, eog=False)
mne.viz.plot_topomap(data[picks], pick_info)

#%%
picks = mne.pick_types(sessions2.info, meg='mag', eeg=False, eog=False, selection=['MEG2041','MEG1942','MEG1943'])

Fs = sessions2.info['sfreq']  # sampling rate
Ts = 1.0/Fs # sampling interval

temp_y1 = sessions2.get_data()
#temp_y2 = np.stack((session2.get_data(picks=picks)))
#temp_y = np.stack((temp_y1,temp_y2))

y = np.mean(temp_y1[:,picks[0],:], axis = 0)

t = np.linspace(0.,tDur,tDur*Fs+1) # time vector

sigma = 20
#g = np.exp(-(t-20)**2 / (2*sigma**2))
xxx = np.linspace(0., 1., Fs)
temp_sin = np.sin(np.pi/2*xxx)

g = np.concatenate((temp_sin,np.ones(18001)))
g = np.concatenate((g,temp_sin[::-1]))
y = y*g

n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(n/2)] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(n/2)]

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,y)
#ax[0].plot(t,g,'g')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')

ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')