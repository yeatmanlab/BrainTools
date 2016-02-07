# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.time_frequency import tfr_morlet
from mne.time_frequency import single_trial_power
from mne.stats import permutation_cluster_test
import os
###############################################################################
# Set parameters
data_path = '/mnt/diskArray/projects/MEG/nlr/child/105_bb/sss_fif/'
rawfiles = np.sort(os.listdir(data_path))
raw_fname_word = []
raw_fname_dot = []
for r in rawfiles:
    if (int(r[r.find('_raw')-1]) % 2 == 0):
        raw_fname_word.append(data_path + r)
    else:
        raw_fname_dot.append(data_path + r)

tmin, tmax = -.2, 1.4

# Setup for reading the raw data
raw = io.Raw(raw_fname_word)
baseline = (None, 0)
events = mne.find_events(raw, stim_channel='STI101')

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True, stim=False)

# get epochs and conditions for each event
event_id = 5
epochs_condition_1 = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=dict(grad=4000e-13, eog=350e-6))
data_condition_1 = epochs_condition_1.get_data()
data_condition_1 *= 1e13
                    
event_id = 2
epochs_condition_2 = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=dict(grad=4000e-13, eog=350e-6))
data_condition_2 = epochs_condition_2.get_data()
data_condition_2 *= 1e13
###############################################################################
# Calculate power and intertrial coherence

freqs = np.arange(2, 50, 2)  # define frequencies of interest
n_cycles = freqs / 2.  # different number of cycle per frequency
power, itc = tfr_morlet(epochs_condition_1, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)

# Baseline correction can be applied to power or done in plots
# To illustrate the baseline correction in plots the next line is commented
# power.apply_baseline(baseline=(-0.5, 0), mode='logratio')

# Inspect power
power.plot_topo(baseline=(-0.2, 0), mode='logratio', vmin=-.4, vmax=.4, title='Average power')

# Inspect ITC
itc.plot_topo(title='Inter-Trial coherence', vmin=0., vmax=1., cmap='Reds')

###############################################################################
# T-test between conditions
# Time vector
ch_name = raw.info['ch_names'][picks[0]]
times = 1e3 * epochs_condition_1.times  # change unit to ms

# Factor to downsample the temporal dimension of the PSD computed by
# single_trial_power.  Decimation occurs after frequency decomposition and can
# be used to reduce memory usage (and possibly comptuational time of downstream
# operations such as nonparametric statistics) if you don't need high
# spectrotemporal resolution.
decim = 2
frequencies = np.arange(7, 30, 3)  # define frequencies of interest
sfreq = raw.info['sfreq']  # sampling in Hz
n_cycles = 1.5

epochs_power_1 = single_trial_power(data_condition_1, sfreq=sfreq,
                                    frequencies=frequencies,
                                    n_cycles=n_cycles, decim=decim)

epochs_power_2 = single_trial_power(data_condition_2, sfreq=sfreq,
                                    frequencies=frequencies,
                                    n_cycles=n_cycles, decim=decim)

epochs_power_1 = epochs_power_1[:, 0, :, :]  # only 1 channel to get 3D matrix
epochs_power_2 = epochs_power_2[:, 0, :, :]  # only 1 channel to get 3D matrix

# Compute ratio with baseline power (be sure to correct time vector with
# decimation factor)
baseline_mask = times[::decim] < 0
epochs_baseline_1 = np.mean(epochs_power_1[:, :, baseline_mask], axis=2)
epochs_power_1 /= epochs_baseline_1[..., np.newaxis]
epochs_baseline_2 = np.mean(epochs_power_2[:, :, baseline_mask], axis=2)
epochs_power_2 /= epochs_baseline_2[..., np.newaxis]

###############################################################################
# Compute statistic
threshold = 6.0
T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_test([epochs_power_1, epochs_power_2],
                             n_permutations=100, threshold=threshold, tail=0)

###############################################################################
# View time-frequency plots
plt.clf()
plt.subplots_adjust(0.12, 0.08, 0.96, 0.94, 0.2, 0.43)
plt.subplot(2, 1, 1)
evoked_contrast = np.mean(data_condition_1, 0) - np.mean(data_condition_2, 0)
plt.plot(times, evoked_contrast.T)
plt.title('Contrast of evoked response (%s)' % ch_name)
plt.xlabel('time (ms)')
plt.ylabel('Magnetic Field (fT/cm)')
plt.xlim(times[0], times[-1])
plt.ylim(-100, 200)

plt.subplot(2, 1, 2)

# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= 0.05:
        T_obs_plot[c] = T_obs[c]

plt.imshow(T_obs,
           extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
           aspect='auto', origin='lower', cmap='RdBu_r')
plt.imshow(T_obs_plot,
           extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
           aspect='auto', origin='lower', cmap='RdBu_r')

plt.xlabel('time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title('Induced power (%s)' % ch_name)
plt.show()
