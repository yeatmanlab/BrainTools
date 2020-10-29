#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:33:39 2020

@author: caffarra
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mne
from mne.stats import (spatio_temporal_cluster_test, permutation_cluster_test)
from mne.stats import (permutation_cluster_1samp_test,
                       spatio_temporal_cluster_1samp_test)
from scipy import stats as stats

from mne.datasets import sample
from mne.channels import (find_ch_connectivity,find_ch_adjacency)
from mne.viz import plot_compare_evokeds
from mne.channels.layout import (_merge_ch_data, _pair_grad_sensors)


root='/mnt/scratch/NLR_MEG'
dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]

print (dirlist)

dirlist.sort()

#Cross-sectional
All= [1,3,4,8,10,11,14,16,18,19,23,40,54,64,67,71,80,6,13,21,26,28,30,32,36,38,43,47,49,53,57,62,78,76,60,74,68,42,66,73,77,72]
# good readers
GR=[1,3,4,8,10,11,14,16,18,19,23,40,54,64,67,71,80]
# poor readers
PR=[6,13,21,26,28,30,32,36,38,43,47,49,53,57,62,78,76,60,74,68,42,66,73,77,72]

All_R = [dirlist[i] for i in All]
Good_R = [dirlist[i] for i in GR]
Poor_R = [dirlist[i] for i in PR]

All_R.sort()
Good_R.sort()
Poor_R.sort()

#%%
# Select the participant sample
Exp = All_R
# get empty lists
Att_LN_evoked_list = list()
Att_HN_evoked_list = list()
Lex_LN_evoked_list = list()
Lex_HN_evoked_list = list()

# Differences
Diff_A_evoked_list = list()
Diff_L_evoked_list = list()

n_chan = 204;

# Get the data organize in subject x time x channels (only grad) for each condition
Att_LN= np.zeros((len(Exp),n_chan,601)) #1441
Att_HN= np.zeros((len(Exp),n_chan,601))
Lex_LN= np.zeros((len(Exp),n_chan,601))
Lex_HN= np.zeros((len(Exp),n_chan,601))

# Difference
Diff_A= np.zeros((len(Exp),n_chan,601))
Diff_L= np.zeros((len(Exp),n_chan,601))

# REad in the data and get the matrix to be analyzed
for x in range(0,len(Exp)):
    Ep_path= '/mnt/scratch/NLR_MEG/'+Exp[x]+'/epochs/'
    os.chdir(Ep_path)  # go into the tsss dir 
    file = 'All_40-sss_'+Exp[x]+'-epo.fif'
    epochs = mne.read_epochs(file, verbose=True)
    Att_LN_evoked = epochs['word_c254_p20_dot'].average().pick(['grad']) # n Preproc 5 no need of i .shift_time(-0.239,relative=False) # select epochs from a group of conditions
    Att_HN_evoked = epochs['word_c254_p80_dot'].average().pick(['grad']) 
    Lex_LN_evoked = epochs['word_c254_p20_word'].average().pick(['grad']) 
    Lex_HN_evoked = epochs['word_c254_p80_word'].average().pick(['grad']) 
 
    Att_LN_evoked_list.append(Att_LN_evoked)
    Att_HN_evoked_list.append(Att_HN_evoked)
    Lex_LN_evoked_list.append(Lex_LN_evoked)
    Lex_HN_evoked_list.append(Lex_HN_evoked)
    
    Att_LN[x,:,:]=Att_LN_evoked._data # 3D matrix
    Att_HN[x,:,:]=Att_HN_evoked._data  
    Lex_LN[x,:,:]=Lex_LN_evoked._data
    Lex_HN[x,:,:]=Lex_HN_evoked._data 
#    
#    Att_WE = Att_LN_evoked
#    Att_WE._data = Att_LN_evoked._data - Att_HN_evoked._data # 3D matrix
#    Lex_WE = Att_LN_evoked
#    Lex_WE._data = Lex_LN_evoked._data - Lex_HN_evoked._data 
#        
#    Diff_A[x,:,:]= Att_WE._data # 3D matrix
#    Diff_L[x,:,:]= Lex_WE._data 
#
#    Diff_A_evoked_list.append(Att_WE)
#    Diff_L_evoked_list.append(Lex_WE)
#
## Select the condition to be compared
#X = [Att_LN.transpose(0, 2, 1),
#    Att_HN.transpose(0, 2, 1)]  # Transpose so that is sj x time x chn

epochs = Att_LN_evoked_list[0] # to get general info about chn and time line

waveform1= Lex_LN_evoked_list
waveform2= Lex_HN_evoked_list
#%%
################## UNCONSTRAINED CLUSTER BASED PERMUTATION ##################################
# Sensor connectivity
connectivity, ch_names = find_ch_adjacency(epochs.info, ch_type='grad')

print(type(connectivity))  # it's a sparse matrix!

plt.imshow(connectivity.toarray(), cmap='gray', origin='lower',
           interpolation='nearest')
plt.xlabel('{} Gradiometers'.format(len(ch_names)))
plt.ylabel('{} Gradiometers'.format(len(ch_names)))
plt.title('Between-sensor adjacency')

#%%
# Permutation
# set cluster threshold
threshold = 5
# set family-wise p-value
p_accept = 0.001
#tfce = dict(start=.2, step=.2)

#cluster_stats = spatio_temporal_cluster_test(X, n_permutations=1024, #max_step=5,
#                                             threshold= threshold, tail=1,
#                                             n_jobs=18, buffer_size=2048,
#                                             connectivity=connectivity)
#T_obs, clusters, p_values, _ = cluster_stats

t_threshold = -stats.distributions.t.ppf(p_accept / 2., len(All_R) - 1)

X = Att_LN.transpose(0, 2, 1) - Att_HN.transpose(0, 2, 1)

T_obs, clusters, p_values, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X, n_permutations=3000, adjacency=connectivity, n_jobs=18, tail=1,
                                       threshold=t_threshold, buffer_size=2048,
                                       verbose=True)
    
good_cluster_inds = np.where(p_values < p_accept)[0]

# Plot the resutls
# configure variables for visualization
colors = {"Word": "crimson", "Noise": "black"}
# conditions to be plotted

# loop over clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for F stat
    f_map = T_obs[time_inds, ...].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = epochs.times[time_inds]

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

    # plot average test statistic and mark significant sensors
    f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs.info, tmin=0)
    f_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap='Reds',
                          vmin=np.min, vmax=np.max, show=False,show_names=False,
                          colorbar=False, mask_params=dict(markersize=10))
    image = ax_topo.images[0]

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.2)
    title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += "s (mean)"
    plot_compare_evokeds(dict(Word = waveform1, Noise = waveform2), title=title, picks=ch_inds, axes=ax_signals, ci=True,
                         colors=colors, show=False, combine ='mean', show_sensors='upper right', 
                         split_legend=True, truncate_yaxis='auto')

    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()
#%%
os.chdir(root)    
os.chdir('figures')
plt.savefig('Sensor_spatio_temporal_cluster_1samp_test_lex.png',dpi=600,papertype='letter',format='png')
plt.savefig('Sensor_spatio_temporal_cluster_1samp_test_lex.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')    
#%%
#################### Cluster based permutation on a selected cluster of electrodes ###################
from mne.stats import permutation_cluster_1samp_test
# condition to be compared
erf_1_list = Att_LN_evoked_list
erf_2_list = Att_HN_evoked_list
# get starting empty matrices
erf_1= np.zeros((len(Exp),481)) 
erf_2= np.zeros((len(Exp),481))

# channels
picks_F= ['MEG0123', 'MEG0122', 'MEG0343','MEG0342', 'MEG0212', 'MEG0213', 'MEG0133', 'MEG0132', 'MEG1513', 'MEG1512', 'MEG0242','MEG0243']
picks_T= ['MEG1513', 'MEG1512', 'MEG0242','MEG0243', 'MEG1523', 'MEG1522', 'MEG1612', 'MEG1613', 'MEG1722','MEG1723', 'MEG1642', 'MEG1643']
picks_V= ['MEG0132', 'MEG0133', 'MEG0142', 'MEG0143', 'MEG0242', 'MEG0243', 'MEG1512', 'MEG1513']
picks_V= ['MEG0132', 'MEG0133', 'MEG0142', 'MEG0143', 'MEG0242', 'MEG0243', 'MEG1512', 'MEG1513', \
          'MEG1522', 'MEG1523', 'MEG1542', 'MEG1543', 'MEG0212', 'MEG0213', \
          'MEG0232', 'MEG0233', 'MEG0442', 'MEG0443', 'MEG0742', 'MEG0743', 'MEG1822', 'MEG1823', \
          'MEG1812', 'MEG1813', 'MEG1622', 'MEG1623', 'MEG1612', 'MEG1613']
picks_L= ['MEG0132', 'MEG0133', 'MEG0142', 'MEG0143', 'MEG0232', 'MEG0233','MEG0242', 'MEG0243', \
          'MEG0442', 'MEG0443', 'MEG1512', 'MEG1513', 'MEG1522', 'MEG1523','MEG1542', 'MEG1543', \
          'MEG1612', 'MEG1613']
picks_R = ['MEG1322','MEG1323','MEG1332','MEG1333','MEG1342','MEG1343','MEG1432','MEG1433', \
           'MEG1442','MEG1443','MEG2612','MEG2613','MEG2622','MEG2623','MEG2632','MEG2633']
# selected cluster
picks = picks_L

for s in range(0,len(Exp)):
    erf_1_l= erf_1_list[s].copy().pick_channels(picks).crop(tmin= 0, tmax= 0.8) # get the channel subset
    erf_2_l= erf_2_list[s].copy().pick_channels(picks).crop(tmin= 0, tmax= 0.8) # get the channel subset 
    erf_1[s]= np.mean(erf_1_l._data, axis =0) # by column avg for the first column -chn
    erf_2[s]= np.mean(erf_2_l._data, axis =0) # by column avg for the first column -chn

XX = erf_1 - erf_2
threshold =5.0
p_accept = 0.01
t_threshold = -stats.distributions.t.ppf(p_accept / 2., len(All_R) - 1)
T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_1samp_test(XX,
                             n_permutations=5000, threshold=t_threshold, tail=0)

# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(range(0,len(clusters)), cluster_p_values):
    if p_val <= p_accept:
        print (clusters[c]) # get the bounderies of the ssignificant cluster
        T_obs_plot[c] = T_obs[c]
        
# the bounderies can be translated into time using this vector (e.g. times[149])
times = 1e3 * erf_1_l.times  # change unit to ms
        
mne.viz.plot_compare_evokeds(dict(Word=erf_1_list, Noise=erf_2_list), picks=picks, colors=colors, \
                             ylim=dict(grad=[-15, 10]), combine= 'mean', ci= True, legend='lower right', title="Left temporal sensors") #show_sensors='upper right', 
plt.plot([0.258,0.383], [-14,-14])
plt.plot([0.615,0.688], [-14,-14])
#%%
os.chdir(root)    
os.chdir('figures')
plt.savefig('Sensor_cluster_fix.png',dpi=600,papertype='letter',format='png')
plt.savefig('Sensor_cluster_fix.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')  
#%%
from mne.stats import permutation_t_test

HN= mne.grand_average(Lex_HN_evoked_list)
LN= mne.grand_average(Lex_LN_evoked_list)
word_diff = mne.combine_evoked([LN, HN], weights = [1, -1])

picks = mne.pick_types(epochs.info, meg='grad', eeg=False, stim=False, eog=False,
                       exclude='bads')
times = epochs.times
temporal_mask = np.logical_and(0.281 <= times, times <= 0.381)
data = X.transpose(0, 2, 1)
data = np.mean(data[:, :, temporal_mask], axis=2)

n_permutations = 2000
T0, p_values, H0 = permutation_t_test(data, n_permutations, n_jobs=18)

significant_sensors = picks[p_values <= 0.001]
significant_sensors_names = [epochs.ch_names[k] for k in significant_sensors]
print("Number of significant sensors : %d" % len(significant_sensors))
print("Sensors names : %s" % significant_sensors_names)

evoked = mne.EvokedArray(-np.log10(p_values)[:, np.newaxis],
                         word_diff.info, tmin=0.)

# Extract mask and indices of active sensors in the layout
stats_picks = mne.pick_channels(evoked.ch_names, significant_sensors_names)
mask = p_values[:, np.newaxis] <= 0.01

evoked.plot_topomap(ch_type='grad', times=[0], scalings=1,
                    time_format=None, cmap='Reds', vmin=0., vmax=np.max,
                    units='-log10(p)', cbar_fmt='-%0.1f', mask=mask,
                    size=3, show_names=lambda x: x[3:] + ' ' * 20,
                    time_unit='s')
