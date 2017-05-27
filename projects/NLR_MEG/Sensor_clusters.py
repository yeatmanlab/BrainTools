#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 12:12:16 2017

@author: sjjoo
"""
import sys
import mne
import matplotlib.pyplot as plt
import imageio
from mne.utils import run_subprocess, logger
import os
from os import path as op
import copy
import shutil
import numpy as np
from numpy.random import randn
from scipy import stats as stats
import scipy.io as sio
import time
from functools import partial
from mne.stats import permutation_t_test

from mne import set_config

raw_dir = '/mnt/scratch/adult'

subs = ['huber_libby','maddox_ross','mccloy_dan','mizrahi_julia',
        'wronkiewicz_mark'
        ]
session = ['huber_libby160324', 'maddox_ross160317','mccloy_dan160322',
           'mizrahi_julia160407','wronkiewicz_mark160408']
conditions1 = ['word_c254_p20_dot', 'word_c254_p50_dot', 'word_c137_p20_dot',
               'word_c254_p80_dot', 'word_c137_p80_dot', #'bigram_c254_p20_dot',
#               'bigram_c254_p50_dot', 'bigram_c137_p20_dot',
               'word_c254_p20_word', 'word_c254_p50_word', 'word_c137_p20_word',
               'word_c254_p80_word', 'word_c137_p80_word', #'bigram_c254_p20_word',
#               'bigram_c254_p50_word', 'bigram_c137_p20_word'
               ]

#%%
for n, s in enumerate(session):
    os.chdir(os.path.join(raw_dir,session[n]))
    os.chdir('epochs')
    
    fn = 'All_40-sss_'+session[n]+'-epo.fif'
    epochs = mne.read_epochs(fn, proj=True, preload=True)
    epochs = epochs.pick_types(meg='grad')
    
    id_word = np.where(epochs.events[:,2] == 202)[0]
    id_noise = np.where(epochs.events[:,2] == 204)[0]
    
    data = epochs.get_data()
    
    times = epochs.times
    
    temporal_mask = np.logical_and(0.13 <= times, times <= 0.17)
    
    idMax = np.min([len(id_word),len(id_noise)])
    
    data_contrast = np.subtract(data[id_noise[:idMax],:,:], data[id_word[:idMax],:,:])
    
    data = np.mean(data_contrast[:, :, temporal_mask], axis=2)
    
    n_permutations = 10e3
#    T0, p_values, H0 = permutation_t_test(data, n_permutations, n_jobs=18)
    t_values = mne.stats.ttest_1samp_no_p(data)
    
#    significant_sensors = np.logical_and(p_values < 0.05, T0 > 0)
    significant_sensors = t_values > 2
#    significant_sensors_names = [epochs.ch_names[k] for k in significant_sensors]
    significant_sensors_names = []
    
    for m, k in enumerate(significant_sensors):
        if k:
            significant_sensors_names.append(epochs.ch_names[m])
    
    print("Number of significant sensors : %d" % len(significant_sensors))
    print("Sensors names : %s" % significant_sensors_names)
    
    evoked = mne.EvokedArray(t_values[:, np.newaxis], #-np.log10(p_values)
                         epochs.info, tmin=0.)

    # Extract mask and indices of active sensors in layout
    stats_picks = mne.pick_channels(evoked.ch_names, significant_sensors_names)
    mask = t_values[:, np.newaxis] > 2
    
    evoked.plot_topomap(ch_type='grad', times=[0], scale=1,
                        time_format=None, cmap='RdBu_r', vmin=np.min, vmax=np.max,
                        unit='t-value', cbar_fmt='-%0.1f', mask=mask,
                        size=3, show_names=lambda x: x[4:] + ' ' * 20)