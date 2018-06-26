#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:30:55 2017

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
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)

from mne import set_config
import matplotlib.font_manager as font_manager

set_config('MNE_MEMMAP_MIN_SIZE', '1M')
set_config('MNE_CACHE_DIR', '.tmp')

mne.set_config('MNE_USE_CUDA', 'true')

this_env = copy.copy(os.environ)
fs_dir = '/mnt/diskArray/projects/avg_fsurfer'
this_env['SUBJECTS_DIR'] = fs_dir

raw_dir = '/mnt/scratch/NLR_MEG4'
#raw_dir = '/mnt/scratch/NLR_MEG_EOG2'

os.chdir(raw_dir)

subs = ['NLR_102_RS','NLR_103_AC','NLR_105_BB','NLR_110_HH','NLR_127_AM',
        'NLR_130_RW','NLR_132_WP','NLR_133_ML','NLR_145_AC','NLR_150_MG',
        'NLR_151_RD','NLR_152_TC','NLR_160_EK','NLR_161_AK','NLR_163_LF',
        'NLR_164_SF','NLR_170_GM','NLR_172_TH','NLR_174_HS','NLR_179_GM',
        'NLR_180_ZD','NLR_187_NB','NLR_201_GS','NLR_203_AM',
        'NLR_204_AM','NLR_205_AC','NLR_206_LM','NLR_207_AH','NLR_211_LB',
        'NLR_GB310','NLR_KB218','NLR_JB423','NLR_GB267','NLR_JB420',
        'NLR_HB275','NLR_197_BK','NLR_GB355','NLR_GB387','NLR_HB205',
        'NLR_IB217','NLR_IB319','NLR_JB227','NLR_JB486','NLR_KB396',
        'NLR_IB357']
session1 = ['102_rs160618','103_ac150609','105_bb150713','110_hh160608','127_am151022',
       '130_rw151221','132_wp160919','133_ml151124','145_ac160621','150_mg160606',
       '151_rd160620','152_tc160422','160_ek160627','161_ak160627','163_lf160707',
       '164_sf160707','170_gm160613','172_th160614','174_hs160620','179_gm160701',
       '180_zd160621','187_nb161017','201_gs150818','203_am150831',
       '204_am150829','205_ac151123','206_lm151119','207_ah160608','211_lb160617',
       'nlr_gb310170614','nlr_kb218170619','nlr_jb423170620','nlr_gb267170620','nlr_jb420170621',
       'nlr_hb275170622','197_bk170622','nlr_gb355170606','nlr_gb387170608','nlr_hb205170825',
       'nlr_ib217170831','nlr_ib319170825','nlr_jb227170811','nlr_jb486170803','nlr_kb396170808',
       'nlr_ib357170912']

n_subjects = len(subs)

#%%
""" CHANGE the file name here !!! """
fname_data = op.join(raw_dir, 'session1_data_pseudo.npy')

method = "dSPM" 
snr = 3.
lambda2 = 1. / snr ** 2

conditions1 = ['word_c254_p20_dot', 'word_c254_p50_dot', 'word_c137_p20_dot',
               'word_c254_p80_dot', 'word_c137_p80_dot', 'bigram_c254_p20_dot',
               'bigram_c254_p50_dot', 'bigram_c137_p20_dot',
               'word_c254_p20_word', 'word_c254_p50_word', 'word_c137_p20_word',
               'word_c254_p80_word', 'word_c137_p80_word', 'bigram_c254_p20_word',
               'bigram_c254_p50_word', 'bigram_c137_p20_word'
               ]
#conditions2 = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
conditions2 = [5, 6, 7, 13, 14, 15]

#X13 = np.empty((20484, 481, n_subjects, len(conditions2)))
X13 = np.empty((20484, 601, n_subjects, len(conditions2)))
fs_vertices = [np.arange(10242)] * 2

n_epochs = np.empty((n_subjects,len(conditions2)))
       

for n, ss in enumerate(session1):  
    os.chdir(os.path.join(raw_dir,session1[n]))
    os.chdir('inverse')
    
    fn = 'Conditions_40-sss_eq_'+session1[n]+'-ave.fif'

    fn_inv = session1[n] + '-depth8-inv.fif'
#    fn_inv = session1[n] + '-40-sss-meg-inv.fif'
    
    inv = mne.minimum_norm.read_inverse_operator(fn_inv, verbose=None)
            
    for iCond, s in enumerate(conditions2):
        evoked = mne.read_evokeds(fn, condition=conditions1[s], baseline=(None,0), kind='average', proj=True)
#            mne.viz.plot_snr_estimate(evoked, inv)
#            os.chdir(os.path.join(raw_dir,session1[n]))
#            os.chdir('covariance')
#            fn_cov = session1[n] + '-40-sss-cov.fif'
#            cov = mne.read_cov(fn_cov)
#            evoked.plot()
#            evoked.plot_topomap(times=np.linspace(0.05, 0.15, 11), ch_type='mag')
#            evoked.plot_white(cov)
#            os.chdir(os.path.join(raw_dir,session1[n]))
#            os.chdir('inverse')
    
        n_epochs[n][iCond] = evoked.nave
        
        stc = mne.minimum_norm.apply_inverse(evoked,inv,lambda2, method=method, pick_ori='normal') #None
        

#        plt.figure()
#        plt.plot(1e3 * stc.times, stc.data[::100, :].T)
#        plt.xlabel('time (ms)')
#        plt.ylabel('%s value' % method)
#        plt.show()

        stc.crop(-0.1, 0.9)
        
        tstep = stc.tstep
        times = stc.times
        # Average brain
        """
        One should check if morph map is current and correct. Otherwise, it will spit out and error.
        Check SUBJECTS_DIR/morph-maps
        """
        morph_mat = mne.compute_morph_matrix(subs[n], 'fsaverage', stc.vertices, 
                                             fs_vertices, smooth=20,
                                             subjects_dir=fs_dir)
        stc_fsaverage = stc.morph_precomputed('fsaverage', fs_vertices, morph_mat, subs[n])
        
#        tmin, tmax = 0.080, 0.120
#        stc_mean = stc_fsaverage.copy().crop(tmin, tmax).mean()
#        
#        labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', surf_name='white', subjects_dir=fs_dir)
#        V1_label_lh = [label for label in labels if label.name == 'L_V1_ROI-lh'][0]
#        V1_label_rh = [label for label in labels if label.name == 'R_V1_ROI-rh'][0]
#        
#        stc_mean_label = stc_mean.in_label(V1_label_lh)
#        data = np.abs(stc_mean_label.data)
#        stc_mean_label.data[data < 0.6 * np.max(data)] = 0.
#        
#        func_labels, _ = mne.stc_to_label(stc_mean_label, src='fsaverage', subjects_dir=fs_dir, smooth=False)
#        
#        stc_anat_label = stc_fsaverage.in_label(V1_label_lh)
#        pca_anat = stc_fsaverage.extract_label_time_course(V1_label_lh, src='fsaverage', mode='pca_flip')[0]
#        
#        stc_func_label = stc.in_label(func_label)
#        pca_func = stc.extract_label_time_course(func_label, src, mode='pca_flip')[0]
#        
#        # flip the pca so that the max power between tmin and tmax is positive
#        pca_anat *= np.sign(pca_anat[np.argmax(np.abs(pca_anat))])
#        pca_func *= np.sign(pca_func[np.argmax(np.abs(pca_anat))])

#        stc_morph = mne.morph_data(subs[n], 'fsaverage', stc, n_jobs=18,
#                            grade=fs_vertices, subjects_dir=fs_dir)
#        stc_morph.save('%s_loose_morph' % conditions1[iCond])
#            
#            tt = np.arange(0.05, 0.15, 0.01)
#            # plot magnetometer data as topomaps
#            evoked.plot()
#            evoked.plot_topomap(tt, ch_type='mag')
#            
#            # compute a 50 ms bin to stabilize topographies
##            evoked.plot_topomap(tt, ch_type='mag', average=0.05)
#            
#            # plot gradiometer data (plots the RMS for each pair of gradiometers)
#            evoked.plot_topomap(tt, ch_type='grad')
#            
#            # plot magnetometer data as an animation
#            evoked.animate_topomap(ch_type='mag', times=times, frame_rate=10)
#            
#            # plot magnetometer data as topomap at 1 time point : 100 ms
#            # and add channel labels and title
#            evoked.plot_topomap(0.1, ch_type='mag', show_names=True, colorbar=False,
#                                size=6, res=128, title='Auditory response')
#            plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88)
#            
        X13[:,:,n,iCond] = stc_fsaverage.data

os.chdir(raw_dir)
np.save(fname_data, X13)
#np.save('session1_times.npy',times)
#np.save('session1_tstep.npy',tstep)
#np.save('session1_n_averages.npy',n_epochs)
np.save('session1_n_pseudo.npy',n_epochs)