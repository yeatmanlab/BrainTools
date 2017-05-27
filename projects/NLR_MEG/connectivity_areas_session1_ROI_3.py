# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:05:40 2016

@author: sjjoo
"""
#%%
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
import seaborn as sns

set_config('MNE_MEMMAP_MIN_SIZE', '1M')
set_config('MNE_CACHE_DIR', '.tmp')

mne.set_config('MNE_USE_CUDA', 'true')

this_env = copy.copy(os.environ)
fs_dir = '/mnt/diskArray/projects/avg_fsurfer'
this_env['SUBJECTS_DIR'] = fs_dir

raw_dir = '/mnt/scratch/NLR_MEG3'

os.chdir(raw_dir)

subs = ['NLR_102_RS','NLR_103_AC','NLR_110_HH','NLR_127_AM',
        'NLR_130_RW','NLR_132_WP','NLR_133_ML','NLR_145_AC','NLR_151_RD',
        'NLR_152_TC','NLR_160_EK','NLR_161_AK','NLR_163_LF','NLR_164_SF',
        'NLR_170_GM','NLR_172_TH','NLR_174_HS','NLR_179_GM','NLR_180_ZD',
        'NLR_187_NB','NLR_203_AM','NLR_204_AM','NLR_205_AC','NLR_206_LM',
        'NLR_207_AH','NLR_211_LB','NLR_150_MG'
        ] # 'NLR_202_DD','NLR_105_BB','NLR_150_MG','NLR_201_GS',

brs = [87, 102, 78, 115,
       91, 121, 77, 91, 93,
       88, 75, 90, 66, 59,
       81, 84, 81, 72, 71, 
       121,75, 66, 90, 93, 
       101, 56, 93] #75 101, 93,
brs = np.array(brs)

age = [125, 132, 138, 109,
       138, 108, 98, 105, 87,
       131, 123, 95, 112, 133,
       152, 103, 89, 138, 93,
       117, 122, 109, 90, 111,
       86, 147]
age = np.divide(age, 12)
# Session 1
# subs are synced up with session1 folder names...
#
session1 = ['102_rs160618','103_ac150609',
            '110_hh160608','127_am161004','130_rw151221',
            '132_wp160919','133_ml151124','145_ac160621',
            '151_rd160620','152_tc160422','160_ek160627',
            '161_ak160627','163_lf160707',
            '164_sf160707','170_gm160613','172_th160614',
            '174_hs160620','179_gm160701','180_zd160621',
            '187_nb161017','203_am150831',
            '204_am150829','205_ac151208','206_lm151119',
            '207_ah160608','211_lb160617','150_mg160606'
            ] #'202_dd150919'(# of average is zero) '105_bb150713'(# of average is less than 10)
              #,(# of average is less than 20) '201_gs150729'(# of average is less than 10)

n_subjects = len(subs)

c_table = (    (0.6510,    0.8078,    0.8902), # Blue, Green, Red, Orange, Purple, yellow 
    (0.1216,    0.4706,    0.7059),
    (0.6980,    0.8745,    0.5412),
    (0.2000,    0.6275,    0.1725),
    (0.9843,    0.6039,    0.6000),
    (0.8902,    0.1020,    0.1098),
    (0.9922,    0.7490,    0.4353),
    (1.0000,    0.4980,         0),
    (0.7922,    0.6980,    0.8392),
    (0.4157,    0.2392,    0.6039),
    (1.0000,    1.0000,    0.6000),
    (0.6941,    0.3490,    0.1569))

method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2

fname_data = op.join(raw_dir, 'session1_data.npy')

m1 = np.transpose(brs) >= 90

m2 = np.logical_not(m1)

#m1[19] = False

m2[12] = False
m2[16] = False
m1[26] = False
m2[26] = False
#m2[15] = False

good_readers = np.where(m1)[0]
poor_readers = np.where(m2)[0]
all_subject = []
all_subject.extend(good_readers)
all_subject.extend(poor_readers)
all_subject.sort()

poor_subs = []
for n in np.arange(0,len(poor_readers)):
    poor_subs.append(subs[poor_readers[n]])

#m1 = np.transpose(age) > 9
#
#m2 = np.logical_not(m1)
#
#m2[12] = False
#m2[16] = False
#m2[26] = False
#old_readers = np.where(m1)[0]
#young_readers = np.where(m2)[0]
#
#all_readers = []
#all_readers.extend(good_readers)
#all_readers.extend(poor_readers)
#all_readers.sort()

#%%
"""
Here we do the real deal...
"""            
# Session 1
load_data = False

method = "dSPM" 
snr = 3.
lambda2 = 1. / snr ** 2
#conditions1 = [0, 2, 4, 6, 8, 16, 18, 20, 22, 24] # Lets compare word vs. scramble
conditions1 = ['word_c254_p20_dot', 'word_c254_p50_dot', 'word_c137_p20_dot',
               'word_c254_p80_dot', 'word_c137_p80_dot', #'bigram_c254_p20_dot',
#               'bigram_c254_p50_dot', 'bigram_c137_p20_dot',
               'word_c254_p20_word', 'word_c254_p50_word', 'word_c137_p20_word',
               'word_c254_p80_word', 'word_c137_p80_word', #'bigram_c254_p20_word',
#               'bigram_c254_p50_word', 'bigram_c137_p20_word'
               ]
#    conditions2 = [16, 22] # Lets compare word vs. scramble
X13 = np.empty((20484, 601, n_subjects, len(conditions1)))
#word_data = np.empty((20484, 421, n_subjects, len(conditions1[8:])))
fs_vertices = [np.arange(10242)] * 2

n_epochs = np.empty((n_subjects,len(conditions1)))

all_con = np.empty((4,4,11,541,n_subjects))
for n, s in enumerate(session1):
    """ Read HCP labels """
    #%%
    labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', surf_name='white', subjects_dir=fs_dir)
    
    PHT_label_lh = [label for label in labels if label.name == 'L_PHT_ROI-lh'][0]
    PHT_label_rh = [label for label in labels if label.name == 'R_PHT_ROI-rh'][0]
    
    TE1p_label_lh = [label for label in labels if label.name == 'L_TE1p_ROI-lh'][0]
    TE1p_label_rh = [label for label in labels if label.name == 'R_TE1p_ROI-rh'][0]
    
    
    TE2p_label_lh = [label for label in labels if label.name == 'L_TE2p_ROI-lh'][0]
    TE2p_label_rh = [label for label in labels if label.name == 'R_TE2p_ROI-rh'][0]
    
    PH_label_lh = [label for label in labels if label.name == 'L_PH_ROI-lh'][0]
    PH_label_rh = [label for label in labels if label.name == 'R_PH_ROI-rh'][0]
    
    FFC_label_lh = [label for label in labels if label.name == 'L_FFC_ROI-lh'][0]
    FFC_label_rh = [label for label in labels if label.name == 'R_FFC_ROI-rh'][0]
    
    
    IFSp_label_lh = [label for label in labels if label.name == 'L_IFSp_ROI-lh'][0]
    IFSp_label_rh = [label for label in labels if label.name == 'R_IFSp_ROI-rh'][0]
    
    IFJp_label_lh = [label for label in labels if label.name == 'L_IFJp_ROI-lh'][0]
    IFJp_label_rh = [label for label in labels if label.name == 'R_IFJp_ROI-rh'][0]
    
    IFJa_label_lh = [label for label in labels if label.name == 'L_IFJa_ROI-lh'][0]
    IFJa_label_rh = [label for label in labels if label.name == 'R_IFJa_ROI-rh'][0]
    
    a45_label_lh = [label for label in labels if label.name == 'L_45_ROI-lh'][0]
    a45_label_rh = [label for label in labels if label.name == 'R_45_ROI-rh'][0]
    
    a44_label_lh = [label for label in labels if label.name == 'L_44_ROI-lh'][0]
    a44_label_rh = [label for label in labels if label.name == 'R_44_ROI-rh'][0]
    
    PGi_label_lh = [label for label in labels if label.name == 'L_PGi_ROI-lh'][0]
    PGi_label_rh = [label for label in labels if label.name == 'R_PGi_ROI-rh'][0]
    
    PGs_label_lh = [label for label in labels if label.name == 'L_PGs_ROI-lh'][0]
    PGs_label_rh = [label for label in labels if label.name == 'R_PGs_ROI-rh'][0]
    
    STSvp_label_lh = [label for label in labels if label.name == 'L_STSvp_ROI-lh'][0]
    STSvp_label_rh = [label for label in labels if label.name == 'R_STSvp_ROI-rh'][0]
    
    STSdp_label_lh = [label for label in labels if label.name == 'L_STSdp_ROI-lh'][0]
    STSdp_label_rh = [label for label in labels if label.name == 'R_STSdp_ROI-rh'][0]
    
    TPOJ1_label_lh = [label for label in labels if label.name == 'L_TPOJ1_ROI-lh'][0]
    TPOJ1_label_rh = [label for label in labels if label.name == 'R_TPOJ1_ROI-rh'][0]
    
    V1_label_lh = [label for label in labels if label.name == 'L_V1_ROI-lh'][0]
    V1_label_rh = [label for label in labels if label.name == 'R_V1_ROI-rh'][0]
    
    """ Frontal """
    broca_vertices = a44_label_lh.vertices
    
    temp = a45_label_lh.vertices
    broca_vertices = np.unique(np.append(broca_vertices, temp))
    
    temp = IFSp_label_lh.vertices
    broca_vertices = np.unique(np.append(broca_vertices, temp))
    
    #temp = temp3.in_label(IFJp_label_lh)
    #broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))
    #
    #temp = temp3.in_label(IFJa_label_lh)
    #broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))
    
    """ Ventral """
    ventral_vertices = TE2p_label_lh.vertices
    
    #temp = temp3.in_label(PH_label_lh)
    #ventral_vertices = np.unique(np.append(ventral_vertices, temp.vertices[0]))
    
    """ Temporal """
    w_vertices = PGi_label_lh.vertices
    
    temp = PGs_label_lh.vertices
    w_vertices = np.unique(np.append(w_vertices, temp))
    
    """ V1 """
    v1_vertices = V1_label_lh.vertices

    os.chdir(os.path.join(raw_dir,session1[n]))
    os.chdir('inverse')
    
    fn = 'Conditions_40-sss_eq_'+session1[n]+'-ave.fif'
    fn_inv = session1[n] + '-inv.fif'
    inv = mne.minimum_norm.read_inverse_operator(fn_inv, verbose=None)
    
    temp_frontal_label_lh = mne.Label(broca_vertices, hemi = 'lh', subject = subs[n])
    temp_temporal_label_lh = mne.Label(w_vertices, hemi = 'lh', subject = subs[n])
    temp_ventral_label_lh = mne.Label(ventral_vertices, hemi = 'lh', subject = subs[n])
    temp_v1_label_lh = mne.Label(v1_vertices, hemi = 'lh', subject = subs[n])
    
    frontal_label_lh = temp_frontal_label_lh.morph(subject_from='fsaverage', subject_to=subs[n], subjects_dir=fs_dir, n_jobs=18) 
    temporal_label_lh = temp_temporal_label_lh.morph(subject_from='fsaverage', subject_to=subs[n], subjects_dir=fs_dir, n_jobs=18) 
    ventral_label_lh = temp_ventral_label_lh.morph(subject_from='fsaverage', subject_to=subs[n], subjects_dir=fs_dir, n_jobs=18) 
    v1_label_lh = temp_v1_label_lh.morph(subject_from='fsaverage', subject_to=subs[n], subjects_dir=fs_dir, n_jobs=18) 
    
    os.chdir(os.path.join(raw_dir,session1[n]))
    os.chdir('epochs')
    fn = 'All_40-sss_'+session1[n]+'-epo.fif'
    
    epochs = mne.read_epochs(fn)
    eid = epochs.events[:,2] == 104
    
    epo = epochs[eid]
    epo.crop(0., 0.9)
    
    snr = 1.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    stcs = mne.minimum_norm.apply_inverse_epochs(epo, inv, lambda2, method,
                                pick_ori="normal", return_generator=True)
    

    # Calculate coherence between areas
    os.chdir(os.path.join(fs_dir,subs[n]))
    os.chdir('bem')
    fn = subs[n] + '-ico-5-src.fif'
    src = mne.read_source_spaces(fn, patch_stats=False, verbose=None)
    src = inv['src']
    
    test_labels = []
    test_labels.append(frontal_label_lh)
    test_labels.append(temporal_label_lh)
    test_labels.append(ventral_label_lh)
    test_labels.append(v1_label_lh)
    
    label_ts = mne.extract_label_time_course(stcs, test_labels, src,
                                             mode='mean_flip',
                                             allow_empty=True,
                                             return_generator=False)
    fmin, fmax = 8., 40.
    sfreq = 600  # the sampling frequency
    
    cwt_frequencies = np.arange(fmin, fmax, 3)
    ylabels = np.arange(int(fmin), int(fmax), 3)
    cwt_n_cycles = cwt_frequencies / 8.

    con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
        label_ts, method='wpli2_debiased', mode='cwt_morlet', sfreq=sfreq,
        fmin=fmin, fmax=fmax, mt_adaptive=True, n_jobs=18, cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles)
    
    all_con[:,:,:,:,n] = con
    
#    n_rows, n_cols = con.shape[:2]
#    names = ['Frontal', 'Temporal', 'Ventral', 'V1']
#    k = 1
#    plt.figure()
#    plt.clf()
#    for i in np.arange(0,4,1):
#        for j in np.arange(0,4,1):
#            plt.subplot(4,4,k)
#            mat = con[i,j,:,:]
#            ax = sns.heatmap(np.flipud(mat), xticklabels=60, yticklabels = ylabels[::-1], vmin=-0.35, vmax=0.35)
#            k = k + 1
#            if j == 0:
#                ax.set_ylabel(names[i])
#            if i == (n_rows - 1):
#                ax.set_xlabel(names[j])
#            if i == 0:
#                ax.set_title(names[j])
#    
#    plt.show()
