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
import matplotlib.font_manager as font_manager

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
        'NLR_207_AH','NLR_211_LB','NLR_150_MG',
        'NLR_GB310','NLR_KB218','NLR_JB423','NLR_GB267','NLR_JB420',
        'NLR_HB275','NLR_197_BK','NLR_GB355','NLR_GB387',
        'NLR_HB205','NLR_IB319','NLR_JB227','NLR_JB486','NLR_KB396'] # 'NLR_202_DD','NLR_105_BB','NLR_150_MG','NLR_201_GS',

brs = [87, 102, 78, 115,
       91, 121, 77, 91, 93,
       88, 75, 90, 66, 59,
       81, 84, 81, 72, 71, 
       121,75, 66, 90, 93, 
       101, 56, 93,
       76, 83, 69, 93, 68,
       97, 73, 73, 88,
       113, 119, 83, 68, 107] #75 101, 93,

brs = np.array(brs)

swe = [89, 93, 66, 123,
        89, 111, 71, 92, 85,
        88, 72, 93, 55, 55,
        89, 57, 66, 79, 61,
        129, 78, 55, 87, 82,
        89, 55, 91,
        59, 85, 55, 82, 55,
        72, 62, 81, 84,
        121, 110, 86, 68, 105]
swe = np.array(swe)

swe_raw = [62, 76, 42, 83,
        67, 76, 21, 54, 21,
        61, 45, 48, 17, 11,
        70, 19, 10, 57, 12,
        86, 51, 13, 30, 54,
        25, 27, 35,
        10, 66, 18, 18, 20,
        37, 23, 17, 36,
        79, 74, 64, 42, 78]
swe_raw = np.array(swe_raw)

twre = [87, 102, 67, 111,
        85, 110, 67, 84, 87,
        86, 63, 81, 60, 55,
        71, 63, 68, 67, 64,
        127, 73, 59, 88, 79,
        91, 57, 92,
        67, 77, 57, 80, 53,
        66, 58, 85, 79,
        116, 110, 78, 66, 101]
twre = np.array(twre)


age = [125, 132, 138, 109,
       138, 108, 98, 105, 87,
       131, 123, 95, 112, 133,
       152, 103, 89, 138, 93,
       117, 122, 109, 90, 111,
       86, 147, 89,
       94, 148, 121, 87, 121,
       107, 105, 88, 94,
       101, 110, 132, 127, 135]
age = np.divide(age, 12)
# Session 1
# subs are synced up with session1 folder names...
#
session1 = ['102_rs160618','103_ac150609','110_hh160608','127_am161004',
            '130_rw151221','132_wp160919','133_ml151124','145_ac160621','151_rd160620',
            '152_tc160422','160_ek160627','161_ak160627','163_lf160707','164_sf160707',
            '170_gm160613','172_th160614','174_hs160620','179_gm160701','180_zd160621',
            '187_nb161017','203_am150831','204_am150829','205_ac151208','206_lm151119',
            '207_ah160608','211_lb160617','150_mg160606',
            'nlr_gb310170614','nlr_kb218170619','nlr_jb423170620','nlr_gb267170620','nlr_jb420170621',
            'nlr_hb275170622','197_bk170622','nlr_gb355170606','nlr_gb387170608',
            'nlr_hb205170825','nlr_ib319170825','nlr_jb227170811','nlr_jb486170803','nlr_kb396170808'] #'202_dd150919'(# of average is zero) '105_bb150713'(# of average is less than 10)
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

m1 = np.logical_and(np.transpose(swe) > 85, np.transpose(age) <= 12)
m2 = np.logical_and(np.transpose(swe) <= 85, np.transpose(age) <= 12)

m1[16] = False # NLR_174_HS: This subject has a lot of noise in the raw data--should have been discarded
m2[16] = False
m1[15] = False # NLR_172_TH
m2[15] = False

m1[19] = False # NLR_187_NB
m2[19] = False
m1[24] = False # NLR_207_AH
m2[24] = False

m1[32] = False # NLR_HB275
m2[32] = False

m1[26] = False # NLR_150_MG
m2[26] = False
m1[29] = False # NLR_JB423
m2[29] = False
m1[33] = False # NLR_197_BK
m2[33] = False
m1[34] = False # NLR_GB355
m2[34] = False
m1[38] = False # NLR_JB227 n_epochs < 10
m2[38] = False

good_readers = np.where(m1)[0]
poor_readers = np.where(m2)[0]

a1 = np.transpose(age) > 9
a2 = np.logical_not(a1)

a1[16] = False # NLR_174_HS: This subject has a lot of noise in the raw data--should have been discarded
a2[16] = False
a1[15] = False # NLR_172_TH
a2[15] = False
a1[19] = False # NLR_187_NB
a2[19] = False
a1[24] = False # NLR_207_AH
a2[24] = False

a1[32] = False # NLR_HB275
a2[32] = False
a1[33] = False # NLR_197_BK
a2[33] = False
a1[26] = False # NLR_150_MG
a2[26] = False
a1[29] = False # NLR_JB423
a2[29] = False
a1[34] = False # NLR_GB355
a2[34] = False
a1[38] = False # NLR_JB227
a2[38] = False

old_readers = np.where(a1)[0]
young_readers = np.where(a2)[0]

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

fname_data = op.join(raw_dir, 'session1_data_ico5_crop.npy')
#%%
"""
Here we do the real deal...
"""            
# Session 1
load_data = True

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
X13 = np.empty((20484, 481, n_subjects, len(conditions1)))
#word_data = np.empty((20484, 421, n_subjects, len(conditions1[8:])))
fs_vertices = [np.arange(10242)] * 2

n_epochs = np.empty((n_subjects,len(conditions1)))
       
if load_data == False:
    for n, s in enumerate(session1):  
        os.chdir(os.path.join(raw_dir,session1[n]))
        os.chdir('inverse')
        
        fn = 'Conditions_40-sss_eq_'+session1[n]+'-ave.fif'
        fn_inv = session1[n] + '-inv-ico5.fif'
        inv = mne.minimum_norm.read_inverse_operator(fn_inv, verbose=None)
        
        for iCond in np.arange(0,len(conditions1)):
            evoked = mne.read_evokeds(fn, condition=conditions1[iCond], 
                                  baseline=(None,0), kind='average', proj=True)
            
            n_epochs[n][iCond] = evoked.nave
            
            stc = mne.minimum_norm.apply_inverse(evoked,inv,lambda2, method=method, pick_ori="normal")
            
            stc.crop(-0.1, 0.7)
            tmin = stc.tmin
            tstep = stc.tstep
            times = stc.times
            # Average brain
            """
            One should check if morph map is current and correct. Otherwise, it will spit out and error.
            Check SUBJECTS_DIR/morph-maps
            """
            morph_mat = mne.compute_morph_matrix(subs[n], 'fsaverage', stc.vertices,
                                                 fs_vertices, smooth=None,
                                                 subjects_dir=fs_dir)
            stc_fsaverage = stc.morph_precomputed('fsaverage', fs_vertices, morph_mat)
        #    morph_dat = mne.morph_data(subs[n], 'fsaverage', stc, n_jobs=16,
        #                        grade=fs_vertices, subjects_dir=fs_dir)
            X13[:,:,n,iCond] = stc_fsaverage.data

    os.chdir(raw_dir)
    np.save(fname_data, X13)
    np.save('session1_times.npy',times)
    np.save('session1_tstep.npy',tstep)
    np.save('session1_n_averages.npy',n_epochs)
else:
    os.chdir(raw_dir)
    X13 = np.load(fname_data)
    times = np.load('session1_times.npy')
    tstep = np.load('session1_tstep.npy')
    n_epochs = np.load('session1_n_averages.npy')
    tmin = -0.1

#%%
"""
Let's just add or modify subjects
"""
X13 = np.load(fname_data)

add_subject = ['179_gm160701']

n = session1.index(add_subject[0])
os.chdir(os.path.join(raw_dir,session1[n]))
os.chdir('inverse')

fn = 'Conditions_40-sss_eq_'+session1[n]+'-ave.fif'

###############################################################################
""" Different sources """
fn_inv = session1[n] + '-inv-ico5.fif'
fn_inv = session1[n] + '-inv-oct6.fif'
###############################################################################

inv = mne.minimum_norm.read_inverse_operator(fn_inv, verbose=None)

for iCond in np.arange(0,len(conditions1)):
    evoked = mne.read_evokeds(fn, condition=conditions1[iCond], 
                          baseline=(None,0), kind='average', proj=True)
    
    stc = mne.minimum_norm.apply_inverse(evoked,inv,lambda2, method=method, pick_ori=None)
    
    stc.crop(-0.1, 0.7)
    tmin = stc.tmin
    tstep = stc.tstep
    times = stc.times
    # Average brain
    fs_vertices = [np.arange(10242)] * 2
    morph_mat = mne.compute_morph_matrix(subs[n], 'fsaverage', stc.vertices,
                                         fs_vertices, smooth=None,
                                         subjects_dir=fs_dir)
    stc_fsaverage = stc.morph_precomputed('fsaverage', fs_vertices, morph_mat)
    
    X13[:,:,n,iCond] = stc_fsaverage.data

os.chdir(raw_dir)
np.save(fname_data, X13)
    
#%%
""" Read HCP labels """
labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', surf_name='white', subjects_dir=fs_dir) #regexp=aparc_label_name
#aparc_label_name = 'PHT_ROI'#'_IP'#'IFSp_ROI'#'STSvp_ROI'#'STSdp_ROI'#'PH_ROI'#'TE2p_ROI' #'SFL_ROI' #'IFSp_ROI' #'TE2p_ROI' #'inferiortemporal' #'pericalcarine'
#        anat_label = mne.read_labels_from_annot('fsaverage', parc='aparc',surf_name='white',
#               subjects_dir=fs_dir, regexp=aparc_label_name)
#%%
#labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'lh', subjects_dir=fs_dir)
#aud_label = [label for label in labels if label.name == 'L_A1_ROI-lh'][0]
#brain.add_label(aud_label, borders=False)
""" Task effects """

#TE2p_mask_lh = mne.Label.get_vertices_used(TE2p_label[0])
#TE2p_mask_rh = mne.Label.get_vertices_used(TE2p_label[1])

PHT_label_lh = [label for label in labels if label.name == 'L_PHT_ROI-lh'][0]
PHT_label_rh = [label for label in labels if label.name == 'R_PHT_ROI-rh'][0]

TE1p_label_lh = [label for label in labels if label.name == 'L_TE1p_ROI-lh'][0]
TE1p_label_rh = [label for label in labels if label.name == 'R_TE1p_ROI-rh'][0]

TE2p_label_lh = [label for label in labels if label.name == 'L_TE2p_ROI-lh'][0]
TE2p_label_rh = [label for label in labels if label.name == 'R_TE2p_ROI-rh'][0]

TE2a_label_lh = [label for label in labels if label.name == 'L_TE2a_ROI-lh'][0]
TE2a_label_rh = [label for label in labels if label.name == 'R_TE2a_ROI-rh'][0]

TF_label_lh = [label for label in labels if label.name == 'L_TF_ROI-lh'][0]
TF_label_rh = [label for label in labels if label.name == 'R_TF_ROI-rh'][0]

PH_label_lh = [label for label in labels if label.name == 'L_PH_ROI-lh'][0]
PH_label_rh = [label for label in labels if label.name == 'R_PH_ROI-rh'][0]

FFC_label_lh = [label for label in labels if label.name == 'L_FFC_ROI-lh'][0]
FFC_label_rh = [label for label in labels if label.name == 'R_FFC_ROI-rh'][0]

a8C_label_lh = [label for label in labels if label.name == 'L_8C_ROI-lh'][0]
a8C_label_rh = [label for label in labels if label.name == 'R_8C_ROI-rh'][0]

p946v_label_lh = [label for label in labels if label.name == 'L_p9-46v_ROI-lh'][0]
p946v_label_rh = [label for label in labels if label.name == 'R_p9-46v_ROI-rh'][0]

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

LIPd_label_lh = [label for label in labels if label.name == 'L_LIPd_ROI-lh'][0]
LIPd_label_rh = [label for label in labels if label.name == 'R_LIPd_ROI-rh'][0]
LIPv_label_lh = [label for label in labels if label.name == 'L_LIPv_ROI-lh'][0]
LIPv_label_rh = [label for label in labels if label.name == 'R_LIPv_ROI-rh'][0]

PBelt_label_lh = [label for label in labels if label.name == 'L_PBelt_ROI-lh'][0]
PBelt_label_rh = [label for label in labels if label.name == 'R_PBelt_ROI-rh'][0]

PSL_label_lh = [label for label in labels if label.name == 'L_PSL_ROI-lh'][0]
PSL_label_rh = [label for label in labels if label.name == 'R_PSL_ROI-rh'][0]

LBelt_label_lh = [label for label in labels if label.name == 'L_LBelt_ROI-lh'][0]
LBelt_label_rh = [label for label in labels if label.name == 'R_LBelt_ROI-rh'][0]

A1_label_lh = [label for label in labels if label.name == 'L_A1_ROI-lh'][0]
A1_label_rh = [label for label in labels if label.name == 'R_A1_ROI-rh'][0]

MBelt_label_lh = [label for label in labels if label.name == 'L_MBelt_ROI-lh'][0]
MBelt_label_rh = [label for label in labels if label.name == 'R_MBelt_ROI-rh'][0]

RI_label_lh = [label for label in labels if label.name == 'L_RI_ROI-lh'][0]
RI_label_rh = [label for label in labels if label.name == 'R_RI_ROI-rh'][0]

#%%
#new_data = X13[:,:,all_subject,:]
#data1 = np.subtract(np.mean(new_data[:,:,:,[5]],axis=3), np.mean(new_data[:,:,:,[0]],axis=3))
#data1 = np.mean(new_data[:,:,:,[5]],axis=3)
#del new_data

data11 = X13[:,:,all_subject,[5]]
#data11 = X13[:,:,good_readers,5] - X13[:,:,good_readers,8]
data11 = np.transpose(data11,[2,1,0])

stat_fun = partial(mne.stats.ttest_1samp_no_p)

temp3 = mne.SourceEstimate(np.transpose(stat_fun(data11)), fs_vertices, tmin, tstep,subject='fsaverage') # np.transpose(stat_fun(data11))
#temp3 = mne.SourceEstimate(np.transpose(data11[0,:,:]), fs_vertices, tmin, tstep,subject='fsaverage')

#threshold = -scipy.stats.distributions.t.ppf(p_thresh, n_samples - 1)
threshold = 1.7 #1.69 2.46
brain3_1 = temp3.plot(hemi='lh', subjects_dir=fs_dir, views = ['ven'], #views=['lat','ven','med'], #transparent = True,
          initial_time=0.18, clim=dict(kind='value', lims=[1.5, threshold, 7]), background='white', colorbar=False) #np.max(temp3.data[:,:])])) #pos_lims=[0, 4, 4.5] #np.max(temp3.data[:,:])]))

#threshold = 1 #1.69 2.46
#brain3_1 = temp3.plot(hemi='lh', subjects_dir=fs_dir, views = ['ven'], #views=['lat','ven','med'], #transparent = True,
#          initial_time=0.18, clim=dict(kind='value', lims=[0, threshold, 5]))

#brain3_1 = temp3.plot(hemi='both', subjects_dir=fs_dir, views = ['ven'], #views=['lat','ven','med'], #transparent = True,
#          initial_time=0.18, clim=dict(kind='value', lims=[1.5, threshold, 5]))

brain3_1.save_image('LH_180.pdf')

#brain3_1.add_label(PHT_label_lh, borders=True, color=c_table[0])
#brain3_1.add_label(TE2a_label_lh, borders=True, color=c_table[1])
#brain3_1.add_label(TE2p_label_lh, borders=True, color=c_table[1])
#brain3_1.add_label(TF_label_lh, borders=True, color=c_table[1])

#brain3_1.add_label(PH_label_lh, borders=True, color=c_table[1])
#brain3_1.add_label(TE1p_label_lh, borders=True, color=c_table[1])
#brain3_1.add_label(FFC_label_lh, borders=True, color=c_table[3])

#brain3_1.add_label(IFSp_label_lh, borders=True, color=c_table[5])
#brain3_1.add_label(IFJp_label_lh, borders=True, color=c_table[6])
#brain3_1.add_label(IFJa_label_lh, borders=True, color=c_table[7])
#brain3_1.add_label(a45_label_lh, borders=True, color=c_table[8])
#brain3_1.add_label(a44_label_lh, borders=True, color=c_table[8])
#brain3_1.add_label(a8C_label_lh, borders=True, color=c_table[8])
#brain3_1.add_label(p946v_label_lh, borders=True, color=c_table[8])
#
#brain3_1.add_label(PGi_label_lh, borders=True, color=c_table[9])
#brain3_1.add_label(PGs_label_lh, borders=True, color=c_table[9])
#
#brain3_1.add_label(STSvp_label_lh, borders=True, color=c_table[2])
#brain3_1.add_label(STSdp_label_lh, borders=True, color=c_table[2])
#brain3_1.add_label(PBelt_label_lh, borders=True, color=c_table[2])
#brain3_1.add_label(PSL_label_lh, borders=True, color=c_table[2])
#brain3_1.add_label(TE1p_label_lh, borders=True, color=c_table[2])
#brain3_1.add_label(LBelt_label_lh, borders=True, color=c_table[2])
#brain3_1.add_label(A1_label_lh, borders=True, color=c_table[2])
#brain3_1.add_label(MBelt_label_lh, borders=True, color=c_table[2])
#brain3_1.add_label(RI_label_lh, borders=True, color=c_table[2])

#
#brain3_1.add_label(V1_label_lh, borders=True, color='k')
#brain3_1.add_label(LIPd_label_lh, borders=True, color='k')
#brain3_1.add_label(LIPv_label_lh, borders=True, color='k')


#brain3_1.save_movie('Response2Word.mp4',time_dilation = 4.0,framerate = 30)

#%%
#brain3_2 = temp3.plot(hemi='rh', subjects_dir=fs_dir,  views='lat',
#          clim=dict(kind='value', lims=[2.9, 3, np.max(temp3.data[:,:])]),
#          initial_time=0.15)
#brain3_2.add_label(PHT_label_rh, borders=True, color=c_table[0])
#brain3_2.add_label(TE2p_label_rh, borders=True, color=c_table[1])
#brain3_2.add_label(PH_label_rh, borders=True, color=c_table[2])
#brain3_2.add_label(FFC_label_rh, borders=True, color=c_table[3])
#brain3_2.add_label(TE1p_label_rh, borders=True, color=c_table[4])
#brain3_2.add_label(IFSp_label_rh, borders=True, color=c_table[5])
#brain3_2.add_label(IFJp_label_rh, borders=True, color=c_table[6])
#brain3_2.add_label(IFJa_label_rh, borders=True, color=c_table[7])
#brain3_2.add_label(a45_label_rh, borders=True, color=c_table[8])

""" Frontal """
temp = temp3.in_label(a44_label_lh)
broca_vertices = temp.vertices[0]

temp = temp3.in_label(a45_label_lh)
broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))

temp = temp3.in_label(IFSp_label_lh)
broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))

#temp = temp3.in_label(IFJp_label_lh)
#broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))

#temp = temp3.in_label(IFJa_label_lh)
#broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))

""" Ventral """
temp = temp3.in_label(TE2p_label_lh)
ventral_vertices = temp.vertices[0]

temp = temp3.in_label(PH_label_lh)
ventral_vertices = np.unique(np.append(ventral_vertices, temp.vertices[0]))
#
#temp = temp3.in_label(FFC_label_lh)
#ventral_vertices = np.unique(np.append(ventral_vertices, temp.vertices[0]))

""" Temporal """
temp = temp3.in_label(PGi_label_lh)
w_vertices = temp.vertices[0]

temp = temp3.in_label(PGs_label_lh)
w_vertices = np.unique(np.append(w_vertices, temp.vertices[0]))

#temp = temp3.in_label(TPOJ1_label_lh)
#w_vertices = np.unique(np.append(w_vertices, temp.vertices[0]))

""" V1 """
temp = temp3.in_label(V1_label_lh)
v1_vertices = temp.vertices[0]

#%%
del data11

#%%
""" V1 """
#mask = np.logical_and(times >= 0.08, times <= 0.12)
#
#lh_label = temp3.in_label(V1_label_lh)
#data = np.mean(lh_label.data[:,mask],axis=1)
#lh_label.data[data < threshold] = 0.
#
#temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
#                      subjects_dir=fs_dir, connected=False)
#temp = temp3.in_label(temp_labels)
#v1_vertices = temp.vertices[0]
#new_label = mne.Label(v1_vertices, hemi='lh')
#brain3_1.add_label(new_label, borders=True, color='k')

""" Ventral """
mask = np.logical_and(times >= 0.16, times <= 0.20)

lh_label = temp3.in_label(TE2p_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < threshold] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
ventral_vertices = temp.vertices[0]

#lh_label = temp3.in_label(TE2a_label_lh)
#data = np.mean(lh_label.data[:,mask],axis=1)
#lh_label.data[data < threshold] = 0.
#
#temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
#                      subjects_dir=fs_dir, connected=False)
#temp = temp3.in_label(temp_labels)
##ventral_vertices = temp.vertices[0]
#ventral_vertices = np.unique(np.append(ventral_vertices, temp.vertices[0]))
#
#lh_label = temp3.in_label(TF_label_lh)
#data = np.mean(lh_label.data[:,mask],axis=1)
#lh_label.data[data < threshold] = 0.
#
#temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
#                      subjects_dir=fs_dir, connected=False)
#temp = temp3.in_label(temp_labels)
#ventral_vertices = np.unique(np.append(ventral_vertices, temp.vertices[0]))

#ventral_vertices = np.delete(ventral_vertices,[0,1])

new_label = mne.Label(ventral_vertices, hemi='lh')
brain3_1.add_label(new_label, borders=True, color='k')

""" STS """
mask = np.logical_and(times >= 0.22, times <= 0.26)

lh_label = temp3.in_label(STSvp_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < threshold] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
w_vertices = temp.vertices[0]

lh_label = temp3.in_label(STSdp_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < threshold] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
w_vertices = np.unique(np.append(w_vertices, temp.vertices[0]))

lh_label = temp3.in_label(TE1p_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < threshold] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
w_vertices = np.unique(np.append(w_vertices, temp.vertices[0]))

#w_vertices = np.delete(w_vertices,[12])

new_label = mne.Label(w_vertices, hemi='lh')
brain3_1.add_label(new_label, borders=True, color='k')

###############################################################################
""" Auditory """
mask = np.logical_and(times >= 0.22, times <= 0.26)

lh_label = temp3.in_label(PBelt_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
pt_vertices = temp.vertices[0]

lh_label = temp3.in_label(PSL_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
pt_vertices = np.unique(np.append(pt_vertices, temp.vertices[0]))

lh_label = temp3.in_label(LBelt_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
pt_vertices = np.unique(np.append(pt_vertices, temp.vertices[0]))

lh_label = temp3.in_label(A1_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
pt_vertices = np.unique(np.append(pt_vertices, temp.vertices[0]))

lh_label = temp3.in_label(MBelt_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
pt_vertices = np.unique(np.append(pt_vertices, temp.vertices[0]))

lh_label = temp3.in_label(RI_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
pt_vertices = np.unique(np.append(pt_vertices, temp.vertices[0]))

new_label = mne.Label(pt_vertices, hemi='lh')
brain3_1.add_label(new_label, borders=True, color='k')

###############################################################################
""" Frontal """
mask = np.logical_and(times >= 0.12, times <= 0.16)
lh_label = temp3.in_label(a44_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
broca_vertices = temp.vertices[0]

lh_label = temp3.in_label(IFJa_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))

lh_label = temp3.in_label(p946v_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))

new_label = mne.Label(broca_vertices, hemi='lh')
brain3_1.add_label(new_label, borders=True, color='k')
###############################################################################
mask = np.logical_and(times >= 0.18, times <= 0.22)
lh_label = temp3.in_label(PGi_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < threshold] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
STGp_vertices = temp.vertices[0]

lh_label = temp3.in_label(PGs_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < threshold] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
STGp_vertices = np.unique(np.append(STGp_vertices, temp.vertices[0]))

new_label = mne.Label(STGp_vertices, hemi='lh')
brain3_1.add_label(new_label, borders=True, color='k')
###############################################################################
#mask = np.logical_and(times >= 0.12, times <= 0.16)
#lh_label = temp3.in_label(LIPv_label_lh)
#data = np.mean(lh_label.data[:,mask],axis=1)
#lh_label.data[data < 1.72] = 0.
#
#temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
#                      subjects_dir=fs_dir, connected=False)
#temp = temp3.in_label(temp_labels)
#LIP_vertices = temp.vertices[0]
#
#lh_label = temp3.in_label(LIPd_label_lh)
#data = np.mean(lh_label.data[:,mask],axis=1)
#lh_label.data[data < 1.72] = 0.
#
#temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
#                      subjects_dir=fs_dir, connected=False)
#temp = temp3.in_label(temp_labels)
#LIP_vertices = np.unique(np.append(LIP_vertices, temp.vertices[0]))
#
#new_label = mne.Label(LIP_vertices, hemi='lh')
#brain3_1.add_label(new_label, borders=True, color='k')

#%%
""" Dot task: All subjects """
plt.figure(2)
plt.clf()

X11 = X13[ventral_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,1)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-2, 3])
plt.title('Dot task: VWFA')

plt.subplot(3,3,2)
plt.hold(True)

plt.plot(times, M1[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,1]-errM1[:,1], M1[:,1]+errM1[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,3]-errM1[:,3], M1[:,3]+errM1[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-2, 3])
plt.title('Good Readers')

plt.subplot(3,3,3)
plt.hold(True)

plt.plot(times, M2[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,1]-errM2[:,1], M1[:,1]+errM2[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,3]-errM2[:,3], M2[:,3]+errM2[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-2, 3])
plt.title('Poor Readers')

X11 = X13[pt_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,4)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Dot task: Frontal')

plt.subplot(3,3,5)
plt.hold(True)

plt.plot(times, M1[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,1]-errM1[:,1], M1[:,1]+errM1[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,3]-errM1[:,3], M1[:,3]+errM1[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Good Readers')

plt.subplot(3,3,6)
plt.hold(True)

plt.plot(times, M2[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,1]-errM2[:,1], M2[:,1]+errM2[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,3]-errM2[:,3], M2[:,3]+errM2[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Poor Readers')

X11 = X13[w_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,7)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Dot task: STG')

plt.subplot(3,3,8)
plt.hold(True)

plt.plot(times, M1[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,1]-errM1[:,1], M1[:,1]+errM1[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,3]-errM1[:,3], M1[:,3]+errM1[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Good Readers')

plt.subplot(3,3,9)
plt.hold(True)

plt.plot(times, M2[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,1]-errM2[:,1], M2[:,1]+errM2[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,3]-errM2[:,3], M2[:,3]+errM2[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Poor Readers')

#%%
"""
Correlation
"""
X11 = X13[w_vertices,:,:,:]
mask = np.logical_and(times >= 0.22, times <= 0.26)
#dot_task = np.mean(X11[:,:,:,0],axis=0)
dot_task = np.mean(X11[:,mask,:,:],axis=0)
sts_response = np.mean(dot_task[:,:,0],axis=0) - np.mean(dot_task[:,:,3],axis=0)
#sts_response = np.mean(temp[mask,:],axis=0)


#plt.figure(20)
#plt.clf()
#ax = plt.subplot()
#ax.scatter(wid_ss[all_subject], sts_response[all_subject], s=30, c='k', alpha=0.5)
#for i, txt in enumerate(all_subject):
#    ax.annotate(subs[txt], (wid_ss[txt], sts_response[txt]))
#
#np.corrcoef(sts_response[all_subject],wid_ss[all_subject])

plt.figure(20)
plt.clf()
ax = plt.subplot()
ax.scatter(twre[all_subject], sts_response[all_subject], s=30, c='k', alpha=0.5)
for i, txt in enumerate(all_subject):
    ax.annotate(subs[txt], (twre[txt], sts_response[txt]))

np.corrcoef(sts_response[all_subject],twre[all_subject])
stats.pearsonr(sts_response[all_subject],twre[all_subject])
stats.ttest_ind(sts_response[good_readers],sts_response[poor_readers])
#sns.regplot(

#%%
""" For FLUX """

plt.figure(400)
plt.clf()

cond = 5

X11 = X13[ventral_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.hold(True)

plt.plot(times, M2[:,cond],'-',linewidth=3,color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,cond]-errM2[:,cond], M2[:,cond]+errM2[:,cond], facecolor=c_table[5], alpha=0.2, edgecolor='none')

#plt.plot(times, M[:,6],'-',linewidth=3,color=c_table[3],label='Med noise')
#plt.fill_between(times, M[:,6]-errM[:,6], M[:,6]+errM[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,cond+3],'--',linewidth=3,color=c_table[3],label='High noise')
plt.fill_between(times, M2[:,cond+3]-errM2[:,cond+3], M2[:,cond+3]+errM2[:,cond+3], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-2, 3.5])
plt.xlim([-0.1,0.7])
plt.yticks([-2,-1,0,1,2,3])
#plt.title('Lexical task: VWFA')
plt.savefig('Lexical_ventral_bottomtop_poor.pdf')
#%%
for iSub in np.arange(0,len(poor_readers)):
    plt.figure(100+iSub)
    plt.clf()
    plt.subplot(1,2,1)
    plt.hold(True)
    plt.plot(times, np.mean(X11[:,:,poor_readers[iSub],0],axis=0), '-', color=c_table[5])
    plt.plot(times, np.mean(X11[:,:,poor_readers[iSub],1],axis=0), '-', color=c_table[3])
    plt.plot(times, np.mean(X11[:,:,poor_readers[iSub],3],axis=0), '-', color=c_table[1])
##    plt.plot([0.1, 0.1],[0, 8],'-',color='k')
    plt.title(subs[poor_readers[iSub]])
    plt.subplot(1,2,2)
    plt.hold(True)
    plt.plot(times, np.mean(X11[:,:,poor_readers[iSub],5],axis=0), '-', color=c_table[5])
    plt.plot(times, np.mean(X11[:,:,poor_readers[iSub],6],axis=0), '-', color=c_table[3])
    plt.plot(times, np.mean(X11[:,:,poor_readers[iSub],8],axis=0), '-', color=c_table[1])
#    plt.plot([0.1, 0.1],[0, 8],'-',color='k')
    plt.title(subs[poor_readers[iSub]])

#%%
#%%
""" Lexical task: All subjects """
plt.figure(4)
plt.clf()

X11 = X13[ventral_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,1)
plt.hold(True)

plt.plot(times, M[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,5]-errM[:,5], M[:,5]+errM[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,6]-errM[:,6], M[:,6]+errM[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,8]-errM[:,8], M[:,8]+errM[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-2, 3])
plt.title('Lexical task: VWFA')

plt.subplot(3,3,2)
plt.hold(True)

plt.plot(times, M1[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,5]-errM1[:,5], M1[:,5]+errM1[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,6]-errM1[:,6], M1[:,6]+errM1[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,8]-errM1[:,8], M1[:,8]+errM1[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-2, 3])
plt.title('Good Readers')

plt.subplot(3,3,3)
plt.hold(True)

plt.plot(times, M2[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,5]-errM2[:,5], M2[:,5]+errM2[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,6]-errM2[:,6], M2[:,6]+errM2[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,8]-errM2[:,8], M2[:,8]+errM2[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-2, 3])
plt.title('Poor Readers')

X11 = X13[pt_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,4)
plt.hold(True)

plt.plot(times, M[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,5]-errM[:,5], M[:,5]+errM[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,6]-errM[:,6], M[:,6]+errM[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,8]-errM[:,8], M[:,8]+errM[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Lexical task: Frontal')

plt.subplot(3,3,5)
plt.hold(True)

plt.plot(times, M1[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,5]-errM1[:,5], M1[:,5]+errM1[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,6]-errM1[:,6], M1[:,6]+errM1[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,8]-errM1[:,8], M1[:,8]+errM1[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Good Readers')

plt.subplot(3,3,6)
plt.hold(True)

plt.plot(times, M2[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,5]-errM2[:,5], M2[:,5]+errM2[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,6]-errM2[:,6], M2[:,6]+errM2[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,8]-errM2[:,8], M2[:,8]+errM2[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Poor Readers')

X11 = X13[w_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,7)
plt.hold(True)

plt.plot(times, M[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,5]-errM[:,5], M[:,5]+errM[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,6]-errM[:,6], M[:,6]+errM[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,8]-errM[:,8], M[:,8]+errM[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Lexical task: STS')

plt.subplot(3,3,8)
plt.hold(True)

plt.plot(times, M1[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,5]-errM1[:,5], M1[:,5]+errM1[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,6]-errM1[:,6], M1[:,6]+errM1[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,8]-errM1[:,8], M1[:,8]+errM1[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Good Readers')

plt.subplot(3,3,9)
plt.hold(True)

plt.plot(times, M2[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,5]-errM2[:,5], M2[:,5]+errM2[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,6]-errM2[:,6], M2[:,6]+errM2[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,8]-errM2[:,8], M2[:,8]+errM2[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Poor Readers')

#%%
""" Young vs. old """

plt.figure(3)
plt.clf()

X11 = X13[ventral_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,old_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,young_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,old_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,young_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,1)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Dot task: VWFA')

plt.subplot(3,3,2)
plt.hold(True)

plt.plot(times, M1[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,1]-errM1[:,1], M1[:,1]+errM1[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,3]-errM1[:,3], M1[:,3]+errM1[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Good Readers')

plt.subplot(3,3,3)
plt.hold(True)

plt.plot(times, M2[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,1]-errM2[:,1], M1[:,1]+errM2[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,3]-errM2[:,3], M2[:,3]+errM2[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Poor Readers')

X11 = X13[pt_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,old_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,young_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,old_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,young_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,4)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Dot task: Frontal')

plt.subplot(3,3,5)
plt.hold(True)

plt.plot(times, M1[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,1]-errM1[:,1], M1[:,1]+errM1[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,3]-errM1[:,3], M1[:,3]+errM1[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Good Readers')

plt.subplot(3,3,6)
plt.hold(True)

plt.plot(times, M2[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,1]-errM2[:,1], M2[:,1]+errM2[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,3]-errM2[:,3], M2[:,3]+errM2[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Poor Readers')

X11 = X13[w_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,old_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,young_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,old_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,young_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,7)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Dot task: STG')

plt.subplot(3,3,8)
plt.hold(True)

plt.plot(times, M1[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,1]-errM1[:,1], M1[:,1]+errM1[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,3]-errM1[:,3], M1[:,3]+errM1[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Good Readers')

plt.subplot(3,3,9)
plt.hold(True)

plt.plot(times, M2[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,1]-errM2[:,1], M2[:,1]+errM2[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,3]-errM2[:,3], M2[:,3]+errM2[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Poor Readers')

"""
Correlation
"""
mask = np.logical_and(times >= 0.22, times <= 0.26)
temp = np.mean(X11[:,:,:,0],axis=0)
sts_response = np.mean(temp[mask,:],axis=0)

plt.figure(30)
plt.clf()
plt.hold(True)

ax = plt.subplot()
ax.scatter(twre[old_readers], sts_response[old_readers], s=30, c='r', alpha=1)
for i, txt in enumerate(old_readers):
    ax.annotate(age[txt], (twre[txt], sts_response[txt]))

ax.scatter(twre[young_readers], sts_response[young_readers], s=30, c='b', alpha=1)
for i, txt in enumerate(young_readers):
    ax.annotate(age[txt], (twre[txt], sts_response[txt]))

np.corrcoef(sts_response[young_readers],twre[young_readers])
np.pear
np.corrcoef(sts_response[old_readers],twre[old_readers])
np.corrcoef(sts_response[all_subject],twre[all_subject])

#%% 
""" Dot task: V1 """
axis_font = {'fontname':'Arial', 'size':'16'}
font_prop = font_manager.FontProperties(size=12)

#ax = plt.subplot() # Defines ax variable by creating an empty plot
## Set the tick labels font
#for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#    label.set_fontname('Arial')
#    label.set_fontsize(13)

#plt.hold(True)
#
#plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
#plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')
#
#plt.plot(times, M[:,1],'-',color=c_table[3],label='Med noise')
#plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')
#
#plt.plot(times, M[:,3],'-',color=c_table[1],label='High noise')
#plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')
#
#plt.grid(b=True)
#plt.ylim([-1, 2])
#plt.legend(loc='upper right', prop=font_prop)
#plt.xlabel('Time after stimulus onset (s)', **axis_font)
#plt.ylabel('dSPM amplitude', **axis_font)
#plt.show()

plt.figure(1)
plt.clf()

X11 = X13[v1_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))

plt.subplot(2,3,1)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 2])
plt.legend()
plt.title('Dot task: V1')

plt.subplot(2,3,4)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='High contrast')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,2],'-',color=c_table[7],label='Low contrast')
plt.fill_between(times, M[:,2]-errM[:,2], M[:,2]+errM[:,2], facecolor=c_table[7], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 2])
plt.legend()

M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(2, 3, 2)
plt.hold(True)

plt.plot(times, M1[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,1]-errM1[:,1], M1[:,1]+errM1[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,3]-errM1[:,3], M1[:,3]+errM1[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 2])
plt.title('Dot task (GR): V1')

plt.subplot(2,3,5)
plt.hold(True)

plt.plot(times, M1[:,0],'-',color=c_table[5],label='High contrast')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,2],'-',color=c_table[7],label='Low contrast')
plt.fill_between(times, M1[:,2]-errM1[:,2], M1[:,2]+errM1[:,2], facecolor=c_table[7], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 2])

plt.subplot(2, 3, 3)
plt.hold(True)

plt.plot(times, M2[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,1]-errM2[:,1], M2[:,1]+errM2[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,3]-errM2[:,3], M2[:,3]+errM2[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 2])
plt.title('Dot task (PR): V1')

plt.subplot(2,3,6)
plt.hold(True)

plt.plot(times, M2[:,0],'-',color=c_table[5],label='High contrast')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,2],'-',color=c_table[7],label='Low contrast')
plt.fill_between(times, M2[:,2]-errM2[:,2], M2[:,2]+errM2[:,2], facecolor=c_table[7], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 2])

""" Plot individual V1 responses """
#for iSub in np.arange(0,len(poor_readers)):
#    plt.figure(100+iSub)
#    plt.clf()
#    plt.subplot(1,2,1)
#    plt.hold(True)
#    plt.plot(times, np.mean(X1[v1_vertices,:,poor_readers[iSub],0],axis=0), '--', color=c_table[5])
#    plt.plot(times, np.mean(X1[v1_vertices,:,poor_readers[iSub],1],axis=0), '--', color=c_table[3])
#    plt.plot(times, np.mean(X1[v1_vertices,:,poor_readers[iSub],3],axis=0), '--', color=c_table[1])
#    plt.plot([0.1, 0.1],[0, 8],'-',color='k')
#    plt.title(subs[poor_readers[iSub]])
#    plt.subplot(1,2,2)
#    plt.hold(True)
#    plt.plot(times, np.mean(X1[v1_vertices,:,poor_readers[iSub],5],axis=0), '-', color=c_table[5])
#    plt.plot(times, np.mean(X1[v1_vertices,:,poor_readers[iSub],6],axis=0), '-', color=c_table[3])
#    plt.plot(times, np.mean(X1[v1_vertices,:,poor_readers[iSub],8],axis=0), '-', color=c_table[1])
#    plt.plot([0.1, 0.1],[0, 8],'-',color='k')
#    plt.title(subs[poor_readers[iSub]])
#%% 
""" Low contrast vs. high contrast """
plt.figure(3)
plt.clf()

X11 = X13[ventral_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,1)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,2],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,2]-errM[:,2], M[:,2]+errM[:,2], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Dot task: VWFA')

plt.subplot(3,3,2)
plt.hold(True)

plt.plot(times, M1[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,2],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,2]-errM1[:,2], M1[:,2]+errM1[:,2], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Good Readers')

plt.subplot(3,3,3)
plt.hold(True)

plt.plot(times, M2[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,2],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,2]-errM2[:,2], M2[:,2]+errM2[:,2], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Poor Readers')

X11 = X13[pt_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,4)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,2],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,2]-errM[:,2], M[:,2]+errM[:,2], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Dot task: Frontal')

plt.subplot(3,3,5)
plt.hold(True)

plt.plot(times, M1[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,2],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,2]-errM1[:,2], M1[:,2]+errM1[:,2], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Good Readers')

plt.subplot(3,3,6)
plt.hold(True)

plt.plot(times, M2[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,2],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,2]-errM2[:,2], M2[:,2]+errM2[:,2], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Poor Readers')

X11 = X13[w_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,7)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,2],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,2]-errM[:,2], M[:,2]+errM[:,2], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Dot task: STG')

plt.subplot(3,3,8)
plt.hold(True)

plt.plot(times, M1[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,2],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,2]-errM1[:,2], M1[:,2]+errM1[:,2], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Good Readers')

plt.subplot(3,3,9)
plt.hold(True)

plt.plot(times, M2[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,2],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,2]-errM2[:,2], M2[:,2]+errM2[:,2], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Poor Readers')

#%%
""" Task effects in V1"""
plt.figure(5)
plt.clf()

X11 = X13[v1_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,1)
plt.hold(True)
plt.plot(times, M[:,0],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,5],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M[:,5]-errM[:,5], M[:,5]+errM[:,5], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2])
plt.legend()
plt.title('V1: Low Noise - all')

plt.subplot(3,3,2)
plt.hold(True)
plt.plot(times, M[:,1],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,6],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M[:,6]-errM[:,6], M[:,6]+errM[:,6], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2])
plt.title('V1: Med Noise - all')

plt.subplot(3,3,3)
plt.hold(True)
plt.plot(times, M[:,3],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,8],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M[:,8]-errM[:,8], M[:,8]+errM[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2])
plt.title('V1:High Noise - all')

plt.subplot(3,3,4)
plt.hold(True)
plt.plot(times, M1[:,0],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,5],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M1[:,5]-errM1[:,5], M1[:,5]+errM1[:,5], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2])
plt.title('Low Noise - good')

plt.subplot(3,3,5)
plt.hold(True)
plt.plot(times, M1[:,1],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M1[:,1]-errM1[:,1], M1[:,1]+errM1[:,1], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,6],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M1[:,6]-errM1[:,6], M1[:,6]+errM1[:,6], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2])
plt.title('Med Noise - good')

plt.subplot(3,3,6)
plt.hold(True)
plt.plot(times, M1[:,3],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M1[:,3]-errM1[:,3], M1[:,3]+errM1[:,3], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,8],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M1[:,8]-errM1[:,8], M1[:,8]+errM1[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2])
plt.title('High Noise - good')

plt.subplot(3,3,7)
plt.hold(True)
plt.plot(times, M2[:,0],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,5],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M2[:,5]-errM2[:,5], M2[:,5]+errM2[:,5], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2])
plt.title('Low Noise - poor')

plt.subplot(3,3,8)
plt.hold(True)
plt.plot(times, M2[:,1],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M2[:,1]-errM2[:,1], M2[:,1]+errM2[:,1], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,6],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M2[:,6]-errM2[:,6], M2[:,6]+errM2[:,6], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2])
plt.title('Med Noise - poor')

plt.subplot(3,3,9)
plt.hold(True)
plt.plot(times, M2[:,3],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M2[:,3]-errM2[:,3], M2[:,3]+errM2[:,3], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,8],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M2[:,8]-errM2[:,8], M2[:,8]+errM2[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2])
plt.title('High Noise - poor')

#%%
""" Task effects in VWFA"""
plt.figure(6)
plt.clf()

X11 = X13[ventral_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,1)
plt.hold(True)
plt.plot(times, M[:,0],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,5],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M[:,5]-errM[:,5], M[:,5]+errM[:,5], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2.5])
#plt.legend()
plt.title('VWFA: Low Noise - all')

plt.subplot(3,3,2)
plt.hold(True)
plt.plot(times, M[:,1],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,6],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M[:,6]-errM[:,6], M[:,6]+errM[:,6], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2.5])
plt.title('VWFA: Med Noise - all')

plt.subplot(3,3,3)
plt.hold(True)
plt.plot(times, M[:,3],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,8],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M[:,8]-errM[:,8], M[:,8]+errM[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2.5])
plt.title('VWFA:High Noise - all')

plt.subplot(3,3,4)
plt.hold(True)
plt.plot(times, M1[:,0],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,5],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M1[:,5]-errM1[:,5], M1[:,5]+errM1[:,5], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2.5])
plt.title('Low Noise - good')

plt.subplot(3,3,5)
plt.hold(True)
plt.plot(times, M1[:,1],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M1[:,1]-errM1[:,1], M1[:,1]+errM1[:,1], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,6],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M1[:,6]-errM1[:,6], M1[:,6]+errM1[:,6], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2.5])
plt.title('Med Noise - good')

plt.subplot(3,3,6)
plt.hold(True)
plt.plot(times, M1[:,3],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M1[:,3]-errM1[:,3], M1[:,3]+errM1[:,3], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,8],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M1[:,8]-errM1[:,8], M1[:,8]+errM1[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2.5])
plt.title('High Noise - good')

plt.subplot(3,3,7)
plt.hold(True)
plt.plot(times, M2[:,0],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,5],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M2[:,5]-errM2[:,5], M2[:,5]+errM2[:,5], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2.5])
plt.title('Low Noise - poor')

plt.subplot(3,3,8)
plt.hold(True)
plt.plot(times, M2[:,1],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M2[:,1]-errM2[:,1], M2[:,1]+errM2[:,1], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,6],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M2[:,6]-errM2[:,6], M2[:,6]+errM2[:,6], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2.5])
plt.title('Med Noise - poor')

plt.subplot(3,3,9)
plt.hold(True)
plt.plot(times, M2[:,3],'--',color=c_table[5],label='Dot')
plt.fill_between(times, M2[:,3]-errM2[:,3], M2[:,3]+errM2[:,3], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,8],'-',color=c_table[1],label='Lexical')
plt.fill_between(times, M2[:,8]-errM2[:,8], M2[:,8]+errM2[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.ylim([-1, 2.5])
plt.title('High Noise - poor')

#%%
""" Good vs. Poor in VWFA"""
plt.figure(7)
plt.clf()

X11 = X13[ventral_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(2, 3, 1)
plt.hold(True)

plt.plot(times, M2[:,0],'-',color=c_table[1],label='Poor readers')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,0],'-',color=c_table[5],label='Good readers')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.legend()
plt.title('VWFA: Low noise, Dot task')

plt.subplot(2, 3, 2)
plt.hold(True)

plt.plot(times, M2[:,1],'-',color=c_table[1],label='Poor readers')
plt.fill_between(times, M2[:,1]-errM2[:,1], M2[:,1]+errM2[:,1], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,1],'-',color=c_table[5],label='Good readers')
plt.fill_between(times, M1[:,1]-errM1[:,1], M1[:,1]+errM1[:,1], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.legend()
plt.title('VWFA: Med noise, Dot task')

plt.subplot(2, 3, 3)
plt.hold(True)

plt.plot(times, M2[:,3],'-',color=c_table[1],label='Poor readers')
plt.fill_between(times, M2[:,3]-errM2[:,3], M2[:,3]+errM2[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,3],'-',color=c_table[5],label='Good readers')
plt.fill_between(times, M1[:,3]-errM1[:,3], M1[:,3]+errM1[:,3], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.legend()
plt.title('VWFA: High noise, Dot task')

plt.subplot(2, 3, 4)
plt.hold(True)

plt.plot(times, M2[:,5],'-',color=c_table[1],label='Poor readers')
plt.fill_between(times, M2[:,5]-errM2[:,5], M2[:,5]+errM2[:,5], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,5],'-',color=c_table[5],label='Good readers')
plt.fill_between(times, M1[:,5]-errM1[:,5], M1[:,5]+errM1[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.legend()
plt.title('VWFA: Low noise, Lexical task')

plt.subplot(2, 3, 5)
plt.hold(True)

plt.plot(times, M2[:,6],'-',color=c_table[1],label='Poor readers')
plt.fill_between(times, M2[:,6]-errM2[:,6], M2[:,6]+errM2[:,6], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,6],'-',color=c_table[5],label='Good readers')
plt.fill_between(times, M1[:,6]-errM1[:,6], M1[:,6]+errM1[:,6], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.legend()
plt.title('VWFA: Med noise, Lexical task')

plt.subplot(2, 3, 6)
plt.hold(True)

plt.plot(times, M2[:,8],'-',color=c_table[1],label='Poor readers')
plt.fill_between(times, M2[:,8]-errM2[:,8], M2[:,8]+errM2[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,3],'-',color=c_table[5],label='Good readers')
plt.fill_between(times, M1[:,8]-errM1[:,8], M1[:,8]+errM1[:,8], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.legend()
plt.title('VWFA: High noise, Lexical task')

#%%
""" Good vs. Poor in STS"""
plt.figure(8)
plt.clf()

X11 = X13[w_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(2, 3, 1)
plt.hold(True)

plt.plot(times, M2[:,0],'-',color=c_table[1],label='Poor readers')
plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,0],'-',color=c_table[5],label='Good readers')
plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.legend()
plt.title('STS: Low noise, Dot task')

plt.subplot(2, 3, 2)
plt.hold(True)

plt.plot(times, M2[:,1],'-',color=c_table[1],label='Poor readers')
plt.fill_between(times, M2[:,1]-errM2[:,1], M2[:,1]+errM2[:,1], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,1],'-',color=c_table[5],label='Good readers')
plt.fill_between(times, M1[:,1]-errM1[:,1], M1[:,1]+errM1[:,1], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.legend()
plt.title('STS: Med noise, Dot task')

plt.subplot(2, 3, 3)
plt.hold(True)

plt.plot(times, M2[:,3],'-',color=c_table[1],label='Poor readers')
plt.fill_between(times, M2[:,3]-errM2[:,3], M2[:,3]+errM2[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,3],'-',color=c_table[5],label='Good readers')
plt.fill_between(times, M1[:,3]-errM1[:,3], M1[:,3]+errM1[:,3], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.legend()
plt.title('STS: High noise, Dot task')

plt.subplot(2, 3, 4)
plt.hold(True)

plt.plot(times, M2[:,5],'-',color=c_table[1],label='Poor readers')
plt.fill_between(times, M2[:,5]-errM2[:,5], M2[:,5]+errM2[:,5], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,5],'-',color=c_table[5],label='Good readers')
plt.fill_between(times, M1[:,5]-errM1[:,5], M1[:,5]+errM1[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.legend()
plt.title('STS: Low noise, Lexical task')

plt.subplot(2, 3, 5)
plt.hold(True)

plt.plot(times, M2[:,6],'-',color=c_table[1],label='Poor readers')
plt.fill_between(times, M2[:,6]-errM2[:,6], M2[:,6]+errM2[:,6], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,6],'-',color=c_table[5],label='Good readers')
plt.fill_between(times, M1[:,6]-errM1[:,6], M1[:,6]+errM1[:,6], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.legend()
plt.title('STS: Med noise, Lexical task')

plt.subplot(2, 3, 6)
plt.hold(True)

plt.plot(times, M2[:,8],'-',color=c_table[1],label='Poor readers')
plt.fill_between(times, M2[:,8]-errM2[:,8], M2[:,8]+errM2[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,3],'-',color=c_table[5],label='Good readers')
plt.fill_between(times, M1[:,8]-errM1[:,8], M1[:,8]+errM1[:,8], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.legend()
plt.title('STS: High noise, Lexical task')

#%%
""" Task effects """
plt.figure(7)
plt.clf()

X11 = X1[ventral_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 1)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M1[:,[0]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[0]],axis=1)-yerr, np.mean(M1[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M1[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[5]],axis=1)-yerr, np.mean(M1[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Ventral')

plt.subplot(3, 3, 2)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M1[:,[1]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[1]],axis=1)-yerr, np.mean(M1[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M1[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[6]],axis=1)-yerr, np.mean(M1[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Ventral')

plt.subplot(3, 3, 3)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M1[:,[3]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[3]],axis=1)-yerr, np.mean(M1[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M1[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[8]],axis=1)-yerr, np.mean(M1[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Ventral')

X11 = X1[broca_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 4)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M1[:,[0]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[0]],axis=1)-yerr, np.mean(M1[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M1[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[5]],axis=1)-yerr, np.mean(M1[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Frontal')

plt.subplot(3, 3, 5)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M1[:,[1]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[1]],axis=1)-yerr, np.mean(M1[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M1[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[6]],axis=1)-yerr, np.mean(M1[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Frontal')

plt.subplot(3, 3, 6)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M1[:,[3]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[3]],axis=1)-yerr, np.mean(M1[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M1[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[8]],axis=1)-yerr, np.mean(M1[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Frontal')

X11 = X1[w_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 7)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M1[:,[0]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[0]],axis=1)-yerr, np.mean(M1[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M1[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[5]],axis=1)-yerr, np.mean(M1[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Temporal')

plt.subplot(3, 3, 8)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M1[:,[1]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[1]],axis=1)-yerr, np.mean(M1[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M1[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[6]],axis=1)-yerr, np.mean(M1[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Temporal')

plt.subplot(3, 3, 9)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M1[:,[3]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[3]],axis=1)-yerr, np.mean(M1[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M1[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[8]],axis=1)-yerr, np.mean(M1[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Temporal')

#%%
""" Task effects """
plt.figure(8)
plt.clf()

X11 = X1[ventral_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 1)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M2[:,[0]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[0]],axis=1)-yerr, np.mean(M2[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M2[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[5]],axis=1)-yerr, np.mean(M2[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Ventral')

plt.subplot(3, 3, 2)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M2[:,[1]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[1]],axis=1)-yerr, np.mean(M2[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M2[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[6]],axis=1)-yerr, np.mean(M2[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Ventral')

plt.subplot(3, 3, 3)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M2[:,[3]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[3]],axis=1)-yerr, np.mean(M2[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M2[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[8]],axis=1)-yerr, np.mean(M2[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Ventral')

X11 = X1[broca_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 4)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M2[:,[0]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[0]],axis=1)-yerr, np.mean(M2[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M2[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[5]],axis=1)-yerr, np.mean(M2[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Frontal')

plt.subplot(3, 3, 5)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M2[:,[1]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[1]],axis=1)-yerr, np.mean(M2[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M2[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[6]],axis=1)-yerr, np.mean(M2[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Frontal')

plt.subplot(3, 3, 6)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M2[:,[3]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[3]],axis=1)-yerr, np.mean(M2[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M2[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[8]],axis=1)-yerr, np.mean(M2[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Frontal')

X11 = X1[w_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 7)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M2[:,[0]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[0]],axis=1)-yerr, np.mean(M2[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M2[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[5]],axis=1)-yerr, np.mean(M2[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Temporal')

plt.subplot(3, 3, 8)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M2[:,[1]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[1]],axis=1)-yerr, np.mean(M2[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M2[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[6]],axis=1)-yerr, np.mean(M2[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Temporal')

plt.subplot(3, 3, 9)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M2[:,[3]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[3]],axis=1)-yerr, np.mean(M2[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M2[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[8]],axis=1)-yerr, np.mean(M2[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Temporal')

#%%
plt.subplot(3,3,2)
plt.hold(True)

plt.plot(times, M1[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,5]-errM1[:,5], M1[:,5]+errM1[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,6]-errM1[:,6], M1[:,6]+errM1[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,8]-errM1[:,8], M1[:,8]+errM1[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Good Readers')

plt.subplot(3,3,3)
plt.hold(True)

plt.plot(times, M2[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,5]-errM2[:,5], M2[:,5]+errM2[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,6]-errM2[:,6], M2[:,6]+errM2[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,8]-errM2[:,8], M2[:,8]+errM2[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Poor Readers')

X11 = X13[broca_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,4)
plt.hold(True)

plt.plot(times, M[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,5]-errM[:,5], M[:,5]+errM[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,6]-errM[:,6], M[:,6]+errM[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,8]-errM[:,8], M[:,8]+errM[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Lexical task: Frontal')

plt.subplot(3,3,5)
plt.hold(True)

plt.plot(times, M1[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,5]-errM1[:,5], M1[:,5]+errM1[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,6]-errM1[:,6], M1[:,6]+errM1[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,8]-errM1[:,8], M1[:,8]+errM1[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Good Readers')

plt.subplot(3,3,6)
plt.hold(True)

plt.plot(times, M2[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,5]-errM2[:,5], M2[:,5]+errM2[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,6]-errM2[:,6], M2[:,6]+errM2[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,8]-errM2[:,8], M2[:,8]+errM2[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Poor Readers')

X11 = X13[w_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))
errM1 = np.std(np.mean(X11[:,:,good_readers,:],axis=0),axis=1) / np.sqrt(len(good_readers))
errM2 = np.std(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1) / np.sqrt(len(poor_readers))

plt.subplot(3,3,7)
plt.hold(True)

plt.plot(times, M[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,5]-errM[:,5], M[:,5]+errM[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,6]-errM[:,6], M[:,6]+errM[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,8]-errM[:,8], M[:,8]+errM[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Lexical task: STS')

plt.subplot(3,3,8)
plt.hold(True)

plt.plot(times, M1[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,5]-errM1[:,5], M1[:,5]+errM1[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,6]-errM1[:,6], M1[:,6]+errM1[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,8]-errM1[:,8], M1[:,8]+errM1[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Good Readers')

plt.subplot(3,3,9)
plt.hold(True)

plt.plot(times, M2[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,5]-errM2[:,5], M2[:,5]+errM2[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,6]-errM2[:,6], M2[:,6]+errM2[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,8]-errM2[:,8], M2[:,8]+errM2[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Poor Readers')
