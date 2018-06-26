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
import csv
os.chdir('/home/sjjoo/git/BrainTools/projects/NLR_MEG')
from plotit import plotit

set_config('MNE_MEMMAP_MIN_SIZE', '1M')
set_config('MNE_CACHE_DIR', '.tmp')

mne.set_config('MNE_USE_CUDA', 'true')

this_env = copy.copy(os.environ)
fs_dir = '/mnt/diskArray/projects/avg_fsurfer'
this_env['SUBJECTS_DIR'] = fs_dir

raw_dir = '/mnt/scratch/NLR_MEG3'

os.chdir(raw_dir)

subs = ['NLR_102_RS','NLR_110_HH','NLR_145_AC','NLR_150_MG',
        'NLR_152_TC','NLR_160_EK','NLR_161_AK','NLR_162_EF','NLR_163_LF',
        'NLR_164_SF','NLR_170_GM','NLR_172_TH','NLR_174_HS','NLR_179_GM',
        'NLR_180_ZD','NLR_201_GS',
        'NLR_204_AM','NLR_205_AC','NLR_207_AH','NLR_210_SB','NLR_211_LB',
        'NLR_GB310','NLR_KB218','NLR_GB267','NLR_JB420', 'NLR_HB275','NLR_GB355'] 

session2 = ['102_rs160815','110_hh160809',
       '145_ac160823','150_mg160825',
       '152_tc160623','160_ek160915','161_ak160916','162_ef160829','163_lf160920',
       '164_sf160920','170_gm160822','172_th160825','174_hs160829','179_gm160913',
       '180_zd160826','201_gs150925',
       '204_am151120','205_ac160202','207_ah160809','210_sb160822','211_lb160823',
       'nlr_gb310170829','nlr_kb218170829','nlr_gb267170911','nlr_jb420170828',
       'nlr_hb275170828','nlr_gb355170907']


twre_index1 = [87,66,84,92,
               86,63,81,53,60,
               55,71,63,68,67,
               64,79,
               59,84,91,76,57,
               67,77,80,53,72,85]
twre_index2 = [90,76,94,115,
               85,75,82,64,75,
               63,83,77,84,75,
               68,79,
               62,90,105,75,71,
               69,83,76,62,73,94]
twre_index1 = np.array(twre_index1)
twre_index2 = np.array(twre_index2)

#age1 = [125.6885, 132.9501, 122.0434, 138.4349, 97.6347, 138.1420, 108.2457, 98.0631, 105.8147, 89.9132,
#       87.6465, 131.8660, 123.7174, 95.959, 112.416, 133.8042, 152.4639, 103.4823, 89.8475, 138.4020,
#       93.8568, 117.0814, 123.6202, 122.9304, 109.1656, 90.6058,
#       111.9593,86.0381,147.2063,95.8699,148.0802,122.5896,88.7162,123.0495,110.6645,105.3069,88.9143,95.2879,106.2852,
#       122.2915,114.4389,136.1496,128.6246,137.9216,122.7528]
#age1 = np.divide(age1, 12)

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

fname_data = op.join(raw_dir, 'session2_data_loose_depth8_normal.npy')

#%%
""" Some checks """
n = 38
os.chdir(os.path.join(raw_dir,session2[n]))
os.chdir('inverse')
fn = 'All_40-sss_eq_'+session2[n]+'-ave.fif'
evoked = mne.read_evokeds(fn, condition=0, 
                baseline=(None,0), kind='average', proj=True)
    
info = evoked.info 
    
if os.path.isdir('../forward'):
    os.chdir('../forward')

trans = session2[n] + '-trans.fif'
#    Take a look at the sensors
mne.viz.plot_trans(info, trans, subject=subs[n], dig=True,
                   meg_sensors=True, subjects_dir=fs_dir)
    
os.chdir(os.path.join(raw_dir,session2[n]))
os.chdir('epochs')
epo = mne.read_epochs('All_40-sss_'+session2[n]+'-epo.fif',proj='delayed')
epo.average().plot(proj='interactive')

#%%
"""
Here we load the data
"""            
# Session 1
os.chdir(raw_dir)
X13 = np.load(fname_data)
orig_times = np.load('session2_times.npy')
tstep = np.load('session2_tstep.npy')
n_epochs = np.load('session2_n_averages.npy')
tmin = -0.1

#m1 = np.logical_and(np.transpose(twre_index) >= 85, np.transpose(age) <= 13)
#m2 = np.logical_and(np.transpose(twre_index) < 85, np.transpose(age) <= 13)
##m4 = np.logical_and(np.transpose(twre_index) >= 80, np.transpose(twre_index) < 90)
#m3 = np.mean(n_epochs,axis=1) < 40
#m1[np.where(m3)] = False
#m2[np.where(m3)] = False
#
#good_readers = np.where(m1)[0]
#poor_readers = np.where(m2)[0]
##middle_readers = np.where(m4)[0]
#
#a1 = np.transpose(age) > 9
#a2 = np.logical_not(a1)
#
#old_readers = np.where(a1)[0]
#young_readers = np.where(a2)[0]
#
#all_subject = []
#all_subject.extend(good_readers)
#all_subject.extend(poor_readers)
##all_subject.extend(middle_readers)
#all_subject.sort()

fs_vertices = [np.arange(10242)] * 2

#%% 
""" Downsample the data """
sample = np.arange(0,len(orig_times),2)
times = orig_times[sample]
tstep = 2*tstep
X11 = X13[:,sample,:,:]
del X13
X11 = np.abs(X11)

#%%
""" Read HCP labels """
labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', surf_name='white', subjects_dir=fs_dir) #regexp=aparc_label_name
#aparc_label_name = 'PHT_ROI'#'_IP'#'IFSp_ROI'#'STSvp_ROI'#'STSdp_ROI'#'PH_ROI'#'TE2p_ROI' #'SFL_ROI' #'IFSp_ROI' #'TE2p_ROI' #'inferiortemporal' #'pericalcarine'
anat_label = mne.read_labels_from_annot('fsaverage', parc='aparc',surf_name='white',
       subjects_dir=fs_dir) #, regexp=aparc_label_name)
#%%
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

IFSa_label_lh = [label for label in labels if label.name == 'L_IFSa_ROI-lh'][0]
IFSa_label_rh = [label for label in labels if label.name == 'R_IFSa_ROI-rh'][0]

IFJp_label_lh = [label for label in labels if label.name == 'L_IFJp_ROI-lh'][0]
IFJp_label_rh = [label for label in labels if label.name == 'R_IFJp_ROI-rh'][0]

IFJa_label_lh = [label for label in labels if label.name == 'L_IFJa_ROI-lh'][0]
IFJa_label_rh = [label for label in labels if label.name == 'R_IFJa_ROI-rh'][0]

a45_label_lh = [label for label in labels if label.name == 'L_45_ROI-lh'][0]
a45_label_rh = [label for label in labels if label.name == 'R_45_ROI-rh'][0]

a44_label_lh = [label for label in labels if label.name == 'L_44_ROI-lh'][0]
a44_label_rh = [label for label in labels if label.name == 'R_44_ROI-rh'][0]

a43_label_lh = [label for label in labels if label.name == 'L_43_ROI-lh'][0]
a43_label_rh = [label for label in labels if label.name == 'R_43_ROI-rh'][0]

a9_46v_lh = [label for label in labels if label.name == 'L_a9-46v_ROI-lh'][0]
a9_46v_rh = [label for label in labels if label.name == 'R_a9-46v_ROI-rh'][0]

PGi_label_lh = [label for label in labels if label.name == 'L_PGi_ROI-lh'][0]
PGi_label_rh = [label for label in labels if label.name == 'R_PGi_ROI-rh'][0]

PGs_label_lh = [label for label in labels if label.name == 'L_PGs_ROI-lh'][0]
PGs_label_rh = [label for label in labels if label.name == 'R_PGs_ROI-rh'][0]

STSvp_label_lh = [label for label in labels if label.name == 'L_STSvp_ROI-lh'][0]
STSvp_label_rh = [label for label in labels if label.name == 'R_STSvp_ROI-rh'][0]

STSdp_label_lh = [label for label in labels if label.name == 'L_STSdp_ROI-lh'][0]
STSdp_label_rh = [label for label in labels if label.name == 'R_STSdp_ROI-rh'][0]

STSva_label_lh = [label for label in labels if label.name == 'L_STSva_ROI-lh'][0]
STSva_label_rh = [label for label in labels if label.name == 'R_STSva_ROI-rh'][0]

STSda_label_lh = [label for label in labels if label.name == 'L_STSda_ROI-lh'][0]
STSda_label_rh = [label for label in labels if label.name == 'R_STSda_ROI-rh'][0]

TPOJ1_label_lh = [label for label in labels if label.name == 'L_TPOJ1_ROI-lh'][0]
TPOJ1_label_rh = [label for label in labels if label.name == 'R_TPOJ1_ROI-rh'][0]

TPOJ2_label_lh = [label for label in labels if label.name == 'L_TPOJ2_ROI-lh'][0]
TPOJ2_label_rh = [label for label in labels if label.name == 'R_TPOJ2_ROI-rh'][0]

V1_label_lh = [label for label in labels if label.name == 'L_V1_ROI-lh'][0]
V1_label_rh = [label for label in labels if label.name == 'R_V1_ROI-rh'][0]

V4_label_lh = [label for label in labels if label.name == 'L_V4_ROI-lh'][0]
V4_label_rh = [label for label in labels if label.name == 'R_V4_ROI-rh'][0]

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

A4_label_lh = [label for label in labels if label.name == 'L_A4_ROI-lh'][0]
A4_label_rh = [label for label in labels if label.name == 'R_A4_ROI-rh'][0]

PFcm_label_lh = [label for label in labels if label.name == 'L_PFcm_ROI-lh'][0]
PFcm_label_rh = [label for label in labels if label.name == 'R_PFcm_ROI-rh'][0]

PFm_label_lh = [label for label in labels if label.name == 'L_PFm_ROI-lh'][0]
PFm_label_rh = [label for label in labels if label.name == 'R_PFm_ROI-rh'][0]

_4_label_lh = [label for label in labels if label.name == 'L_4_ROI-lh'][0]
_4_label_rh = [label for label in labels if label.name == 'R_4_ROI-rh'][0]

_1_label_lh = [label for label in labels if label.name == 'L_1_ROI-lh'][0]
_1_label_rh = [label for label in labels if label.name == 'R_1_ROI-rh'][0]

_2_label_lh = [label for label in labels if label.name == 'L_2_ROI-lh'][0]
_2_label_rh = [label for label in labels if label.name == 'R_2_ROI-rh'][0]

_3a_label_lh = [label for label in labels if label.name == 'L_3a_ROI-lh'][0]
_3a_label_rh = [label for label in labels if label.name == 'R_3a_ROI-rh'][0]

_3b_label_lh = [label for label in labels if label.name == 'L_3b_ROI-lh'][0]
_3b_label_rh = [label for label in labels if label.name == 'R_3b_ROI-rh'][0]

_43_label_lh = [label for label in labels if label.name == 'L_43_ROI-lh'][0]
_43_label_rh = [label for label in labels if label.name == 'R_43_ROI-rh'][0]

_6r_label_lh = [label for label in labels if label.name == 'L_6r_ROI-lh'][0]
_6r_label_rh = [label for label in labels if label.name == 'R_6r_ROI-rh'][0]

OP1_label_lh = [label for label in labels if label.name == 'L_OP1_ROI-lh'][0]
OP1_label_rh = [label for label in labels if label.name == 'R_OP1_ROI-rh'][0]

OP23_label_lh = [label for label in labels if label.name == 'L_OP2-3_ROI-lh'][0]
OP23_label_rh = [label for label in labels if label.name == 'R_OP2-3_ROI-rh'][0]

OP4_label_lh = [label for label in labels if label.name == 'L_OP4_ROI-lh'][0]
OP4_label_rh = [label for label in labels if label.name == 'R_OP4_ROI-rh'][0]

PFop_label_lh = [label for label in labels if label.name == 'L_PFop_ROI-lh'][0]
PFop_label_rh = [label for label in labels if label.name == 'R_PFop_ROI-rh'][0]

A5_label_lh = [label for label in labels if label.name == 'L_A5_ROI-lh'][0]
A5_label_rh = [label for label in labels if label.name == 'R_A5_ROI-rh'][0]

STV_label_lh = [label for label in labels if label.name == 'L_STV_ROI-lh'][0]
STV_label_rh = [label for label in labels if label.name == 'R_STV_ROI-rh'][0]

RI_label_lh = [label for label in labels if label.name == 'L_RI_ROI-lh'][0]
RI_label_rh = [label for label in labels if label.name == 'R_RI_ROI-rh'][0]

PF_label_lh = [label for label in labels if label.name == 'L_PF_ROI-lh'][0]
PF_label_rh = [label for label in labels if label.name == 'R_PF_ROI-rh'][0]

PFt_label_lh = [label for label in labels if label.name == 'L_PFt_ROI-lh'][0]
PFt_label_rh = [label for label in labels if label.name == 'R_PFt_ROI-rh'][0]

p47r_label_lh = [label for label in labels if label.name == 'L_p47r_ROI-lh'][0]
p47r_label_rh = [label for label in labels if label.name == 'R_p47r_ROI-rh'][0]

FOP5_label_lh = [label for label in labels if label.name == 'L_FOP5_ROI-lh'][0]
FOP5_label_rh = [label for label in labels if label.name == 'R_FOP5_ROI-rh'][0]

FOP4_label_lh = [label for label in labels if label.name == 'L_FOP4_ROI-lh'][0]
FOP4_label_rh = [label for label in labels if label.name == 'R_FOP4_ROI-rh'][0]

FOP3_label_lh = [label for label in labels if label.name == 'L_FOP3_ROI-lh'][0]
FOP3_label_rh = [label for label in labels if label.name == 'R_FOP3_ROI-rh'][0]

FOP2_label_lh = [label for label in labels if label.name == 'L_FOP2_ROI-lh'][0]
FOP2_label_rh = [label for label in labels if label.name == 'R_FOP2_ROI-rh'][0]

Ig_label_lh = [label for label in labels if label.name == 'L_Ig_ROI-lh'][0]
Ig_label_rh = [label for label in labels if label.name == 'R_Ig_ROI-rh'][0]

temp1_label_lh = [label for label in anat_label if label.name == 'parsopercularis-lh'][0]
temp1_label_rh = [label for label in anat_label if label.name == 'parsopercularis-rh'][0]
temp2_label_lh = [label for label in anat_label if label.name == 'parsorbitalis-lh'][0]
temp2_label_rh = [label for label in anat_label if label.name == 'parsorbitalis-rh'][0]
temp3_label_lh = [label for label in anat_label if label.name == 'parstriangularis-lh'][0]
temp3_label_rh = [label for label in anat_label if label.name == 'parstriangularis-rh'][0]
temp4_label_lh = [label for label in anat_label if label.name == 'precentral-lh'][0]
temp4_label_rh = [label for label in anat_label if label.name == 'precentral-rh'][0]

#%%
#new_data = X13[:,:,all_subject,:]
#data1 = np.subtract(np.mean(new_data[:,:,:,[5]],axis=3), np.mean(new_data[:,:,:,[0]],axis=3))
#data1 = np.mean(new_data[:,:,:,[5]],axis=3)
#del new_data

#lex_hC_lN = X13[:,:,:,5]
#dot_hC_lN = X13[:,:,:,0]
data11 = np.mean(X11[:,:,:,[5]],axis=3) - np.mean(X11[:,:,:,[8]],axis=3)
data11 = np.transpose(data11,[2,1,0])

#%%
stat_fun = partial(mne.stats.ttest_1samp_no_p)

s_group = all_subject

p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(s_group) - 1)

#subjects_dir = mne.utils.get_subjects_dir(subjects_dir, raise_error=True)
#label_dir = op.join(fs_dir, 'fsaverage', 'label')
#lh = mne.read_label(op.join(label_dir, 'lh.Medial_wall.label'))
#rh = mne.read_label(op.join(label_dir, 'rh.Medial_wall.label'))
#medial_vertices = np.concatenate((lh.vertices[lh.vertices < 10242], rh.vertices[rh.vertices < 10242] + 10242))

#lex_hC_lN[medial_vertices,:,:] = 0

#if concatenate is True:
#    return np.concatenate((lh.vertices[lh.vertices < 10242],
#                           rh.vertices[rh.vertices < 10242] + 10242))
#else: 
#    return [lh.vertices, rh.vertices]
#    
temp3 = mne.SourceEstimate(np.transpose(stat_fun(data11[s_group,:,:])), fs_vertices, tmin, tstep, subject='fsaverage')

brain3_1 = temp3.plot(hemi='lh', subjects_dir=fs_dir, views = 'lat', initial_time=0.40, #['lat','ven','med']
           clim=dict(kind='value', lims=[1.5, t_threshold, 7])) #clim=dict(kind='value', lims=[2, t_threshold, 7]), size=(800,800))

#temp3 = mne.SourceEstimate(np.mean(np.abs(X11[:,:,:,5]),axis=2), fs_vertices, tmin, tstep, subject='fsaverage')
#
#brain3_1 = temp3.plot(hemi='lh', subjects_dir=fs_dir, views = ['lat','ven','med'], initial_time=0.26, #['lat','ven','med']
#           clim=dict(kind='value', lims=[2, 2.5, 5])) #clim=dict(kind='value', lims=[2, t_threshold, 7]), size=(800,800))

brain3_1.save_movie('Lex_LH_free_depth8_all_Word_Normal.mp4',time_dilation = 6.0,framerate = 24)

"""
plot(self, subject=None, surface='inflated', hemi='lh', colormap='auto', 
     time_label='auto', smoothing_steps=10, transparent=None, alpha=1.0, 
     time_viewer=False, subjects_dir=None, figure=None, views='lat', 
     colorbar=True, clim='auto', cortex='classic', size=800, background='black', 
     foreground='white', initial_time=None, time_unit='s')
"""

#%%
""" Spatio-temporal clustering """
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
s_space = mne.grade_to_tris(5)

# Left hemisphere
s_space_lh = s_space[s_space[:,0] < 10242]
connectivity = mne.spatial_tris_connectivity(s_space_lh, remap_vertices = True)

T_obs, clusters, cluster_p_values, H0 = clu = \
    mne.stats.spatio_temporal_cluster_1samp_test(data11[:,:,0:10242], connectivity=connectivity, n_jobs=12,
                                       threshold=t_threshold)
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
fsave_vertices = [np.arange(10242), np.array([], int)]
stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep,
                                             vertices=fsave_vertices,
                                             subject='fsaverage')

#%%
brain1 = stc_all_cluster_vis.plot(
    hemi='lh', views='lateral', subjects_dir=fs_dir,
    time_label='Duration significant (ms)', size=(800, 800),
    smoothing_steps=5, clim=dict(kind='value', lims=[20, 40, 300]))

#brain1.add_label(A4_label_lh, borders=True, color=c_table[2])
#brain1.add_label(A5_label_lh, borders=True, color=c_table[2])
#brain1.add_label(STSdp_label_lh, borders=True, color=c_table[2])
#brain1.add_label(TPOJ1_label_lh, borders=True, color=c_table[2])
#brain1.add_label(PBelt_label_lh, borders=True, color=c_table[2])
#brain1.add_label(LBelt_label_lh, borders=True, color=c_table[2])
brain1.add_label(A1_label_lh, borders=True, color='k')
temp_auditory_label_l = mne.Label(A4_label_lh.vertices, hemi='lh',name=u'sts_l',pos=A4_label_lh.pos,values= A4_label_lh.values) + \
        mne.Label(A5_label_lh.vertices, hemi='lh',name=u'sts_l',pos=A5_label_lh.pos,values= A5_label_lh.values) + \
        mne.Label(STSdp_label_lh.vertices, hemi='lh',name=u'sts_l',pos=STSdp_label_lh.pos,values= STSdp_label_lh.values)+ \
        mne.Label(TPOJ1_label_lh.vertices, hemi='lh',name=u'sts_l',pos=TPOJ1_label_lh.pos,values= TPOJ1_label_lh.values)+ \
        mne.Label(PBelt_label_lh.vertices, hemi='lh',name=u'sts_l',pos=PBelt_label_lh.pos,values= PBelt_label_lh.values)+ \
        mne.Label(LBelt_label_lh.vertices, hemi='lh',name=u'sts_l',pos=LBelt_label_lh.pos,values= LBelt_label_lh.values)

brain1.add_label(temp_auditory_label_l, borders=True, color=c_table[2])

lh_label = stc_all_cluster_vis.in_label(temp_auditory_label_l)
data = lh_label.data
lh_label.data[data <= 40] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = stc_all_cluster_vis.in_label(temp_labels)
aud_vertices_l = temp.vertices[0]

new_label = mne.Label(aud_vertices_l, hemi='lh')
brain1.add_label(new_label, borders=True, color=c_table[2])

#brain1.add_label(PFcm_label_lh, borders=True, color=c_table[0])
#brain1.add_label(PFop_label_lh, borders=True, color=c_table[0])
#brain1.add_label(RI_label_lh, borders=True, color=c_table[0])
#brain1.add_label(PF_label_lh, borders=True, color=c_table[0])
#brain1.add_label(PFt_label_lh, borders=True, color=c_table[0])
temp_auditory2_label_l = mne.Label(PFcm_label_lh.vertices, hemi='lh',name=u'sts_l',pos=PFcm_label_lh.pos,values= PFcm_label_lh.values) + \
        mne.Label(PFop_label_lh.vertices, hemi='lh',name=u'sts_l',pos=PFop_label_lh.pos,values= PFop_label_lh.values) + \
        mne.Label(RI_label_lh.vertices, hemi='lh',name=u'sts_l',pos=RI_label_lh.pos,values= RI_label_lh.values)+ \
        mne.Label(PF_label_lh.vertices, hemi='lh',name=u'sts_l',pos=PF_label_lh.pos,values= PF_label_lh.values)+ \
        mne.Label(PFt_label_lh.vertices, hemi='lh',name=u'sts_l',pos=PFt_label_lh.pos,values= PFt_label_lh.values)

brain1.add_label(temp_auditory2_label_l, borders=True, color=c_table[0])

lh_label = stc_all_cluster_vis.in_label(temp_auditory2_label_l)
data = lh_label.data
lh_label.data[data <= 40] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = stc_all_cluster_vis.in_label(temp_labels)
aud2_vertices_l = temp.vertices[0]

new_label = mne.Label(aud2_vertices_l, hemi='lh')
brain1.add_label(new_label, borders=True, color=c_table[0])

#brain1.add_label(_3a_label_lh, borders=True, color=c_table[4])
#brain1.add_label(_3b_label_lh, borders=True, color=c_table[4])
#brain1.add_label(_4_label_lh, borders=True, color=c_table[4])
temp_motor_label_l = mne.Label(_3a_label_lh.vertices, hemi='lh',name=u'sts_l',pos=_3a_label_lh.pos,values= _3a_label_lh.values) + \
        mne.Label(_3b_label_lh.vertices, hemi='lh',name=u'sts_l',pos=_3b_label_lh.pos,values= _3b_label_lh.values) + \
        mne.Label(_4_label_lh.vertices, hemi='lh',name=u'sts_l',pos=_4_label_lh.pos,values= _4_label_lh.values)

brain1.add_label(temp_motor_label_l, borders=True, color=c_table[4])

lh_label = stc_all_cluster_vis.in_label(temp_motor_label_l)
data = lh_label.data
lh_label.data[data <= 40] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = stc_all_cluster_vis.in_label(temp_labels)
motor_vertices_l = temp.vertices[0]

new_label = mne.Label(motor_vertices_l, hemi='lh')
brain1.add_label(new_label, borders=True, color=c_table[4])

#brain1.add_label(_6r_label_lh, borders=True, color=c_table[6])
#brain1.add_label(a44_label_lh, borders=True, color=c_table[6])
#brain1.add_label(a45_label_lh, borders=True, color=c_table[6])
temp_broca_label_l = mne.Label(_6r_label_lh.vertices, hemi='lh',name=u'sts_l',pos=_6r_label_lh.pos,values= _6r_label_lh.values) + \
        mne.Label(a44_label_lh.vertices, hemi='lh',name=u'sts_l',pos=a44_label_lh.pos,values= a44_label_lh.values) + \
        mne.Label(a45_label_lh.vertices, hemi='lh',name=u'sts_l',pos=a45_label_lh.pos,values= a45_label_lh.values) + \
        mne.Label(FOP4_label_lh.vertices, hemi='lh',name=u'sts_l',pos=FOP4_label_lh.pos,values= FOP4_label_lh.values) + \
        mne.Label(FOP3_label_lh.vertices, hemi='lh',name=u'sts_l',pos=FOP3_label_lh.pos,values= FOP3_label_lh.values) + \
        mne.Label(FOP5_label_lh.vertices, hemi='lh',name=u'sts_l',pos=FOP5_label_lh.pos,values= FOP5_label_lh.values)
        
brain1.add_label(temp_broca_label_l, borders=True, color=c_table[6])

lh_label = stc_all_cluster_vis.in_label(temp_broca_label_l)
data = lh_label.data
lh_label.data[data <= 40] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = stc_all_cluster_vis.in_label(temp_labels)
broca_vertices_l = temp.vertices[0]

new_label = mne.Label(broca_vertices_l, hemi='lh')
brain1.add_label(new_label, borders=True, color=c_table[6])

#brain1.add_label(FOP5_label_lh, borders=True, color=c_table[6])
#brain1.add_label(FOP4_label_lh, borders=True, color=c_table[6])
#brain1.add_label(FOP3_label_lh, borders=True, color=c_table[6])
#brain1.add_label(FOP2_label_lh, borders=True, color=c_table[6])
#
#brain1.add_label(p47r_label_lh, borders=True, color=c_table[8])
#brain1.add_label(IFSa_label_lh, borders=True, color=c_table[8])
#brain1.add_label(a9_46v_lh, borders=True, color=c_table[8])
temp_frontal_label_l = mne.Label(p47r_label_lh.vertices, hemi='lh',name=u'sts_l',pos=p47r_label_lh.pos,values= p47r_label_lh.values) + \
        mne.Label(IFSa_label_lh.vertices, hemi='lh',name=u'sts_l',pos=IFSa_label_lh.pos,values= IFSa_label_lh.values) + \
        mne.Label(a9_46v_lh.vertices, hemi='lh',name=u'sts_l',pos=a9_46v_lh.pos,values= a9_46v_lh.values)

brain1.add_label(temp_frontal_label_l, borders=True, color=c_table[8])

lh_label = stc_all_cluster_vis.in_label(temp_frontal_label_l)
data = lh_label.data
lh_label.data[data <= 40] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = stc_all_cluster_vis.in_label(temp_labels)
frontal_vertices_l = temp.vertices[0]

new_label = mne.Label(frontal_vertices_l, hemi='lh')
brain1.add_label(new_label, borders=True, color=c_table[8])

#brain1.add_label(OP23_label_lh, borders=True, color='k')
#brain1.add_label(Ig_label_lh, borders=True, color='k')
#brain1.add_label(OP4_label_lh, borders=True, color='k')
#brain1.add_label(OP1_label_lh, borders=True, color='k')
temp_sylvian_label_l = mne.Label(OP23_label_lh.vertices, hemi='lh',name=u'sts_l',pos=OP23_label_lh.pos,values= OP23_label_lh.values) + \
        mne.Label(Ig_label_lh.vertices, hemi='lh',name=u'sts_l',pos=Ig_label_lh.pos,values= Ig_label_lh.values) + \
        mne.Label(OP4_label_lh.vertices, hemi='lh',name=u'sts_l',pos=OP4_label_lh.pos,values= OP4_label_lh.values) + \
        mne.Label(OP1_label_lh.vertices, hemi='lh',name=u'sts_l',pos=OP1_label_lh.pos,values= OP1_label_lh.values) + \
        mne.Label(FOP2_label_lh.vertices, hemi='lh',name=u'sts_l',pos=FOP2_label_lh.pos,values= FOP2_label_lh.values)
        
brain1.add_label(temp_sylvian_label_l, borders=True, color=c_table[8])

lh_label = stc_all_cluster_vis.in_label(temp_sylvian_label_l)
data = lh_label.data
lh_label.data[data <= 40] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = stc_all_cluster_vis.in_label(temp_labels)
sylvian_vertices_l = temp.vertices[0]

new_label = mne.Label(sylvian_vertices_l, hemi='lh')
brain1.add_label(new_label, borders=True, color=c_table[8])

#%% Right hemisphere
s_space_rh = s_space[s_space[:,0] >= 10242]
connectivity = mne.spatial_tris_connectivity(s_space_rh, remap_vertices = True)

T_obs, clusters, cluster_p_values, H0 = clu = \
    mne.stats.spatio_temporal_cluster_1samp_test(data11[:,:,10242:], connectivity=connectivity, n_jobs=18,
                                       threshold=t_threshold)
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
fsave_vertices = [np.array([], int), np.arange(10242)]
stc_all_cluster_vis2 = summarize_clusters_stc(clu, tstep=tstep,
                                             vertices=fsave_vertices,
                                             subject='fsaverage')

brain2 = stc_all_cluster_vis2.plot(
    hemi='rh', views='lateral', subjects_dir=fs_dir,
    time_label='Duration significant (ms)', size=(800, 800),
    smoothing_steps=5, clim=dict(kind='value', lims=[10, 100, 300]))

#%%
""" AUD 1 """
M = np.mean(np.mean(X11[aud_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(X11[aud_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(subs))

plt.figure(1)
plt.clf()  

plt.subplot(1,2,1)
plt.hold(True)
plt.plot(times, M[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,5]-errM[:,5], M[:,5]+errM[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,8],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,8]-errM[:,8], M[:,8]+errM[:,8], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 3])
plt.title('Lexical task')

plt.subplot(1,2,2)
plt.hold(True)
plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,3],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 3])
plt.title('Dot task')

#%%
""" AUD 2 """

M = np.mean(np.mean(X11[aud2_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(X11[aud2_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))
temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[aud2_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[aud2_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1
temp2 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp2[aud2_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp2[aud2_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp2

plotit(times, M, M1, M2, errM, errM1, errM2, yMin=-0, yMax=3)

#%%
""" Motor """

M = np.mean(np.mean(X11[motor_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(X11[motor_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))
temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[motor_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[motor_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1
temp2 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp2[motor_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp2[motor_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp2

plotit(times, M, M1, M2, errM, errM1, errM2, yMin=-0, yMax=3)

#%%
""" Broca """

M = np.mean(np.mean(X11[broca_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(X11[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))
temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1
temp2 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp2[broca_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp2[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp2

plotit(times, M, M1, M2, errM, errM1, errM2, yMin=-0, yMax=3, title='Broca')

#%%
""" Sylvian """

M = np.mean(np.mean(X11[broca_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(X11[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))
temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1
temp2 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp2[broca_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp2[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp2

plotit(times, M, M1, M2, errM, errM1, errM2, yMin=-0, yMax=3, title='Sylvian')

#%%
""" Frontal """

M = np.mean(np.mean(X11[frontal_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(X11[frontal_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))
temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[frontal_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[frontal_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1
temp2 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp2[frontal_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp2[frontal_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp2

plotit(times, M, M1, M2, errM, errM1, errM2, yMin=-0, yMax=3, title='Frontal')


#%%
""" VTC """
task = 5

plt.figure(2)
plt.clf()

M = np.mean(np.mean(X11[vtc_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(X11[vtc_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))
temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[vtc_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[vtc_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1
temp2 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp2[vtc_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp2[vtc_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp2

plt.subplot(2,3,1)
plt.hold(True)

plt.plot(times, M[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,task]-errM[:,task], M[:,task]+errM[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,task+1]-errM[:,task+1], M[:,task+1]+errM[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,task+3]-errM[:,task+3], M[:,task+3]+errM[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Lexical task: VTC')

plt.subplot(2,3,2)
plt.hold(True)

plt.plot(times, M1[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,task]-errM1[:,task], M1[:,task]+errM1[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,task+1]-errM1[:,task+1], M1[:,task+1]+errM1[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,task+3]-errM1[:,task+3], M1[:,task+3]+errM1[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Good Readers')
#plt.legend(loc='upper right')

plt.subplot(2,3,3)
plt.hold(True)

plt.plot(times, M2[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,task]-errM2[:,task], M2[:,task]+errM2[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,task+1]-errM2[:,task+1], M2[:,task+1]+errM2[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,task+3]-errM2[:,task+3], M2[:,task+3]+errM2[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Poor Readers')

task = 0

plt.subplot(2,3,4)
plt.hold(True)

plt.plot(times, M[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,task]-errM[:,task], M[:,task]+errM[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,task+1]-errM[:,task+1], M[:,task+1]+errM[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,task+3]-errM[:,task+3], M[:,task+3]+errM[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Dot task: VTC')

plt.subplot(2,3,5)
plt.hold(True)

plt.plot(times, M1[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,task]-errM1[:,task], M1[:,task]+errM1[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,task+1]-errM1[:,task+1], M1[:,task+1]+errM1[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,task+3]-errM1[:,task+3], M1[:,task+3]+errM1[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Good Readers')
#plt.legend(loc='upper right')

plt.subplot(2,3,6)
plt.hold(True)

plt.plot(times, M2[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,task]-errM2[:,task], M2[:,task]+errM2[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,task+1]-errM2[:,task+1], M2[:,task+1]+errM2[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,task+3]-errM2[:,task+3], M2[:,task+3]+errM2[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Poor Readers')

#%%
temp1 = X11[vtc_vertices_l,:,:,:]
s_group = good_readers
for iSub in np.arange(0,len(s_group)):
    plt.figure(100+iSub)
    plt.clf()
    plt.subplot(1,2,1)
    plt.hold(True)
    plt.plot(times, np.mean(temp1[:,:,s_group[iSub],0],axis=0), '-', color=c_table[5])
    plt.plot(times, np.mean(temp1[:,:,s_group[iSub],1],axis=0), '-', color=c_table[3])
    plt.plot(times, np.mean(temp1[:,:,s_group[iSub],3],axis=0), '-', color=c_table[1])
##    plt.plot([0.1, 0.1],[0, 8],'-',color='k')
    plt.title(subs[s_group[iSub]])
    plt.subplot(1,2,2)
    plt.hold(True)
    plt.plot(times, np.mean(temp1[:,:,s_group[iSub],5],axis=0), '-', color=c_table[5])
    plt.plot(times, np.mean(temp1[:,:,s_group[iSub],6],axis=0), '-', color=c_table[3])
    plt.plot(times, np.mean(temp1[:,:,s_group[iSub],8],axis=0), '-', color=c_table[1])
#    plt.plot([0.1, 0.1],[0, 8],'-',color='k')
    plt.title(subs[s_group[iSub]])

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
""" V1 responses """
task = 5

plt.figure(5)
plt.clf()

M = np.mean(np.mean(X11[v1_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(X11[v1_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))
temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[v1_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[v1_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1
temp2 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp2[v1_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp2[v1_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp2

plt.subplot(2,3,1)
plt.hold(True)

plt.plot(times, M[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,task]-errM[:,task], M[:,task]+errM[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,task+1]-errM[:,task+1], M[:,task+1]+errM[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,task+3]-errM[:,task+3], M[:,task+3]+errM[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 7])
plt.title('Lexical task: V1')

plt.subplot(2,3,2)
plt.hold(True)

plt.plot(times, M1[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,task]-errM1[:,task], M1[:,task]+errM1[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,task+1]-errM1[:,task+1], M1[:,task+1]+errM1[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,task+3]-errM1[:,task+3], M1[:,task+3]+errM1[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 7])
plt.title('Good Readers')
#plt.legend(loc='upper right')

plt.subplot(2,3,3)
plt.hold(True)

plt.plot(times, M2[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,task]-errM2[:,task], M2[:,task]+errM2[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,task+1]-errM2[:,task+1], M2[:,task+1]+errM2[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,task+3]-errM2[:,task+3], M2[:,task+3]+errM2[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 7])
plt.title('Poor Readers')

task = 0

plt.subplot(2,3,4)
plt.hold(True)

plt.plot(times, M[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,task]-errM[:,task], M[:,task]+errM[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,task+1]-errM[:,task+1], M[:,task+1]+errM[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,task+3]-errM[:,task+3], M[:,task+3]+errM[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 7])
plt.title('Dot task: V1')

plt.subplot(2,3,5)
plt.hold(True)

plt.plot(times, M1[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,task]-errM1[:,task], M1[:,task]+errM1[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,task+1]-errM1[:,task+1], M1[:,task+1]+errM1[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,task+3]-errM1[:,task+3], M1[:,task+3]+errM1[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 7])
plt.title('Good Readers')
#plt.legend(loc='upper right')

plt.subplot(2,3,6)
plt.hold(True)

plt.plot(times, M2[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,task]-errM2[:,task], M2[:,task]+errM2[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,task+1]-errM2[:,task+1], M2[:,task+1]+errM2[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,task+3]-errM2[:,task+3], M2[:,task+3]+errM2[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 7])
plt.title('Poor Readers')

#%%
plt.figure(6)
plt.clf()

task = 5

plt.subplot(2,3,1)
plt.hold(True)

plt.plot(times, M[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,task]-errM[:,task], M[:,task]+errM[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+2],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,task+2]-errM[:,task+2], M[:,task+2]+errM[:,task+2], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+4],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,task+4]-errM[:,task+4], M[:,task+4]+errM[:,task+4], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Lexical task: VTC')

plt.subplot(2,3,2)
plt.hold(True)

plt.plot(times, M1[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,task]-errM1[:,task], M1[:,task]+errM1[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+2],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,task+2]-errM1[:,task+2], M1[:,task+2]+errM1[:,task+2], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+4],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,task+4]-errM1[:,task+4], M1[:,task+4]+errM1[:,task+4], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Good Readers')
#plt.legend(loc='upper right')

plt.subplot(2,3,3)
plt.hold(True)

plt.plot(times, M2[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,task]-errM2[:,task], M2[:,task]+errM2[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+2],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,task+2]-errM2[:,task+2], M2[:,task+2]+errM2[:,task+2], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+4],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,task+4]-errM2[:,task+4], M2[:,task+4]+errM2[:,task+4], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Poor Readers')

task = 0

plt.subplot(2,3,4)
plt.hold(True)

plt.plot(times, M[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,task]-errM[:,task], M[:,task]+errM[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+2],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,task+2]-errM[:,task+2], M[:,task+2]+errM[:,task+2], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+4],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,task+4]-errM[:,task+4], M[:,task+4]+errM[:,task+4], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Dot task: VTC')

plt.subplot(2,3,5)
plt.hold(True)

plt.plot(times, M1[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,task]-errM1[:,task], M1[:,task]+errM1[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+2],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,task+2]-errM1[:,task+2], M1[:,task+2]+errM1[:,task+2], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+4],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,task+4]-errM1[:,task+4], M1[:,task+4]+errM1[:,task+4], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Good Readers')
#plt.legend(loc='upper right')

plt.subplot(2,3,6)
plt.hold(True)

plt.plot(times, M2[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,task]-errM2[:,task], M2[:,task]+errM2[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+2],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,task+2]-errM2[:,task+2], M2[:,task+2]+errM2[:,task+2], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+4],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,task+4]-errM2[:,task+4], M2[:,task+4]+errM2[:,task+4], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Poor Readers')
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
""" Broca """
task = 5

plt.figure(3)
plt.clf()

M = np.mean(np.mean(X11[broca_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(X11[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))
temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1
temp2 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp2[broca_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp2[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp2

plt.subplot(2,3,1)
plt.hold(True)

plt.plot(times, M[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,task]-errM[:,task], M[:,task]+errM[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,task+1]-errM[:,task+1], M[:,task+1]+errM[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,task+3]-errM[:,task+3], M[:,task+3]+errM[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Lexical task: STS')

plt.subplot(2,3,2)
plt.hold(True)

plt.plot(times, M1[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,task]-errM1[:,task], M1[:,task]+errM1[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,task+1]-errM1[:,task+1], M1[:,task+1]+errM1[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,task+3]-errM1[:,task+3], M1[:,task+3]+errM1[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Good Readers')
#plt.legend(loc='upper right')

plt.subplot(2,3,3)
plt.hold(True)

plt.plot(times, M2[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,task]-errM2[:,task], M2[:,task]+errM2[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,task+1]-errM2[:,task+1], M2[:,task+1]+errM2[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,task+3]-errM2[:,task+3], M2[:,task+3]+errM2[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Poor Readers')

task = 0

plt.subplot(2,3,4)
plt.hold(True)

plt.plot(times, M[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,task]-errM[:,task], M[:,task]+errM[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,task+1]-errM[:,task+1], M[:,task+1]+errM[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,task+3]-errM[:,task+3], M[:,task+3]+errM[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Dot task: STS')

plt.subplot(2,3,5)
plt.hold(True)

plt.plot(times, M1[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M1[:,task]-errM1[:,task], M1[:,task]+errM1[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M1[:,task+1]-errM1[:,task+1], M1[:,task+1]+errM1[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M1[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M1[:,task+3]-errM1[:,task+3], M1[:,task+3]+errM1[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Good Readers')
#plt.legend(loc='upper right')

plt.subplot(2,3,6)
plt.hold(True)

plt.plot(times, M2[:,task],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M2[:,task]-errM2[:,task], M2[:,task]+errM2[:,task], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M2[:,task+1]-errM2[:,task+1], M2[:,task+1]+errM2[:,task+1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M2[:,task+3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M2[:,task+3]-errM2[:,task+3], M2[:,task+3]+errM2[:,task+3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 5])
plt.title('Poor Readers')

#%%
""" Lexical task: All subjects """
plt.figure(4)
plt.clf()

X11 = np.abs(X13[ventral_vertices,:,:,:])
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
plt.ylim([-1, 5])
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
plt.ylim([-1, 5])
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
plt.ylim([-1, 5])
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
