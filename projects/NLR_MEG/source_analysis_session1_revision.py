# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:05:40 2016

@author: sjjoo
"""
#%%
#import sys
import mne
#import imageio
from mne.utils import run_subprocess, logger
import os
from os import path as op
import copy
#import shutil
import numpy as np
from numpy.random import randn
from scipy import stats as stats
#import scipy.io as sio
import time
from functools import partial
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)

from mne import set_config

import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from pandas import DataFrame
from sklearn import linear_model
import statsmodels.api as sm

#import csv
os.chdir('/home/sjjoo/git/BrainTools/projects/NLR_MEG')
from plotit3 import plotit3
from plotsig3 import plotsig3
from plotit2 import plotit2
from plotsig2 import plotsig2
from plotcorr3 import plotcorr3

set_config('MNE_MEMMAP_MIN_SIZE', '1M')
set_config('MNE_CACHE_DIR', '.tmp')

mne.set_config('MNE_USE_CUDA', 'true')

this_env = copy.copy(os.environ)
fs_dir = '/mnt/scratch/subjects'
this_env['SUBJECTS_DIR'] = fs_dir

raw_dir = '/mnt/scratch/NLR_MEG'

os.chdir(raw_dir)

#%%
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

subs2 = ['NLR_102_RS','NLR_110_HH','NLR_145_AC','NLR_150_MG',
        'NLR_152_TC','NLR_160_EK','NLR_161_AK','NLR_162_EF','NLR_163_LF', # 162, 201 only had the second session
        'NLR_164_SF','NLR_170_GM','NLR_172_TH','NLR_174_HS','NLR_179_GM', # 'NLR_170_GM': no EOG channel
        'NLR_180_ZD','NLR_201_GS',
        'NLR_204_AM','NLR_205_AC','NLR_207_AH','NLR_210_SB','NLR_211_LB',
        'NLR_GB310','NLR_KB218','NLR_GB267','NLR_JB420', 'NLR_HB275','NLR_GB355'] 

session2 = ['102_rs160815','110_hh160809','145_ac160823','150_mg160825',
       '152_tc160623','160_ek160915','161_ak160916','162_ef160829','163_lf160920',
       '164_sf160920','170_gm160822','172_th160825','174_hs160829','179_gm160913',
       '180_zd160826','201_gs150925',
       '204_am151120','205_ac160202','207_ah160809','210_sb160822','211_lb160823',
       'nlr_gb310170829','nlr_kb218170829','nlr_gb267170911','nlr_jb420170828','nlr_hb275170828','nlr_gb355170907']

subIndex1 = np.nonzero(np.in1d(subs,subs2))[0]
subIndex2 = np.empty([1,len(subIndex1)],dtype=int)[0]
for i in range(0,len(subIndex1)):
    subIndex2[i] = np.nonzero(np.in1d(subs2,subs[subIndex1[i]]))[0]

twre_index = [87,93,108,66,116,85,110,71,84,92,87,86,63,81,60,55,71,63,68,67,64,127,79,
              73,59,84,79,91,57,67,77,57,80,53,72,58,85,79,116,117,107,78,66,101,67]
twre_index = np.array(twre_index)

brs = [87,102,108,78,122,91,121,77,91,93,93,88,75,90,66,59,81,84,81,72,71,121,
       81,75,66,90,93,101,56,78,83,69,88,60,88,73,82,81,115,127,124,88,68,110,96]
brs = np.array(brs)

twre_index1 = twre_index[subIndex1]
twre_index2_all = [90,76,94,115,
               85,75,82,64,75,
               63,83,77,84,75,
               68,79,
               62,90,105,75,71,
               69,83,76,62,73,94]

twre_index2_all = np.array(twre_index2_all)
twre_index2 = twre_index2_all[subIndex2]

brs1 = brs[subIndex1]
brs2_all = [98,88,102,110,99,91,88,79,105,86,81,88,89,77,83,81,86,98,116,104,86,90,91,97,57,99,102]
brs2_all = np.array(brs2_all)
brs2 = brs2_all[subIndex2]

twre_diff = np.subtract(twre_index2,twre_index1)
brs_diff = np.subtract(brs2,brs1)

swe_raw = [62, 76, 74, 42, 75, 67, 76, 21, 54, 35, 21, 61, 45, 48, 17, 11, 70, 19, 10, 57,
           12, 86, 53, 51, 13, 28, 54, 25, 27, 10, 66, 18, 18, 20, 37, 23, 17, 36, 79, 82,
           74, 64, 42, 78, 35]
swe_raw = np.array(swe_raw)

lwid = [49,60,60,51,62,54,65,23,44,35,31,52,44,39,27,30,57,33,24,48,19,66,45,
        43,22,33,51,36,35,25,55,34,26,26,39,27,24,29,61,71,65,56,36,62,51]
lwid = np.array(lwid)

rf = [88,103,95,67,120,85,108,71,91,87,88,76,76,93,60,40,86,61,66,81,59,130,93,85,49,76,90,96,42,64,74,49,84,56,
      76,61,80,89,111,120,132,88,65,102,72]
rf = np.array(rf)

age = [125.6885, 132.9501, 122.0434, 138.4349, 97.6347, 138.1420, 108.2457, 98.0631, 105.8147, 89.9132,
       87.6465, 131.8660, 123.7174, 95.959, 112.416, 133.8042, 152.4639, 103.4823, 89.8475, 138.4020,
       93.8568, 117.0814, 123.6202, 122.9304, 109.1656, 90.6058,
       111.9593,86.0381,147.2063,95.8699,148.0802,122.5896,88.7162,123.0495,110.6645,105.3069,88.9143,95.2879,106.2852,
       122.2915,114.4389,136.1496,128.6246,137.9216,122.7528]
age = np.divide(age, 12)

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

fname_data = op.join(raw_dir, 'session1_data_loose_depth8_normal.npy')

#%%
"""
Here we load the data for Session 1
"""            
t0 = time.time()

os.chdir(raw_dir)
X13 = np.load(fname_data)
orig_times = np.load('session1_times.npy')
tstep = np.load('session1_tstep.npy')
n_epochs = np.load('session1_n_averages.npy')
tmin = -0.1

""" Downsample the data """
ss = 3 # was originally 2
sample = np.arange(0,len(orig_times),ss) 
sRate = 600 / ss

times = orig_times[sample]
tstep = ss*tstep
X11 = X13[:,sample,:,:]
del X13
X11 = np.abs(X11)

print("\n\nElasped time: %0.2d mins %0.2d secs\n\n" % (divmod(time.time()-t0, 60)))

#%%
""" Grouping subjects """
reading_thresh = 80

m1 = np.logical_and(np.transpose(twre_index) > reading_thresh, np.transpose(age) <= 13)
m2 = np.logical_and(np.transpose(twre_index) <= reading_thresh, np.transpose(age) <= 13)

#m1 = np.logical_and(np.transpose(brs) >= reading_thresh, np.transpose(age) <= 13)
#m2 = np.logical_and(np.transpose(brs) < reading_thresh, np.transpose(age) <= 13)

#m1 = np.logical_and(np.transpose(swe_raw) >= np.median(swe_raw), np.transpose(age) <= 13)
#m2 = np.logical_and(np.transpose(swe_raw) < np.median(swe_raw), np.transpose(age) <= 13)

orig_twre = twre_index
orig_age = age
orig_swe = swe_raw

m3 = np.mean(n_epochs,axis=1) < 40

m1[np.where(m3)] = False
m2[np.where(m3)] = False
twre_index = twre_index[np.where(~m3)[0]]
age = age[np.where(~m3)[0]]
swe_raw = swe_raw[np.where(~m3)[0]]

good_readers = np.where(m1)[0]
poor_readers = np.where(m2)[0]

a1 = np.transpose(age) > np.mean(age)
a2 = np.logical_not(a1)

a1[np.where(m3)] = False
a2[np.where(m3)] = False

old_readers = np.where(a1)[0]
young_readers = np.where(a2)[0]

all_subject = []
all_subject.extend(good_readers)
all_subject.extend(poor_readers)
all_subject.sort()

fs_vertices = [np.arange(10242)] * 2

#%%
""" Read HCP labels """
labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', surf_name='white', subjects_dir=fs_dir) #regexp=aparc_label_name
#aparc_label_name = 'PHT_ROI'#'_IP'#'IFSp_ROI'#'STSvp_ROI'#'STSdp_ROI'#'PH_ROI'#'TE2p_ROI' #'SFL_ROI' #'IFSp_ROI' #'TE2p_ROI' #'inferiortemporal' #'pericalcarine'
anat_label = mne.read_labels_from_annot('fsaverage', parc='aparc.a2009s',surf_name='white',
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
IPS1_label_lh = [label for label in labels if label.name == 'L_IPS1_ROI-lh'][0]
IPS1_label_rh = [label for label in labels if label.name == 'R_IPS1_ROI-rh'][0]
_7Am_label_lh = [label for label in labels if label.name == 'L_7Am_ROI-lh'][0]
_7Am_label_rh = [label for label in labels if label.name == 'R_7Am_ROI-rh'][0]
VIP_label_lh = [label for label in labels if label.name == 'L_VIP_ROI-lh'][0]
VIP_label_rh = [label for label in labels if label.name == 'R_VIP_ROI-rh'][0]
_7AL_label_lh = [label for label in labels if label.name == 'L_7AL_ROI-lh'][0]
_7AL_label_rh = [label for label in labels if label.name == 'R_7AL_ROI-rh'][0]

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

AVI_label_lh = [label for label in labels if label.name == 'L_AVI_ROI-lh'][0]
AVI_label_rh = [label for label in labels if label.name == 'R_AVI_ROI-rh'][0]
_47l_label_lh = [label for label in labels if label.name == 'L_47l_ROI-lh'][0]
_47l_label_rh = [label for label in labels if label.name == 'R_47l_ROI-rh'][0]

temp1_label_lh = [label for label in anat_label if label.name == 'Pole_occipital-lh'][0]
#temp1_label_rh = [label for label in anat_label if label.name == 'parsopercularis-rh'][0]
temp2_label_lh = [label for label in anat_label if label.name == 'S_occipital_ant-lh'][0]
#temp2_label_rh = [label for label in anat_label if label.name == 'parsorbitalis-rh'][0]
temp3_label_lh = [label for label in anat_label if label.name == 'G_and_S_occipital_inf-lh'][0]
#temp3_label_rh = [label for label in anat_label if label.name == 'parstriangularis-rh'][0]
temp4_label_lh = [label for label in anat_label if label.name == 'S_calcarine-lh'][0]
#temp4_label_rh = [label for label in anat_label if label.name == 'precentral-rh'][0]

#%%
""" Lexical task: Word - Noise """
data11 = X11[:,:,all_subject,5] - X11[:,:,all_subject,8]
data11 = np.transpose(data11,[2,1,0])

""" Lexical and dot combined: Word - Noise """
data12 = (X11[:,:,all_subject,5] + X11[:,:,all_subject,0])/2 \
    - (X11[:,:,all_subject,8] + X11[:,:,all_subject,3])/2
data12 = np.transpose(data12,[2,1,0])

#%%
""" Spatio-temporal clustering: session 1 Lexical task"""
t0 = time.time()
print("\n\n Start time: %s \n\n" % time.ctime())

p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
s_space = mne.grade_to_tris(5)

# Left hemisphere
s_space_lh = s_space[s_space[:,0] < 10242]
#connectivity = mne.spatial_tris_connectivity(s_space_lh, remap_vertices = True)
connectivity = mne.spatial_tris_connectivity(s_space)

T_obs, clusters, cluster_p_values, H0 = clu = \
    mne.stats.spatio_temporal_cluster_1samp_test(data11[:,:,:], n_permutations=1024, connectivity=connectivity, n_jobs=12,
                                       threshold=t_threshold)
good_cluster_inds = np.where(cluster_p_values < p_threshold)[0]

#fsave_vertices = [np.arange(10242), np.array([], int)]
fsave_vertices = [np.arange(10242), np.arange(10242)]
#fsave_vertices = [np.arange(10242), np.array([], int)]

stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep, 
                                             vertices=fsave_vertices,
                                             subject='fsaverage')
print("\n\n Elasped time: %0.2d mins %0.2d secs" % (divmod(time.time()-t0, 60)))

#%%
""" Spatio-temporal clustering: session 1 lexical+dot task"""
dur_thresh = 100
t0 = time.time()

T_obs, clusters, cluster_p_values, H0 = clu = \
    mne.stats.spatio_temporal_cluster_1samp_test(data12[:,:,:], n_permutations=1024, connectivity=connectivity, n_jobs=12,
                                       threshold=t_threshold)
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
fsave_vertices = [np.arange(10242), np.arange(10242)]
stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep, 
                                             vertices=fsave_vertices,
                                             subject='fsaverage')
print("\n\nElasped time: %0.2d mins %0.2d secs" % (divmod(time.time()-t0, 60)))

brain3 = stc_all_cluster_vis.plot(
    hemi='lh', views='lateral', subjects_dir=fs_dir,
    time_label='Duration significant (ms)', size=(800, 800),
    smoothing_steps=20, clim=dict(kind='value', lims=[40, dur_thresh, 200]),background='white',foreground='black')

#%%
""" ROI definition """
#dur_thresh = 100
#"""
#plot(self, subject=None, surface='inflated', hemi='lh', colormap='auto', 
#     time_label='auto', smoothing_steps=10, transparent=None, alpha=1.0, 
#     time_viewer=False, subjects_dir=None, figure=None, views='lat', 
#     colorbar=True, clim='auto', cortex='classic', size=800, background='black', 
#     foreground='white', initial_time=None, time_unit='s')
#"""
#brain1 = stc_all_cluster_vis.plot(
#    hemi='lh', views='lateral', subjects_dir=fs_dir,
#    time_label='Duration significant (ms)', size=(800, 800),
#    smoothing_steps=20, clim=dict(kind='value', lims=[40, dur_thresh, 200]),background='white',foreground='black')

""" Sort out vertices here """
#temp_frontal_label_l = mne.Label(FOP4_label_lh.vertices, hemi='lh',name=u'sts_l',pos=FOP4_label_lh.pos, \
#                                 values= FOP4_label_lh.values)
#
#brain1.add_label(temp_frontal_label_l, borders=True, color=c_table[8])
#
#lh_label = stc_all_cluster_vis.in_label(temp_frontal_label_l)
#data = lh_label.data
#lh_label.data[data < dur_thresh] = 0.
#
#temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
#                      subjects_dir=fs_dir, connected=False)
#temp = stc_all_cluster_vis.in_label(temp_labels)
#frontal_vertices_l = temp.vertices[0]
#
#new_label = mne.Label(frontal_vertices_l, hemi='lh')
#brain1.add_label(new_label, borders=True, color=c_table[8])
""" Done """

os.chdir('figures_rev')
#brain1.save_image('Lexical_LH_STClustering.pdf', antialiased=True)
#brain1.save_image('Lexical_LH_STClustering.png', antialiased=True)
os.chdir('..')

brain3.add_label(A1_label_lh, borders=True, color=[0,0,0]) # Show A1

temp_auditory_label_l = mne.Label(A4_label_lh.vertices, hemi='lh',name=u'sts_l',pos=A4_label_lh.pos,values= A4_label_lh.values) + \
        mne.Label(A5_label_lh.vertices, hemi='lh',name=u'sts_l',pos=A5_label_lh.pos,values= A5_label_lh.values) + \
        mne.Label(STSdp_label_lh.vertices, hemi='lh',name=u'sts_l',pos=STSdp_label_lh.pos,values= STSdp_label_lh.values)+ \
        mne.Label(TPOJ1_label_lh.vertices, hemi='lh',name=u'sts_l',pos=TPOJ1_label_lh.pos,values= TPOJ1_label_lh.values)+ \
        mne.Label(PBelt_label_lh.vertices, hemi='lh',name=u'sts_l',pos=PBelt_label_lh.pos,values= PBelt_label_lh.values)+ \
        mne.Label(LBelt_label_lh.vertices, hemi='lh',name=u'sts_l',pos=LBelt_label_lh.pos,values= LBelt_label_lh.values)

#brain1.add_label(temp_auditory_label_l, borders=True, color=c_table[2])

lh_label = stc_all_cluster_vis.in_label(temp_auditory_label_l)
data = lh_label.data
lh_label.data[data < dur_thresh] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = stc_all_cluster_vis.in_label(temp_labels)
stg_vertices_l = temp.vertices[0]

new_label = mne.Label(stg_vertices_l, hemi='lh')
brain3.add_label(new_label, borders=True, color=c_table[1])
#brain1.remove_labels()

temp_auditory2_label_l = mne.Label(PFcm_label_lh.vertices, hemi='lh',name=u'sts_l',pos=PFcm_label_lh.pos,values= PFcm_label_lh.values) + \
        mne.Label(RI_label_lh.vertices, hemi='lh',name=u'sts_l',pos=RI_label_lh.pos,values= RI_label_lh.values)+ \
        mne.Label(PF_label_lh.vertices, hemi='lh',name=u'sts_l',pos=PF_label_lh.pos,values= PF_label_lh.values)

#brain1.add_label(temp_auditory2_label_l, borders=True, color=c_table[0])

lh_label = stc_all_cluster_vis.in_label(temp_auditory2_label_l)
data = lh_label.data
lh_label.data[data < dur_thresh] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = stc_all_cluster_vis.in_label(temp_labels)
tpj_vertices_l = temp.vertices[0]
tpj_vertices_l = np.sort(np.concatenate((tpj_vertices_l, \
                                          [16, 2051, 2677, 2678, 2679, 5042, 8296, 8297, 8299, 8722, 8723, 9376])))

new_label = mne.Label(tpj_vertices_l, hemi='lh')
brain3.add_label(new_label, borders=True, color=c_table[0])

#brain1.add_label(_1_label_lh, borders=True, color=c_table[4])
temp_motor_label_l = mne.Label(_3a_label_lh.vertices, hemi='lh',name=u'sts_l',pos=_3a_label_lh.pos,values= _3a_label_lh.values) + \
        mne.Label(_3b_label_lh.vertices, hemi='lh',name=u'sts_l',pos=_3b_label_lh.pos,values= _3b_label_lh.values) + \
        mne.Label(_4_label_lh.vertices, hemi='lh',name=u'sts_l',pos=_4_label_lh.pos,values= _4_label_lh.values) + \
        mne.Label(_1_label_lh.vertices, hemi='lh',name=u'sts_l',pos=_1_label_lh.pos,values= _1_label_lh.values)

#brain1.add_label(temp_motor_label_l, borders=True, color=c_table[4])

lh_label = stc_all_cluster_vis.in_label(temp_motor_label_l)
data = lh_label.data
lh_label.data[data < dur_thresh] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = stc_all_cluster_vis.in_label(temp_labels)
motor_vertices_l = temp.vertices[0]

new_label = mne.Label(motor_vertices_l, hemi='lh')
brain3.add_label(new_label, borders=True, color=c_table[4])

temp_broca_label_l = \
        mne.Label(a44_label_lh.vertices, hemi='lh',name=u'sts_l',pos=a44_label_lh.pos,values= a44_label_lh.values) + \
        mne.Label(a45_label_lh.vertices, hemi='lh',name=u'sts_l',pos=a45_label_lh.pos,values= a45_label_lh.values) + \
        mne.Label(AVI_label_lh.vertices, hemi='lh',name=u'sts_l',pos=AVI_label_lh.pos,values= AVI_label_lh.values) + \
        mne.Label(FOP5_label_lh.vertices, hemi='lh',name=u'sts_l',pos=FOP5_label_lh.pos,values= FOP5_label_lh.values) + \
        mne.Label(_47l_label_lh.vertices, hemi='lh',name=u'sts_l',pos=_47l_label_lh.pos,values= _47l_label_lh.values)
        
#brain1.add_label(temp_broca_label_l, borders=True, color=c_table[6])

lh_label = stc_all_cluster_vis.in_label(temp_broca_label_l)
data = lh_label.data
lh_label.data[data < dur_thresh] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = stc_all_cluster_vis.in_label(temp_labels)
broca_vertices_l = temp.vertices[0]
broca_vertices_l = np.sort(np.concatenate((broca_vertices_l,[1187,3107,3108,3109,6745,7690,7691])))

new_label = mne.Label(broca_vertices_l, hemi='lh')
brain3.add_label(new_label, borders=True, color=c_table[6])

temp_sylvian_label_l = mne.Label(OP23_label_lh.vertices, hemi='lh',name=u'sts_l',pos=OP23_label_lh.pos,values= OP23_label_lh.values) + \
        mne.Label(Ig_label_lh.vertices, hemi='lh',name=u'sts_l',pos=Ig_label_lh.pos,values= Ig_label_lh.values) + \
        mne.Label(OP4_label_lh.vertices, hemi='lh',name=u'sts_l',pos=OP4_label_lh.pos,values= OP4_label_lh.values) + \
        mne.Label(OP1_label_lh.vertices, hemi='lh',name=u'sts_l',pos=OP1_label_lh.pos,values= OP1_label_lh.values) + \
        mne.Label(FOP2_label_lh.vertices, hemi='lh',name=u'sts_l',pos=FOP2_label_lh.pos,values= FOP2_label_lh.values) + \
        mne.Label(_6r_label_lh.vertices, hemi='lh',name=u'sts_l',pos=_6r_label_lh.pos,values= _6r_label_lh.values)
        
#brain1.add_label(temp_sylvian_label_l, borders=True, color=c_table[8])

lh_label = stc_all_cluster_vis.in_label(temp_sylvian_label_l)
data = lh_label.data
lh_label.data[data < dur_thresh] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = stc_all_cluster_vis.in_label(temp_labels)
sylvian_vertices_l = temp.vertices[0]
sylvian_vertices_l = np.sort(np.concatenate((sylvian_vertices_l,[905,1892,2825,2526,4157,4158,4159,6239,8290,8293,9194,9203])))

new_label = mne.Label(sylvian_vertices_l, hemi='lh')
brain3.add_label(new_label, borders=True, color=c_table[8])

# right hemisphere
brain2 = stc_all_cluster_vis.plot(
    hemi='rh', views='lateral', subjects_dir=fs_dir,
    time_label='Duration significant (ms)', size=(800, 800),
    smoothing_steps=20, clim=dict(kind='value', lims=[40, dur_thresh, 200]),background='white',foreground='black')

stg_vertices_r = A5_label_rh.vertices
stg_vertices_r = np.sort([2001,2002,2419,2420,2421,2418,2754,2417,13075,13076,13077,13078,\
                          13079,13080,13081,12069,12070,12071,12072])
new_label = mne.Label(stg_vertices_r, hemi='rh')
brain2.add_label(new_label, borders=True, color=c_table[5])

os.chdir('figures_rev')
brain2.save_image('RH_STClustering.pdf', antialiased=True)
brain2.save_image('RH_STClustering.png', antialiased=True)

np.save('STG_Vert_rev', stg_vertices_l)
np.save('IFG_Vert_rev', broca_vertices_l)
np.save('TPJ_Vert_rev', tpj_vertices_l)
np.save('Motor_Vert_rev', motor_vertices_l)
np.save('Sylvian_Vert_rev', sylvian_vertices_l)
np.save('STG_Vert_r_rev', stg_vertices_r)

#%%
figureDir = '%s/figures_rev' % raw_dir
nReps = 3000
boot_pVal = 0.05

#%%
""" Left STG: Word vs. Noise """
stg_vertices_l = np.load('STG_Vert_rev.npy')
temp1 = X11[:,:,all_subject,:]
M = np.mean(np.mean(temp1[stg_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(temp1[stg_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))
del temp1

temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[stg_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[stg_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
diffM1 = np.mean(np.mean(temp1[stg_vertices_l,:,:,5],axis=0),axis=1) - np.mean(np.mean(temp1[stg_vertices_l,:,:,8],axis=0),axis=1)
diffM2 = np.mean(np.mean(temp1[stg_vertices_l,:,:,0],axis=0),axis=1) - np.mean(np.mean(temp1[stg_vertices_l,:,:,3],axis=0),axis=1)
del temp1

temp1 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp1[stg_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp1[stg_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
diffM3 = np.mean(np.mean(temp1[stg_vertices_l,:,:,5],axis=0),axis=1) - np.mean(np.mean(temp1[stg_vertices_l,:,:,8],axis=0),axis=1)
diffM4 = np.mean(np.mean(temp1[stg_vertices_l,:,:,0],axis=0),axis=1) - np.mean(np.mean(temp1[stg_vertices_l,:,:,3],axis=0),axis=1)
del temp1

# For calculating p-values
X = np.mean(X11[stg_vertices_l,:,:,:],axis=0)
###############################################################################
""" Timecourse: Lexical task """
task1 = 5
task2 = 8

plotit2(times, M, errM, task1, task2, yMin=0, yMax=2.3, subject = 'all: STG')
plotsig2(times,nReps,X, task1, task2, all_subject, boot_pVal)
os.chdir(figureDir)
plt.savefig('STG_lexical_all.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_lexical_all.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M1, errM1, task1, task2, yMin=0, yMax=2.7, subject = 'typical: STG')
plotsig2(times,nReps,X, task1, task2, good_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('STG_lexical_good.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_lexical_good.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M2, errM2, task1, task2, yMin=0, yMax=2.7, subject = 'struggling: STG')
plotsig2(times,nReps,X, task1, task2, poor_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('STG_lexical_poor.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_lexical_poor.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

""" Timecourse: Dot task """
task1 = 0
task2 = 3

plotit2(times, M, errM, task1, task2, yMin=0, yMax=2.3, subject = 'all: STG')
plotsig2(times,nReps,X, task1, task2, all_subject, boot_pVal)
os.chdir(figureDir)
plt.savefig('STG_dot_all.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_dot_all.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M1, errM1, task1, task2, yMin=0, yMax=2.7, subject = 'typical: STG')
plotsig2(times,nReps,X, task1, task2, good_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('STG_dot_good.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_dot_good.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M2, errM2, task1, task2, yMin=0, yMax=2.7, subject = 'struggling: STG')
plotsig2(times,nReps,X, task1, task2, poor_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('STG_dot_poor.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_dot_poor.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

""" Correlation: Lexical """
temp1 = X11[:,:,good_readers,:]
M1 = np.mean(temp1[stg_vertices_l,:,:,:],axis=0)
temp2 = X11[:,:,poor_readers,:]
M2 = np.mean(temp2[stg_vertices_l,:,:,:],axis=0)
del temp1, temp2

t_window1 = np.multiply(np.divide(np.add([400,500],[100,100]),1000.), sRate)
t_window1 = [np.int(i) for i in t_window1]

task = 5

lowNoise1_good = np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task], axis = 0) - np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task+3], axis = 0)
lowNoise1_good_err = np.std(lowNoise1_good) / np.sqrt(len(lowNoise1_good))


lowNoise1_poor = np.mean(M2[t_window1[0]:t_window1[1],:,task], axis = 0) - np.mean(M2[t_window1[0]:t_window1[1],:,task+3], axis = 0)
lowNoise1_poor_err = np.std(lowNoise1_poor) / np.sqrt(len(lowNoise1_poor))

temp_meg = np.concatenate((lowNoise1_good,lowNoise1_poor))
temp_read = np.concatenate((orig_twre[good_readers],orig_twre[poor_readers]))

plt.figure()
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_meg, temp_read, deg=1)
ax.plot(temp_meg, fit[0] * temp_meg + fit[1], color=[0,0,0])
ax.plot(temp_meg, temp_read, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
ax.plot(lowNoise1_good, orig_twre[good_readers], 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
os.chdir('figures')
plt.savefig('STG_corr_lexical.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_corr_lexical.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

np.corrcoef(temp_read,temp_meg)
r, p = stats.pearsonr(temp_read,temp_meg)
print('lexical(all): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[good_readers],lowNoise1_good)
r, p = stats.pearsonr(orig_twre[good_readers],lowNoise1_good)
print('lexical(good): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[poor_readers],lowNoise1_poor)
r, p = stats.pearsonr(orig_twre[poor_readers],lowNoise1_poor)
print('lexical(poor): correlation = %.4f, p = %.4f' %(r, p))

""" Correlation: Dot task """
t_window1_dot = np.multiply(np.divide(np.add([300,400],[100,100]),1000.), sRate)
t_window1_dot = [np.int(i) for i in t_window1_dot]

task = 0

lowNoise1_good = np.mean(M1[np.int(t_window1_dot[0]):np.int(t_window1_dot[1]),:,task], axis = 0) - np.mean(M1[np.int(t_window1_dot[0]):np.int(t_window1_dot[1]),:,task+3], axis = 0)
lowNoise1_good_err = np.std(lowNoise1_good) / np.sqrt(len(lowNoise1_good))

lowNoise1_poor = np.mean(M2[t_window1_dot[0]:t_window1_dot[1],:,task], axis = 0) - np.mean(M2[t_window1_dot[0]:t_window1_dot[1],:,task+3], axis = 0)
lowNoise1_poor_err = np.std(lowNoise1_poor) / np.sqrt(len(lowNoise1_poor))

temp_meg = np.concatenate((lowNoise1_good,lowNoise1_poor))
temp_read = np.concatenate((orig_twre[good_readers],orig_twre[poor_readers]))

plt.figure()
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_meg, temp_read, deg=1)
ax.plot(temp_meg, fit[0] * temp_meg + fit[1], color=[0,0,0])
ax.plot(temp_meg, temp_read, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
ax.plot(lowNoise1_good, orig_twre[good_readers], 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
os.chdir(figureDir)
plt.savefig('STG_corr_dot.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_corr_dot.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

np.corrcoef(temp_read,temp_meg)
r, p = stats.pearsonr(temp_read,temp_meg)
print('Dot(all): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[good_readers],lowNoise1_good)
r, p = stats.pearsonr(orig_twre[good_readers],lowNoise1_good)
print('Dot(good): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[poor_readers],lowNoise1_poor)
r, p = stats.pearsonr(orig_twre[poor_readers],lowNoise1_poor)
print('Dot(poor): correlation = %.4f, p = %.4f' %(r, p))

""" Task effects: Word response in lexical vs. dot task """
t0 = time.time()
task1 = 0
task2 = 5

temp2_good = np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task2], axis = 0)
temp2_poor = np.mean(M2[np.int(t_window1[0]):np.int(t_window1[1]),:,task2], axis = 0)

temp3_good = np.mean(M1[np.int(t_window1_dot[0]):np.int(t_window1_dot[1]),:,task1], axis = 0)
temp3_poor = np.mean(M2[np.int(t_window1_dot[0]):np.int(t_window1_dot[1]),:,task1], axis = 0)

temp2 = np.concatenate((temp2_good,temp2_poor)) # lexical
temp3 = np.concatenate((temp3_good,temp3_poor)) # dot

plt.figure()
plt.clf()
ax = plt.subplot()
fit = np.polyfit(temp3, temp2, deg=1)
ax.plot(temp3, fit[0] * temp3 + fit[1], color=[0,0,0])
ax.plot(temp3_poor, temp2_poor, 'o', markerfacecolor=c_table[5], markeredgecolor=[1,1,1], markersize=10)
ax.plot(temp3_good, temp2_good, 'o', markerfacecolor=c_table[3], markeredgecolor=[1,1,1], markersize=10)
plt.axis('square')
plt.ylim([0, 7])
plt.xlim([0, 7])
r, p = stats.pearsonr(temp3,temp2)
print('STG: lexical vs. dot task (all): correlation = %.4f, p = %.7f' %(r, p))
os.chdir(figureDir)
plt.savefig('STG_lexical_dot.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_lexical_dot.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
del M, M1, M2

#%%
""" Broca """
broca_vertices_l = np.load('IFG_Vert.npy')
temp1 = X11[:,:,all_subject,:]
M = np.mean(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))
del temp1

temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1

temp1 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp1

# For calculating p-values
X = np.mean(X11[broca_vertices_l,:,:,:],axis=0)
###############################################################################
""" Timecourse: Lexical task """
task1 = 5
task2 = 8

plotit2(times, M, errM, task1, task2, yMin=0, yMax=2.3, subject = 'all: IFG')
plotsig2(times,nReps,X, task1, task2, all_subject, boot_pVal)
os.chdir(figureDir)
plt.savefig('IFG_lexical_all.png',dpi=600,papertype='letter',format='png')
plt.savefig('IFG_lexical_all.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M1, errM1, task1, task2, yMin=0, yMax=2.7, subject = 'typical: IFG')
plotsig2(times,nReps,X, task1, task2, good_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('IFG_lexical_good.png',dpi=600,papertype='letter',format='png')
plt.savefig('IFG_lexical_good.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M2, errM2, task1, task2, yMin=0, yMax=2.7, subject = 'struggling: IFG')
plotsig2(times,nReps,X, task1, task2, poor_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('IFG_lexical_poor.png',dpi=600,papertype='letter',format='png')
plt.savefig('IFG_lexical_poor.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

""" Timecourse: Dot task """
task1 = 0
task2 = 3

plotit2(times, M, errM, task1, task2, yMin=0, yMax=2.3, subject = 'all: IFG')
plotsig2(times,nReps,X, task1, task2, all_subject, boot_pVal)
os.chdir(figureDir)
plt.savefig('IFG_dot_all.png',dpi=600,papertype='letter',format='png')
plt.savefig('IFG_dot_all.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M1, errM1, task1, task2, yMin=0, yMax=2.7, subject = 'typical: IFG')
plotsig2(times,nReps,X, task1, task2, good_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('IFG_dot_good.png',dpi=600,papertype='letter',format='png')
plt.savefig('IFG_dot_good.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M2, errM2, task1, task2, yMin=0, yMax=2.7, subject = 'struggling: IFG')
plotsig2(times,nReps,X, task1, task2, poor_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('IFG_dot_poor.png',dpi=600,papertype='letter',format='png')
plt.savefig('IFG_dot_poor.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

""" Correlation: Lexical """
temp1 = X11[:,:,good_readers,:]
M1 = np.mean(temp1[broca_vertices_l,:,:,:],axis=0)
temp2 = X11[:,:,poor_readers,:]
M2 = np.mean(temp2[broca_vertices_l,:,:,:],axis=0)
del temp1, temp2

t_window1 = np.multiply(np.divide(np.add([400,500],[100,100]),1000.), sRate)
t_window1 = [np.int(i) for i in t_window1]

task = 5

lowNoise1_good = np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task], axis = 0) - np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task+3], axis = 0)
lowNoise1_good_err = np.std(lowNoise1_good) / np.sqrt(len(lowNoise1_good))

lowNoise1_poor = np.mean(M2[t_window1[0]:t_window1[1],:,task], axis = 0) - np.mean(M2[t_window1[0]:t_window1[1],:,task+3], axis = 0)
lowNoise1_poor_err = np.std(lowNoise1_poor) / np.sqrt(len(lowNoise1_poor))

temp_meg = np.concatenate((lowNoise1_good,lowNoise1_poor))
temp_read = np.concatenate((orig_twre[good_readers],orig_twre[poor_readers]))

plt.figure()
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_meg, temp_read, deg=1)
ax.plot(temp_meg, fit[0] * temp_meg + fit[1], color=[0,0,0])
ax.plot(temp_meg, temp_read, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
ax.plot(lowNoise1_good, orig_twre[good_readers], 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
os.chdir('figures')
plt.savefig('IFG_corr_lexical.png',dpi=600,papertype='letter',format='png')
plt.savefig('IFG_corr_lexical.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

np.corrcoef(temp_read,temp_meg)
r, p = stats.pearsonr(temp_read,temp_meg)
print('lexical(all): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[good_readers],lowNoise1_good)
r, p = stats.pearsonr(orig_twre[good_readers],lowNoise1_good)
print('lexical(good): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[poor_readers],lowNoise1_poor)
r, p = stats.pearsonr(orig_twre[poor_readers],lowNoise1_poor)
print('lexical(poor): correlation = %.4f, p = %.4f' %(r, p))

""" Correlation: Dot task """
task = 0

lowNoise1_good = np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task], axis = 0) - np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task+3], axis = 0)
lowNoise1_good_err = np.std(lowNoise1_good) / np.sqrt(len(lowNoise1_good))

lowNoise1_poor = np.mean(M2[t_window1[0]:t_window1[1],:,task], axis = 0) - np.mean(M2[t_window1[0]:t_window1[1],:,task+3], axis = 0)
lowNoise1_poor_err = np.std(lowNoise1_poor) / np.sqrt(len(lowNoise1_poor))

temp_meg = np.concatenate((lowNoise1_good,lowNoise1_poor))
temp_read = np.concatenate((orig_twre[good_readers],orig_twre[poor_readers]))

plt.figure()
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_meg, temp_read, deg=1)
ax.plot(temp_meg, fit[0] * temp_meg + fit[1], color=[0,0,0])
ax.plot(temp_meg, temp_read, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
ax.plot(lowNoise1_good, orig_twre[good_readers], 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
os.chdir('figures')
plt.savefig('IFG_corr_lexical.png',dpi=600,papertype='letter',format='png')
plt.savefig('IFG_corr_lexical.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

np.corrcoef(temp_read,temp_meg)
r, p = stats.pearsonr(temp_read,temp_meg)
print('Dot(all): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[good_readers],lowNoise1_good)
r, p = stats.pearsonr(orig_twre[good_readers],lowNoise1_good)
print('Dot(good): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[poor_readers],lowNoise1_poor)
r, p = stats.pearsonr(orig_twre[poor_readers],lowNoise1_poor)
print('Dot(poor): correlation = %.4f, p = %.4f' %(r, p))

""" Task effects """
t0 = time.time()
task1 = 0
task2 = 5

temp1 = X11[:,:,good_readers,:]
MM = np.mean(temp1[broca_vertices_l,:,:,:],axis=0)
#task_effect_good = np.mean(MM[np.int(t_window1[0]):np.int(t_window1[1]),:,task2], axis = 0) - np.mean(MM[np.int(t_window1[0]):np.int(t_window1[1]),:,task1], axis = 0)
temp2_good = np.mean(MM[np.int(t_window1[0]):np.int(t_window1[1]),:,task2], axis = 0)
temp3_good = np.mean(MM[np.int(t_window1[0]):np.int(t_window1[1]),:,task1], axis = 0)
del temp1

temp1 = X11[:,:,poor_readers,:]
MM = np.mean(temp1[broca_vertices_l,:,:,:],axis=0)
#task_effect_poor = np.mean(MM[np.int(t_window1[0]):np.int(t_window1[1]),:,task2], axis = 0) - np.mean(MM[np.int(t_window1[0]):np.int(t_window1[1]),:,task1], axis = 0)
temp2_poor = np.mean(MM[np.int(t_window1[0]):np.int(t_window1[1]),:,task2], axis = 0)
temp3_poor = np.mean(MM[np.int(t_window1[0]):np.int(t_window1[1]),:,task1], axis = 0)
del temp1

temp2 = np.concatenate((temp2_good,temp2_poor)) # lexical
temp3 = np.concatenate((temp3_good,temp3_poor)) # dot

plt.figure()
plt.clf()
ax = plt.subplot()
fit = np.polyfit(temp3, temp2, deg=1)
ax.plot(temp3, fit[0] * temp3 + fit[1], color=[0,0,0])
ax.plot(temp3_poor, temp2_poor, 'o', markerfacecolor=[.5,.5,.5], markeredgecolor=[1,1,1], markersize=10)
ax.plot(temp3_good, temp2_good, 'o', markerfacecolor=[.5,.5,.5], markeredgecolor=[1,1,1], markersize=10)
plt.axis('square')
plt.ylim([0, 5])
plt.xlim([0, 5])
r, p = stats.pearsonr(temp3,temp2)
print('IFG: lexical vs. dot task (all): correlation = %.4f, p = %.5f' %(r, p))
os.chdir(figureDir)
plt.savefig('IFG_lexical_dot.png',dpi=600,papertype='letter',format='png')
plt.savefig('IFG_lexical_dot.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

#%%
""" TPJ """
tpj_vertices_l = np.load('TPJ_Vert.npy')

temp1 = X11[:,:,all_subject,:]
M = np.mean(np.mean(temp1[tpj_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(temp1[tpj_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))
del temp1

temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[tpj_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[tpj_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
diffM1 = np.mean(np.mean(temp1[tpj_vertices_l,:,:,5],axis=0),axis=1) - np.mean(np.mean(temp1[tpj_vertices_l,:,:,8],axis=0),axis=1)
diffM2 = np.mean(np.mean(temp1[tpj_vertices_l,:,:,0],axis=0),axis=1) - np.mean(np.mean(temp1[tpj_vertices_l,:,:,3],axis=0),axis=1)
del temp1

temp1 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp1[tpj_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp1[tpj_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
diffM3 = np.mean(np.mean(temp1[tpj_vertices_l,:,:,5],axis=0),axis=1) - np.mean(np.mean(temp1[tpj_vertices_l,:,:,8],axis=0),axis=1)
diffM4 = np.mean(np.mean(temp1[tpj_vertices_l,:,:,0],axis=0),axis=1) - np.mean(np.mean(temp1[tpj_vertices_l,:,:,3],axis=0),axis=1)
del temp1

# For calculating p-values
X = np.mean(X11[tpj_vertices_l,:,:,:],axis=0)
###############################################################################
""" Timecourse: Lexical task """
task1 = 5
task2 = 8

plotit2(times, M, errM, task1, task2, yMin=0, yMax=2.3, subject = 'all: TPJ')
plotsig2(times,nReps,X, task1, task2, all_subject, boot_pVal)
os.chdir(figureDir)
plt.savefig('TPJ_lexical_all.png',dpi=600,papertype='letter',format='png')
plt.savefig('TPJ_lexical_all.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M1, errM1, task1, task2, yMin=0, yMax=2.7, subject = 'typical: TPJ')
plotsig2(times,nReps,X, task1, task2, good_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('TPJ_lexical_good.png',dpi=600,papertype='letter',format='png')
plt.savefig('TPJ_lexical_good.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M2, errM2, task1, task2, yMin=0, yMax=2.7, subject = 'struggling: TPJ')
plotsig2(times,nReps,X, task1, task2, poor_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('TPJ_lexical_poor.png',dpi=600,papertype='letter',format='png')
plt.savefig('TPJ_lexical_poor.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

""" Timecourse: Dot task """
task1 = 0
task2 = 3

plotit2(times, M, errM, task1, task2, yMin=0, yMax=2.3, subject = 'all: TPJ')
plotsig2(times,nReps,X, task1, task2, all_subject, boot_pVal)
os.chdir(figureDir)
plt.savefig('TPJ_dot_all.png',dpi=600,papertype='letter',format='png')
plt.savefig('TPJ_dot_all.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M1, errM1, task1, task2, yMin=0, yMax=2.7, subject = 'typical: TPJ')
plotsig2(times,nReps,X, task1, task2, good_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('TPJ_dot_good.png',dpi=600,papertype='letter',format='png')
plt.savefig('TPJ_dot_good.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M2, errM2, task1, task2, yMin=0, yMax=2.7, subject = 'struggling: TPJ')
plotsig2(times,nReps,X, task1, task2, poor_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('TPJ_dot_poor.png',dpi=600,papertype='letter',format='png')
plt.savefig('TPJ_dot_poor.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

""" Correlation: Lexical """
temp1 = X11[:,:,good_readers,:]
M1 = np.mean(temp1[tpj_vertices_l,:,:,:],axis=0)
temp2 = X11[:,:,poor_readers,:]
M2 = np.mean(temp2[tpj_vertices_l,:,:,:],axis=0)
del temp1, temp2

t_window1 = np.multiply(np.divide(np.add([400,500],[100,100]),1000.), sRate)
t_window1 = [np.int(i) for i in t_window1]

task = 5

lowNoise1_good = np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task], axis = 0) - np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task+3], axis = 0)
lowNoise1_good_err = np.std(lowNoise1_good) / np.sqrt(len(lowNoise1_good))

lowNoise1_poor = np.mean(M2[t_window1[0]:t_window1[1],:,task], axis = 0) - np.mean(M2[t_window1[0]:t_window1[1],:,task+3], axis = 0)
lowNoise1_poor_err = np.std(lowNoise1_poor) / np.sqrt(len(lowNoise1_poor))

temp_meg = np.concatenate((lowNoise1_good,lowNoise1_poor))
temp_read = np.concatenate((orig_twre[good_readers],orig_twre[poor_readers]))

plt.figure()
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_meg, temp_read, deg=1)
ax.plot(temp_meg, fit[0] * temp_meg + fit[1], color=[0,0,0])
ax.plot(temp_meg, temp_read, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
ax.plot(lowNoise1_good, orig_twre[good_readers], 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
os.chdir('figures')
plt.savefig('TPJ_corr_lexical.png',dpi=600,papertype='letter',format='png')
plt.savefig('TPJ_corr_lexical.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

np.corrcoef(temp_read,temp_meg)
r, p = stats.pearsonr(temp_read,temp_meg)
print('lexical(all): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[good_readers],lowNoise1_good)
r, p = stats.pearsonr(orig_twre[good_readers],lowNoise1_good)
print('lexical(good): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[poor_readers],lowNoise1_poor)
r, p = stats.pearsonr(orig_twre[poor_readers],lowNoise1_poor)
print('lexical(poor): correlation = %.4f, p = %.4f' %(r, p))

""" Correlation: Dot task """
t_window1_dot = np.multiply(np.divide(np.add([300,400],[100,100]),1000.), sRate)
t_window1_dot = [np.int(i) for i in t_window1_dot]

task = 0

lowNoise1_good = np.mean(M1[np.int(t_window1_dot[0]):np.int(t_window1_dot[1]),:,task], axis = 0) - np.mean(M1[np.int(t_window1_dot[0]):np.int(t_window1_dot[1]),:,task+3], axis = 0)
lowNoise1_good_err = np.std(lowNoise1_good) / np.sqrt(len(lowNoise1_good))

lowNoise1_poor = np.mean(M2[t_window1_dot[0]:t_window1_dot[1],:,task], axis = 0) - np.mean(M2[t_window1_dot[0]:t_window1_dot[1],:,task+3], axis = 0)
lowNoise1_poor_err = np.std(lowNoise1_poor) / np.sqrt(len(lowNoise1_poor))

temp_meg = np.concatenate((lowNoise1_good,lowNoise1_poor))
temp_read = np.concatenate((orig_twre[good_readers],orig_twre[poor_readers]))

plt.figure()
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_meg, temp_read, deg=1)
ax.plot(temp_meg, fit[0] * temp_meg + fit[1], color=[0,0,0])
ax.plot(temp_meg, temp_read, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
ax.plot(lowNoise1_good, orig_twre[good_readers], 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
os.chdir('figures')
plt.savefig('TPJ_corr_dot.png',dpi=600,papertype='letter',format='png')
plt.savefig('TPJ_corr_dot.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

np.corrcoef(temp_read,temp_meg)
r, p = stats.pearsonr(temp_read,temp_meg)
print('Dot(all): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[good_readers],lowNoise1_good)
r, p = stats.pearsonr(orig_twre[good_readers],lowNoise1_good)
print('Dot(good): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[poor_readers],lowNoise1_poor)
r, p = stats.pearsonr(orig_twre[poor_readers],lowNoise1_poor)
print('Dot(poor): correlation = %.4f, p = %.4f' %(r, p))

""" Task effects: Word response in lexical vs. dot task """
t0 = time.time()
task1 = 0
task2 = 5

temp2_good = np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task2], axis = 0)
temp2_poor = np.mean(M2[np.int(t_window1[0]):np.int(t_window1[1]),:,task2], axis = 0)

temp3_good = np.mean(M1[np.int(t_window1_dot[0]):np.int(t_window1_dot[1]),:,task1], axis = 0)
temp3_poor = np.mean(M2[np.int(t_window1_dot[0]):np.int(t_window1_dot[1]),:,task1], axis = 0)

temp2 = np.concatenate((temp2_good,temp2_poor)) # lexical
temp3 = np.concatenate((temp3_good,temp3_poor)) # dot

plt.figure()
plt.clf()
ax = plt.subplot()
fit = np.polyfit(temp3, temp2, deg=1)
ax.plot(temp3, fit[0] * temp3 + fit[1], color=[0,0,0])
ax.plot(temp3_poor, temp2_poor, 'o', markerfacecolor=[.5,.5,.5], markeredgecolor=[1,1,1], markersize=10)
ax.plot(temp3_good, temp2_good, 'o', markerfacecolor=[.5,.5,.5], markeredgecolor=[1,1,1], markersize=10)
plt.axis('square')
plt.ylim([0, 7])
plt.xlim([0, 7])
r, p = stats.pearsonr(temp3,temp2)
print('TPJ: lexical vs. dot task (all): correlation = %.4f, p = %.7f' %(r, p))
os.chdir(figureDir)
plt.savefig('TPJ_lexical_dot.png',dpi=600,papertype='letter',format='png')
plt.savefig('TPJ_lexical_dot.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
del M, M1, M2

#%%
""" Motor """
M = np.mean(np.mean(tX11[motor_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(tX11[motor_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))

temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[motor_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[motor_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1

temp1 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp1[motor_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp1[motor_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp1

# For calculating p-values
X = np.mean(X11[motor_vertices_l,:,:,:],axis=0)

############################################################################### 
t0 = time.time()
plotit2(times, M, errM, 0, 3, yMin=0, yMax=2.3, subject = 'all')
plotsig2(times,nReps,X, 0, 3, all_subject, boot_pVal)
C = np.mean(X11[motor_vertices_l,:,:,0],axis=0) - np.mean(X11[motor_vertices_l,:,:,3],axis=0)

#corr = plotcorr3(times, C[:,all_subject], twre_index)
#plt.text(times[np.where(corr == np.max(corr))[0][0]],0.5,np.str(times[np.where(corr == np.max(corr))[0][0]]))
#plt.text(times[np.where(corr == np.max(corr))[0][0]],0.4,np.str(np.max(corr)))

os.chdir(figureDir)
plt.savefig('Motor_dot_all.png',dpi=600,papertype='letter',format='png')
plt.savefig('Motor_dot_all.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
#print "\n\nElasped time: %0.2d mins %0.2d secs" % (divmod(time.time()-t0, 60))

t0 = time.time()
plotit2(times, M1, errM1, 0, 3, yMin=0, yMax=2.7, subject = 'typical')
plotsig2(times,nReps,X, 0, 3, good_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('Motor_dot_good.png',dpi=600,papertype='letter',format='png')
plt.savefig('Motor_dot_good.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
#print "\n\nElasped time: %0.2d mins %0.2d secs" % (divmod(time.time()-t0, 60))

t0 = time.time()
plotit2(times, M2, errM2, 0, 3, yMin=0, yMax=2.7, subject = 'struggling')
plotsig2(times,nReps,X, 0, 3, poor_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('Motor_dot_poor.png',dpi=600,papertype='letter',format='png')
plt.savefig('Motor_dot_poor.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
#print "\n\nElasped time: %0.2d mins %0.2d secs" % (divmod(time.time()-t0, 60))
###############################################################################

t0 = time.time()
plotit2(times, M, errM, 5, 8, yMin=0, yMax=2.3, subject = 'all')
plotsig2(times,nReps,X, 5, 8, all_subject, boot_pVal)

#C = np.mean(X11[motor_vertices_l,:,:,5],axis=0) - np.mean(X11[motor_vertices_l,:,:,8],axis=0)
#corr2 = plotcorr3(times, C[:,all_subject], twre_index)
#plt.text(times[np.where(corr2 == np.max(corr2))[0][0]],0.5,np.str(times[np.where(corr2 == np.max(corr2))[0][0]]))
#plt.text(times[np.where(corr2 == np.max(corr2))[0][0]],0.4,np.str(np.max(corr2)))

os.chdir(figureDir)
plt.savefig('Motor_lexical_all.png',dpi=600,papertype='letter',format='png')
plt.savefig('Motor_lexical_all.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
#print "\n\nElasped time: %0.2d mins %0.2d secs" % (divmod(time.time()-t0, 60))

t0 = time.time()
plotit2(times, M1, errM1, 5, 8, yMin=0, yMax=2.7, subject = 'typical')
plotsig2(times,nReps,X, 5, 8, good_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('Motor_lexical_good.png',dpi=600,papertype='letter',format='png')
plt.savefig('Motor_lexical_good.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
#print "\n\nElasped time: %0.2d mins %0.2d secs" % (divmod(time.time()-t0, 60))

t0 = time.time()
plotit2(times, M2, errM2, 5, 8, yMin=0, yMax=2.7, subject = 'struggling')
plotsig2(times,nReps,X, 5, 8, poor_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('Motor_lexical_poor.png',dpi=600,papertype='letter',format='png')
plt.savefig('Motor_lexical_poor.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
#print "\n\nElasped time: %0.2d mins %0.2d secs" % (divmod(time.time()-t0, 60))


#%%
""" Sylvian """
M = np.mean(np.mean(tX11[sylvian_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(tX11[sylvian_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))

temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[sylvian_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[sylvian_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1

temp1 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp1[sylvian_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp1[sylvian_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp1

# For calculating p-values
X = np.mean(X11[sylvian_vertices_l,:,:,:],axis=0)

############################################################################### 
t0 = time.time()
plotit2(times, M, errM, 0, 3, yMin=0, yMax=2.7, subject = 'all')
plotsig2(times,nReps,X, 0, 3, all_subject, boot_pVal)

#C = np.mean(X11[sylvian_vertices_l,:,:,0],axis=0) - np.mean(X11[sylvian_vertices_l,:,:,3],axis=0)
#corr = plotcorr3(times, C[:,all_subject], twre_index)
#plt.text(times[np.where(corr == np.max(corr))[0][0]],0.5,np.str(times[np.where(corr == np.max(corr))[0][0]]))
#plt.text(times[np.where(corr == np.max(corr))[0][0]],0.4,np.str(np.max(corr)))

os.chdir(figureDir)
plt.savefig('Sylvian_dot_all.png',dpi=600,papertype='letter',format='png')
plt.savefig('Sylvian_dot_all.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
#print "\n\nElasped time: %0.2d mins %0.2d secs" % (divmod(time.time()-t0, 60))

t0 = time.time()
plotit2(times, M1, errM1, 0, 3, yMin=0, yMax=2.7, subject = 'typical')
plotsig2(times,nReps,X, 0, 3, good_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('Sylvian_dot_good.png',dpi=600,papertype='letter',format='png')
plt.savefig('Sylvian_dot_good.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
#print "\n\nElasped time: %0.2d mins %0.2d secs" % (divmod(time.time()-t0, 60))

t0 = time.time()
plotit2(times, M2, errM2, 0, 3, yMin=0, yMax=2.7, subject = 'struggling')
plotsig2(times,nReps,X, 0, 3, poor_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('Sylvian_dot_poor.png',dpi=600,papertype='letter',format='png')
plt.savefig('Sylvian_dot_poor.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
#print "\n\nElasped time: %0.2d mins %0.2d secs" % (divmod(time.time()-t0, 60))
###############################################################################

t0 = time.time()
plotit2(times, M, errM, 5, 8, yMin=0, yMax=2.7, subject = 'all')
plotsig2(times,nReps,X, 5, 8, all_subject, boot_pVal)

#C = np.mean(X11[sylvian_vertices_l,:,:,5],axis=0) - np.mean(X11[sylvian_vertices_l,:,:,8],axis=0)
#corr2 = plotcorr3(times, C[:,all_subject], twre_index)
#plt.text(times[np.where(corr2 == np.max(corr2))[0][0]],0.5,np.str(times[np.where(corr2 == np.max(corr2))[0][0]]))
#plt.text(times[np.where(corr2 == np.max(corr2))[0][0]],0.4,np.str(np.max(corr2)))

os.chdir(figureDir)
plt.savefig('Sylvian_lexical_all.png',dpi=600,papertype='letter',format='png')
plt.savefig('Sylvian_lexical_all.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
#print "\n\nElasped time: %0.2d mins %0.2d secs" % (divmod(time.time()-t0, 60))

t0 = time.time()
plotit2(times, M1, errM1, 5, 8, yMin=0, yMax=2.7, subject = 'typical')
plotsig2(times,nReps,X, 5, 8, good_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('Sylvian_lexical_good.png',dpi=600,papertype='letter',format='png')
plt.savefig('Sylvian_lexical_good.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
#print "\n\nElasped time: %0.2d mins %0.2d secs" % (divmod(time.time()-t0, 60))

t0 = time.time()
plotit2(times, M2, errM2, 5, 8, yMin=0, yMax=2.7, subject = 'struggling')
plotsig2(times,nReps,X, 5, 8, poor_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('Sylvian_lexical_poor.png',dpi=600,papertype='letter',format='png')
plt.savefig('Sylvian_lexical_poor.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
#print "\n\nElasped time: %0.2d mins %0.2d secs" % (divmod(time.time()-t0, 60))

#%%
""" Making bar plots """

t_window1 = np.multiply(np.divide(np.add([300,600],[100,100]),1000.), sRate)
t_window1 = [np.int(i) for i in t_window1] 
dot_window1 = np.multiply(np.divide(np.add([300,600],[100,100]),1000.), sRate)
dot_window1 = [np.int(i) for i in dot_window1]
t_window2 = np.multiply(np.divide(np.add([600,700],[100,100]),1000.), sRate)
t_window2 = [np.int(i) for i in t_window2] 

dot_early = np.multiply(np.divide(np.add([300,400],[100,100]),1000.), sRate)
dot_early = [np.int(i) for i in dot_early]
dot_late = np.multiply(np.divide(np.add([500,600],[100,100]),1000.), sRate)
dot_late = [np.int(i) for i in dot_late]

#temp_vertices = broca_vertices_l
temp_vertices = stg_vertices_l
# AUD 1
# Lexical task
task = 5
temp1 = X11[:,:,good_readers,:]
M1 = np.mean(temp1[temp_vertices,:,:,:],axis=0)
lowNoise1_good = np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task], axis = 0) - np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task+3], axis = 0)
lowNoise1_good_err = np.std(lowNoise1_good) / np.sqrt(len(lowNoise1_good))
medNoise1_good = np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task+1], axis = 0) - np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task+3], axis = 0)
medNoise1_good_err = np.std(medNoise1_good) / np.sqrt(len(medNoise1_good))

del temp1

temp2 = X11[:,:,poor_readers,:]
M2 = np.mean(temp2[temp_vertices,:,:,:],axis=0)
lowNoise1_poor = np.mean(M2[t_window1[0]:t_window1[1],:,task], axis = 0) - np.mean(M2[t_window1[0]:t_window1[1],:,task+3], axis = 0)
lowNoise1_poor_err = np.std(lowNoise1_poor) / np.sqrt(len(lowNoise1_poor))
medNoise1_poor = np.mean(M2[t_window1[0]:t_window1[1],:,task+1], axis = 0) - np.mean(M2[t_window1[0]:t_window1[1],:,task+3], axis = 0)
medNoise1_poor_err = np.std(medNoise1_poor) / np.sqrt(len(medNoise1_poor))

#lowNoise2_poor = np.mean(M2[t_window2[0]:t_window2[1],:,task], axis = 0) - np.mean(M2[t_window2[0]:t_window2[1],:,task+3], axis = 0)
#lowNoise2_poor_err = np.std(lowNoise2_poor) / np.sqrt(len(lowNoise2_poor))
#medNoise2_poor = np.mean(M2[t_window2[0]:t_window2[1],:,task+1], axis = 0) - np.mean(M2[t_window2[0]:t_window2[1],:,task+3], axis = 0)
#medNoise2_poor_err = np.std(medNoise2_poor) / np.sqrt(len(medNoise2_poor))
del temp2

# Dot task
task = 0
temp1 = X11[:,:,good_readers,:]
M1 = np.mean(temp1[temp_vertices,:,:,:],axis=0)
dot_lowNoise1_good = np.mean(M1[dot_window1[0]:dot_window1[1],:,task], axis = 0) - np.mean(M1[dot_window1[0]:dot_window1[1],:,task+3], axis = 0)
dot_lowNoise1_good_err = np.std(dot_lowNoise1_good) / np.sqrt(len(dot_lowNoise1_good))
dot_medNoise1_good = np.mean(M1[dot_window1[0]:dot_window1[1],:,task+1], axis = 0) - np.mean(M1[dot_window1[0]:dot_window1[1],:,task+3], axis = 0)
dot_medNoise1_good_err = np.std(dot_medNoise1_good) / np.sqrt(len(dot_medNoise1_good))

dot_lowNoise2_early_good = np.mean(M1[dot_early[0]:dot_early[1],:,task], axis = 0) - np.mean(M1[dot_early[0]:dot_early[1],:,task+3], axis = 0)
dot_lowNoise2_early_good_err = np.std(dot_lowNoise2_early_good) / np.sqrt(len(dot_lowNoise2_early_good))
dot_lowNoise2_late_good = np.mean(M1[dot_late[0]:dot_late[1],:,task], axis = 0) - np.mean(M1[dot_late[0]:dot_late[1],:,task+3], axis = 0)
dot_lowNoise2_late_good_err = np.std(dot_lowNoise2_late_good) / np.sqrt(len(dot_lowNoise2_late_good))

del temp1

temp2 = X11[:,:,poor_readers,:]
M2 = np.mean(temp2[temp_vertices,:,:,:],axis=0)
dot_lowNoise1_poor = np.mean(M2[dot_window1[0]:dot_window1[1],:,task], axis = 0) - np.mean(M2[dot_window1[0]:dot_window1[1],:,task+3], axis = 0)
dot_lowNoise1_poor_err = np.std(dot_lowNoise1_poor) / np.sqrt(len(dot_lowNoise1_poor))
dot_medNoise1_poor = np.mean(M2[dot_window1[0]:dot_window1[1],:,task+1], axis = 0) - np.mean(M2[dot_window1[0]:dot_window1[1],:,task+3], axis = 0)
dot_medNoise1_poor_err = np.std(dot_medNoise1_poor) / np.sqrt(len(dot_medNoise1_poor))

dot_lowNoise2_early_poor = np.mean(M2[dot_early[0]:dot_early[1],:,task], axis = 0) - np.mean(M2[dot_early[0]:dot_early[1],:,task+3], axis = 0)
dot_lowNoise2_early_poor_err = np.std(dot_lowNoise2_early_poor) / np.sqrt(len(dot_lowNoise2_early_poor))
dot_lowNoise2_late_poor = np.mean(M2[dot_late[0]:dot_late[1],:,task], axis = 0) - np.mean(M2[dot_late[0]:dot_late[1],:,task+3], axis = 0)
dot_lowNoise2_late_poor_err = np.std(dot_lowNoise2_late_poor) / np.sqrt(len(dot_lowNoise2_late_poor))

del temp2

"""
Correlation
"""

aaa = np.array(subs)
temp_meg1 = np.concatenate((dot_lowNoise1_good,dot_lowNoise1_poor))
temp_read = np.concatenate((orig_twre[good_readers],orig_twre[poor_readers]))
temp_brs = np.concatenate((brs[good_readers],brs[poor_readers]))
temp_rf = np.concatenate((rf[good_readers],rf[poor_readers]))
temp_raw = np.concatenate((orig_swe[good_readers],orig_swe[poor_readers]))

temp_age = np.concatenate((orig_age[good_readers],orig_age[poor_readers]))

#temp_read = temp_raw
#temp_id = np.where(temp_meg>4.5)[0]
#temp_meg = np.concatenate((temp_meg[0:temp_id], temp_meg[temp_id+1:len(temp_meg)]))
#temp_read = np.concatenate((temp_read[0:temp_id], temp_read[temp_id+1:len(temp_read)]))

plt.figure(20)
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_meg1, temp_read, deg=1)
ax.plot(temp_meg1, fit[0] * temp_meg1 + fit[1], color=[0,0,0])
#fit = np.polyfit(dot_lowNoise1_good, orig_twre[good_readers], deg=1)
#ax.plot(dot_lowNoise1_good, fit[0] * dot_lowNoise1_good + fit[1], color=c_table[5])
ax.plot(temp_meg1, temp_read, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
#ax.plot(dot_lowNoise1_good, orig_twre[good_readers], 'o', markerfacecolor=c_table[5], markeredgecolor=[1,1,1], markersize=10)

#plt.xlim([-1,4])
#for i, txt in enumerate(temp_age):
#    ax.annotate(temp_age[i], (temp_meg1[i], temp_read[i]))
#plt.ylim([-1,6])
#plt.xlim([50,130])
np.corrcoef(temp_read,temp_meg1)
r, p = stats.pearsonr(temp_read,temp_meg1)
print('dot(all): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[good_readers],dot_lowNoise1_good)
r, p = stats.pearsonr(orig_twre[good_readers],dot_lowNoise1_good)
print('dot(good): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[poor_readers],dot_lowNoise1_poor)
r, p = stats.pearsonr(orig_twre[poor_readers],dot_lowNoise1_poor)
print('dot(poor): correlation = %.4f, p = %.4f' %(r, p))
os.chdir('figures')
plt.savefig('STG_corr_dot.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_corr_dot.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

""" Lexical task """
temp_meg2 = np.concatenate((lowNoise1_good,lowNoise1_poor))
plt.figure(21)
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_meg2, temp_read, deg=1)
ax.plot(temp_meg2, fit[0] * temp_meg2 + fit[1], color=[0,0,0])
#fit = np.polyfit(lowNoise1_good, orig_twre[good_readers], deg=1)
#ax.plot(lowNoise1_good, fit[0] * lowNoise1_good + fit[1], color=c_table[5])
ax.plot(temp_meg2, temp_read, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
ax.plot(lowNoise1_good, orig_twre[good_readers], 'o', markerfacecolor=c_table[5], markeredgecolor=[1,1,1], markersize=10)
#plt.xlim([-1,4])

np.corrcoef(temp_read,temp_meg2)
r, p = stats.pearsonr(temp_read,temp_meg2)
print('lexical(all): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[good_readers],lowNoise1_good)
r, p = stats.pearsonr(orig_twre[good_readers],lowNoise1_good)
print('lexical(good): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[poor_readers],lowNoise1_poor)
r, p = stats.pearsonr(orig_twre[poor_readers],lowNoise1_poor)
print('lexical(poor): correlation = %.4f, p = %.4f' %(r, p))
os.chdir('figures')
plt.savefig('STG_corr_lexical.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_corr_lexical.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

""" Dot vs. Lexical """
plt.figure(22)
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_meg2, temp_meg1, deg=1)
ax.plot(temp_meg2, fit[0] * temp_meg2 + fit[1], color=[0,0,0])
#fit = np.polyfit(lowNoise1_good, orig_twre[good_readers], deg=1)
#ax.plot(lowNoise1_good, fit[0] * lowNoise1_good + fit[1], color=c_table[5])
ax.plot(temp_meg2, temp_meg1, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
#ax.plot(lowNoise1_good, orig_twre[good_readers], 'o', markerfacecolor=c_table[5], markeredgecolor=[1,1,1], markersize=10)
#plt.xlim([-1,4])
#plt.ylim([-1,4])
np.corrcoef(temp_meg1,temp_meg2)
r, p = stats.pearsonr(temp_meg1,temp_meg2)
print('Dot_Lexical: correlation = %.4f, p = %.4f' %(r, p))
#np.corrcoef(orig_twre[good_readers],lowNoise1_good)
#r, p = stats.pearsonr(orig_twre[good_readers],lowNoise1_good)
#print('lexical(good): correlation = %.4f, p = %.4f' %(r, p))
#np.corrcoef(orig_twre[poor_readers],lowNoise1_poor)
#r, p = stats.pearsonr(orig_twre[poor_readers],lowNoise1_poor)
#print('lexical(poor): correlation = %.4f, p = %.4f' %(r, p))
os.chdir('figures')
plt.savefig('STG_corr_dot_lexical.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_corr_dot_lexical.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')


temp_meg3 = np.concatenate((dot_lowNoise2_early_good,dot_lowNoise2_early_poor))
plt.figure(23)
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_meg3, temp_read, deg=1)
ax.plot(temp_meg3, fit[0] * temp_meg3 + fit[1], color=[0,0,0])

ax.plot(temp_meg3, temp_read, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)

np.corrcoef(temp_read,temp_meg3)
r, p = stats.pearsonr(temp_read,temp_meg3)
print('dot(all)_early: correlation = %.4f, p = %.4f' %(r, p))

os.chdir('figures')
plt.savefig('STG_corr_dot_early.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_corr_dot_early.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

temp_meg4 = np.concatenate((dot_lowNoise2_late_good,dot_lowNoise2_late_poor))
plt.figure(23)
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_meg4, temp_read, deg=1)
ax.plot(temp_meg4, fit[0] * temp_meg4 + fit[1], color=[0,0,0])

ax.plot(temp_meg4, temp_read, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)

np.corrcoef(temp_read,temp_meg4)
r, p = stats.pearsonr(temp_read,temp_meg4)
print('dot(all)_late: correlation = %.4f, p = %.4f' %(r, p))

os.chdir('figures')
plt.savefig('STG_corr_dot_late.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_corr_dot_late.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plt.figure(24)
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_meg4, temp_meg3, deg=1)
ax.plot(temp_meg4, fit[0] * temp_meg4 + fit[1], color=[0,0,0])

ax.plot(temp_meg4, temp_meg3, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)

np.corrcoef(temp_meg3,temp_meg4)
r, p = stats.pearsonr(temp_meg3,temp_meg4)
print('dot(all)_late: correlation = %.4f, p = %.4f' %(r, p))

os.chdir('figures')
plt.savefig('STG_corr_dot_early_late.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_corr_dot_early_late.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plt.figure(25)
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_age, temp_meg1, deg=1)
ax.plot(temp_age, fit[0] * temp_age + fit[1], color=[0,0,0])

ax.plot(temp_age, temp_meg1, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)

np.corrcoef(temp_meg1,temp_age)
r, p = stats.pearsonr(temp_meg1,temp_age)
print('dot(all)_age: correlation = %.4f, p = %.4f' %(r, p))

os.chdir('figures')
plt.savefig('STG_corr_dot_age.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_corr_dot_age.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plt.figure(26)
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_age, temp_meg2, deg=1)
ax.plot(temp_age, fit[0] * temp_age + fit[1], color=[0,0,0])

ax.plot(temp_age, temp_meg2, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)

np.corrcoef(temp_meg2,temp_age)
r, p = stats.pearsonr(temp_meg2,temp_age)
print('lexical(all)_age: correlation = %.4f, p = %.4f' %(r, p))

os.chdir('figures')
plt.savefig('STG_corr_lexical_age.png',dpi=600,papertype='letter',format='png')
plt.savefig('STG_corr_lexical_age.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

#%%
""" Right STG: Word vs. Noise """
stg_vertices_r = np.load('STG_Vert_r.npy')
temp1 = X11[:,:,all_subject,:]
M = np.mean(np.mean(temp1[stg_vertices_r,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(temp1[stg_vertices_r,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))
del temp1

temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[stg_vertices_r,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[stg_vertices_r,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
diffM1 = np.mean(np.mean(temp1[stg_vertices_r,:,:,5],axis=0),axis=1) - np.mean(np.mean(temp1[stg_vertices_r,:,:,8],axis=0),axis=1)
diffM2 = np.mean(np.mean(temp1[stg_vertices_r,:,:,0],axis=0),axis=1) - np.mean(np.mean(temp1[stg_vertices_r,:,:,3],axis=0),axis=1)
del temp1

temp1 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp1[stg_vertices_r,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp1[stg_vertices_r,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
diffM3 = np.mean(np.mean(temp1[stg_vertices_r,:,:,5],axis=0),axis=1) - np.mean(np.mean(temp1[stg_vertices_r,:,:,8],axis=0),axis=1)
diffM4 = np.mean(np.mean(temp1[stg_vertices_r,:,:,0],axis=0),axis=1) - np.mean(np.mean(temp1[stg_vertices_r,:,:,3],axis=0),axis=1)
del temp1

# For calculating p-values
X = np.mean(X11[stg_vertices_r,:,:,:],axis=0)
###############################################################################
""" Timecourse: Lexical task """
task1 = 5
task2 = 8

plotit2(times, M, errM, task1, task2, yMin=0, yMax=2.3, subject = 'all: STG')
plotsig2(times,nReps,X, task1, task2, all_subject, boot_pVal)
os.chdir(figureDir)
plt.savefig('STGr_lexical_all.png',dpi=600,papertype='letter',format='png')
plt.savefig('STGr_lexical_all.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M1, errM1, task1, task2, yMin=0, yMax=2.7, subject = 'typical: STG')
plotsig2(times,nReps,X, task1, task2, good_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('STGr_lexical_good.png',dpi=600,papertype='letter',format='png')
plt.savefig('STGr_lexical_good.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M2, errM2, task1, task2, yMin=0, yMax=2.7, subject = 'struggling: STG')
plotsig2(times,nReps,X, task1, task2, poor_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('STGr_lexical_poor.png',dpi=600,papertype='letter',format='png')
plt.savefig('STGr_lexical_poor.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

""" Timecourse: Dot task """
task1 = 0
task2 = 3

plotit2(times, M, errM, task1, task2, yMin=0, yMax=2.3, subject = 'all: STG')
plotsig2(times,nReps,X, task1, task2, all_subject, boot_pVal)
os.chdir(figureDir)
plt.savefig('STGr_dot_all.png',dpi=600,papertype='letter',format='png')
plt.savefig('STGr_dot_all.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M1, errM1, task1, task2, yMin=0, yMax=2.7, subject = 'typical: STG')
plotsig2(times,nReps,X, task1, task2, good_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('STGr_dot_good.png',dpi=600,papertype='letter',format='png')
plt.savefig('STGr_dot_good.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

plotit2(times, M2, errM2, task1, task2, yMin=0, yMax=2.7, subject = 'struggling: STG')
plotsig2(times,nReps,X, task1, task2, poor_readers, boot_pVal)
os.chdir(figureDir)
plt.savefig('STGr_dot_poor.png',dpi=600,papertype='letter',format='png')
plt.savefig('STGr_dot_poor.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

""" Correlation: Lexical """
temp1 = X11[:,:,good_readers,:]
M1 = np.mean(temp1[stg_vertices_r,:,:,:],axis=0)
temp2 = X11[:,:,poor_readers,:]
M2 = np.mean(temp2[stg_vertices_r,:,:,:],axis=0)
del temp1, temp2

t_window1 = np.multiply(np.divide(np.add([400,500],[100,100]),1000.), sRate)
t_window1 = [np.int(i) for i in t_window1]

task = 5

lowNoise1_good = np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task], axis = 0) - np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task+3], axis = 0)
lowNoise1_good_err = np.std(lowNoise1_good) / np.sqrt(len(lowNoise1_good))


lowNoise1_poor = np.mean(M2[t_window1[0]:t_window1[1],:,task], axis = 0) - np.mean(M2[t_window1[0]:t_window1[1],:,task+3], axis = 0)
lowNoise1_poor_err = np.std(lowNoise1_poor) / np.sqrt(len(lowNoise1_poor))

temp_meg = np.concatenate((lowNoise1_good,lowNoise1_poor))
temp_read = np.concatenate((orig_twre[good_readers],orig_twre[poor_readers]))

plt.figure()
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_meg, temp_read, deg=1)
ax.plot(temp_meg, fit[0] * temp_meg + fit[1], color=[0,0,0])
ax.plot(temp_meg, temp_read, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
ax.plot(lowNoise1_good, orig_twre[good_readers], 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
os.chdir('figures')
plt.savefig('STGr_corr_lexical.png',dpi=600,papertype='letter',format='png')
plt.savefig('STGr_corr_lexical.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

np.corrcoef(temp_read,temp_meg)
r, p = stats.pearsonr(temp_read,temp_meg)
print('lexical(all): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[good_readers],lowNoise1_good)
r, p = stats.pearsonr(orig_twre[good_readers],lowNoise1_good)
print('lexical(good): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[poor_readers],lowNoise1_poor)
r, p = stats.pearsonr(orig_twre[poor_readers],lowNoise1_poor)
print('lexical(poor): correlation = %.4f, p = %.4f' %(r, p))

""" Correlation: Dot task """
t_window1_dot = np.multiply(np.divide(np.add([300,400],[100,100]),1000.), sRate)
t_window1_dot = [np.int(i) for i in t_window1_dot]

task = 0

lowNoise1_good = np.mean(M1[np.int(t_window1_dot[0]):np.int(t_window1_dot[1]),:,task], axis = 0) - np.mean(M1[np.int(t_window1_dot[0]):np.int(t_window1_dot[1]),:,task+3], axis = 0)
lowNoise1_good_err = np.std(lowNoise1_good) / np.sqrt(len(lowNoise1_good))

lowNoise1_poor = np.mean(M2[t_window1_dot[0]:t_window1_dot[1],:,task], axis = 0) - np.mean(M2[t_window1_dot[0]:t_window1_dot[1],:,task+3], axis = 0)
lowNoise1_poor_err = np.std(lowNoise1_poor) / np.sqrt(len(lowNoise1_poor))

temp_meg = np.concatenate((lowNoise1_good,lowNoise1_poor))
temp_read = np.concatenate((orig_twre[good_readers],orig_twre[poor_readers]))

plt.figure()
plt.clf()
ax = plt.subplot()

fit = np.polyfit(temp_meg, temp_read, deg=1)
ax.plot(temp_meg, fit[0] * temp_meg + fit[1], color=[0,0,0])
ax.plot(temp_meg, temp_read, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
ax.plot(lowNoise1_good, orig_twre[good_readers], 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)
os.chdir('figures')
plt.savefig('STGr_corr_dot.png',dpi=600,papertype='letter',format='png')
plt.savefig('STGr_corr_dot.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')

np.corrcoef(temp_read,temp_meg)
r, p = stats.pearsonr(temp_read,temp_meg)
print('Dot(all): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[good_readers],lowNoise1_good)
r, p = stats.pearsonr(orig_twre[good_readers],lowNoise1_good)
print('Dot(good): correlation = %.4f, p = %.4f' %(r, p))
np.corrcoef(orig_twre[poor_readers],lowNoise1_poor)
r, p = stats.pearsonr(orig_twre[poor_readers],lowNoise1_poor)
print('Dot(poor): correlation = %.4f, p = %.4f' %(r, p))

""" Task effects: Word response in lexical vs. dot task """
t0 = time.time()
task1 = 0
task2 = 5

temp2_good = np.mean(M1[np.int(t_window1[0]):np.int(t_window1[1]),:,task2], axis = 0)
temp2_poor = np.mean(M2[np.int(t_window1[0]):np.int(t_window1[1]),:,task2], axis = 0)

temp3_good = np.mean(M1[np.int(t_window1_dot[0]):np.int(t_window1_dot[1]),:,task1], axis = 0)
temp3_poor = np.mean(M2[np.int(t_window1_dot[0]):np.int(t_window1_dot[1]),:,task1], axis = 0)

temp2 = np.concatenate((temp2_good,temp2_poor)) # lexical
temp3 = np.concatenate((temp3_good,temp3_poor)) # dot

plt.figure()
plt.clf()
ax = plt.subplot()
fit = np.polyfit(temp3, temp2, deg=1)
ax.plot(temp3, fit[0] * temp3 + fit[1], color=[0,0,0])
ax.plot(temp3_poor, temp2_poor, 'o', markerfacecolor=c_table[5], markeredgecolor=[1,1,1], markersize=10)
ax.plot(temp3_good, temp2_good, 'o', markerfacecolor=c_table[3], markeredgecolor=[1,1,1], markersize=10)
plt.axis('square')
plt.ylim([0, 7])
plt.xlim([0, 7])
r, p = stats.pearsonr(temp3,temp2)
print('STG: lexical vs. dot task (all): correlation = %.4f, p = %.7f' %(r, p))
os.chdir(figureDir)
plt.savefig('STGr_lexical_dot.png',dpi=600,papertype='letter',format='png')
plt.savefig('STGr_lexical_dot.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')
del M, M1, M2

#%%
D = {'STG_Dot': temp_meg1,
     'STG_Lex': temp_meg2,
     'Age': temp_age,
     'TWRE': temp_read,
     'BRS': temp_brs,
     'RF': temp_rf,
     }
df = DataFrame(D,columns=['STG_Dot','STG_Lex','Age','TWRE','BRS','RF'])

X = df[['Age','TWRE']]
Y = df['STG_Dot']

X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)

X = df[['Age','BRS']]
Y = df['STG_Dot']

X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)

X = df[['Age','RF']]
Y = df['STG_Dot']

X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)
