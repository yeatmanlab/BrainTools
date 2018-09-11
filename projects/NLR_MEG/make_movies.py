#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:07:26 2017

@author: sjjoo
"""
from pyface.qt import QtGui, QtCore

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

conditions1 = ['word_c254_p20_dot', 'word_c254_p50_dot', 'word_c137_p20_dot',
               'word_c254_p80_dot', 'word_c137_p80_dot', 'bigram_c254_p20_dot',
               'bigram_c254_p50_dot', 'bigram_c137_p20_dot',
               'word_c254_p20_word', 'word_c254_p50_word', 'word_c137_p20_word',
               'word_c254_p80_word', 'word_c137_p80_word', 'bigram_c254_p20_word',
               'bigram_c254_p50_word', 'bigram_c137_p20_word'
               ]
conditions2 = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]

for n,s in enumerate(subs):
    os.chdir(os.path.join(raw_dir,session1[n]))
    os.chdir('inverse')
    for iCond, s in enumerate(conditions2):    
        fname_lh = conditions1[s] + '_loose_morph-lh.stc'
        temp3_1 = mne.read_source_estimate(fname_lh, subject='fsaverage')
        
        brain3_1 = temp3_1.plot(hemi='lh', subjects_dir=fs_dir, views = ['lat','ven','med'], #transparent = True,
              initial_time=0.15, clim=dict(kind='value', lims=[0, 5, 10]))#, background='white', colorbar=False) #np.max(temp3.data[:,:])])) #pos_lims=[0, 4, 4.5] #np.max(temp3.data[:,:])]))
    #    brain3_2 = temp3_1.plot(hemi='rh', subjects_dir=fs_dir, views = ['lat','ven','med'], #transparent = True,
    #          initial_time=0.10, clim=dict(kind='value', lims=[0, 5, 13]))
    #    brain3_1.save_image('LH_'+s+'_100.png')
        brain3_1.save_movie('LH_loose.mp4',time_dilation = 4.0,framerate = 30)