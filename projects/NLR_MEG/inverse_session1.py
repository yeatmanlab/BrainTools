#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:50:06 2017

@author: sjjoo
"""

import sys
import mne
import matplotlib.pyplot as plt
from mne.utils import run_subprocess, logger
import os
from os import path as op
import copy
import shutil
import numpy as np
from numpy.random import randn
from scipy import stats as stats
import time
from functools import partial

from mne import set_config
set_config('MNE_MEMMAP_MIN_SIZE', '1M')
set_config('MNE_CACHE_DIR', '.tmp')

mne.set_config('MNE_USE_CUDA', 'true')

this_env = copy.copy(os.environ)
#fs_dir = '/mnt/diskArray/projects/freesurfer'
fs_dir = '/mnt/diskArray/projects/avg_fsurfer'

this_env['SUBJECTS_DIR'] = fs_dir
#this_env['FREESURFER_HOME'] = '/usr/local/freesurfer'

raw_dir = '/mnt/scratch/NLR_MEG4'

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

#%%
for n, s in enumerate(session1):
    os.chdir(os.path.join(raw_dir,session1[n]))
    
    os.chdir('inverse')
    fn = 'All_40-sss_eq_'+session1[n]+'-ave.fif'
    evoked = mne.read_evokeds(fn, condition=0, 
                              baseline=(None,0), kind='average', proj=True)
    
    info = evoked.info 
    
    os.chdir('../forward')
    fn = session1[n] + '-sss-fwd.fif'
    fwd = mne.read_forward_solution(fn,force_fixed=False,surf_ori=True)
    
    #Inverse here
    os.chdir('../covariance')
    fn = session1[n] + '-40-sss-cov.fif'
    cov = mne.read_cov(fn)
    
    os.chdir('../inverse')
    # Free: loose = 1; Loose: loose = 0.2
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, loose=0.2, depth=0.8, use_cps=True)
    
    fn = session1[n] + '-depth8-inv.fif'
    mne.minimum_norm.write_inverse_operator(fn,inv)