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

subs = ['NLR_102_RS','NLR_105_BB','NLR_110_HH','NLR_127_AM',
        'NLR_132_WP','NLR_145_AC','NLR_150_MG',
        'NLR_152_TC','NLR_160_EK','NLR_161_AK','NLR_162_EF','NLR_163_LF',
        'NLR_164_SF','NLR_170_GM','NLR_172_TH','NLR_174_HS','NLR_179_GM',
        'NLR_180_ZD','NLR_201_GS','NLR_203_AM',
        'NLR_204_AM','NLR_205_AC','NLR_207_AH','NLR_210_SB','NLR_211_LB',
        'NLR_GB310','NLR_KB218','NLR_GB267','NLR_JB420',
        'NLR_HB275','NLR_GB355'] 

session2 = ['102_rs160815','105_bb161011','110_hh160809','127_am161004',
       '132_wp161122','145_ac160823','150_mg160825',
       '152_tc160623','160_ek160915','161_ak160916','162_ef160829','163_lf160920',
       '164_sf160920','170_gm160822','172_th160825','174_hs160829','179_gm160913',
       '180_zd160826','201_gs150925','203_am151029',
       '204_am151120','205_ac160202','207_ah160809','210_sb160822','211_lb160823',
       'nlr_gb310170829','nlr_kb218170829','nlr_gb267170911','nlr_jb420170828',
       'nlr_hb275170828','nlr_gb355170907']

#%%
for n, s in enumerate(session2):
    os.chdir(os.path.join(raw_dir,session2[n]))
    
    os.chdir('inverse')
    fn = 'All_40-sss_eq_'+session2[n]+'-ave.fif'
    evoked = mne.read_evokeds(fn, condition=0, 
                              baseline=(None,0), kind='average', proj=True)
    
    info = evoked.info 
    
    os.chdir('../forward')
    fn = session2[n] + '-sss-fwd.fif'
    fwd = mne.read_forward_solution(fn,force_fixed=False,surf_ori=True)
    
    #Inverse here
    os.chdir('../covariance')
    fn = session2[n] + '-40-sss-cov.fif'
    cov = mne.read_cov(fn)
    
    os.chdir('../inverse')
    # Free: loose = 1; Loose: loose = 0.2
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, loose=0.2, depth=0.8, use_cps=True)
    
    fn = session2[n] + '-depth8-inv.fif'
    mne.minimum_norm.write_inverse_operator(fn,inv)