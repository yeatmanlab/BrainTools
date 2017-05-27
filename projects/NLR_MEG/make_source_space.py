#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 09:16:49 2017

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

raw_dir = '/mnt/scratch/NLR_MEG3'

os.chdir(raw_dir)

subs = ['NLR_102_RS','NLR_103_AC','NLR_105_BB','NLR_110_HH','NLR_127_AM',
        'NLR_130_RW','NLR_132_WP','NLR_133_ML','NLR_145_AC','NLR_150_MG','NLR_151_RD',
        'NLR_152_TC','NLR_160_EK','NLR_161_AK','NLR_162_EF','NLR_163_LF','NLR_164_SF',
        'NLR_170_GM','NLR_172_TH','NLR_174_HS','NLR_179_GM','NLR_180_ZD','NLR_187_NB',
        'NLR_201_GS','NLR_202_DD','NLR_203_AM','NLR_204_AM','NLR_205_AC','NLR_206_LM',
        'NLR_207_AH','NLR_210_SB','NLR_211_LB'
        ]



for n, s in enumerate(subs):    
    subject = s
                       
    # Create source space
    os.chdir(os.path.join(fs_dir,subject,'bem'))

#    if s == '205_ac151208' or s == '205_ac160202':
    """ NLR_205: Head is too small to create ico5 """
    spacing='oct6' # 8196 = 4098 * 2
    fn2 = subject + '-' + 'oct-6' + '-src.fif'
#    else:
#        spacing='ico5' # 10242 * 2
#        fn2 = subject + '-' + 'ico-5' + '-src.fif'

    src = mne.setup_source_space(subject=subject, spacing=spacing, # source spacing = 5 mm
                                 subjects_dir=fs_dir, add_dist=False, n_jobs=18, overwrite=True)
    src = mne.add_source_space_distances(src, dist_limit=np.inf, n_jobs=18, verbose=None)
    mne.write_source_spaces(fn2, src, overwrite=True)
