# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:05:40 2016

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

raw_dir = '/mnt/scratch/NLR_MEG3'

os.chdir(raw_dir)

subs = ['NLR_102_RS','NLR_103_AC','NLR_110_HH','NLR_145_AC','NLR_150_MG',
        'NLR_151_RD','NLR_152_TC',
        'NLR_160_EK','NLR_161_AK','NLR_163_LF','NLR_164_SF','NLR_170_GM','NLR_172_TH',
        'NLR_174_HS','NLR_179_GM','NLR_180_ZD','NLR_207_AH','NLR_211_LB',
        'NLR_201_GS','NLR_203_AM','NLR_204_AM','NLR_206_LM','NLR_105_BB',
        'NLR_127_AM']

#for n, s in enumerate(subs):
#    run_subprocess(['mne', 'watershed_bem', '--subject', subs[n],
#                 '--overwrite'], env=this_env)
##    mne.bem.make_watershed_bem(subject,fs_dir,overwrite=True)
#
## Task a look...
#for n, s in enumerate(subs):
#    mne.viz.plot_bem(subject=subs[n],subjects_dir=fs_dir)

""" Co-register...
mne.gui.coregistration(tabbed=False,subject=subject,subjects_dir=fs_dir)
# Recommended way is to use mne coreg from terminal
"""

for n in np.arange(0,len(subs)):
    run_subprocess(['mne', 'make_scalp_surfaces', '--subject', subs[n]])
    
# Session 1
# subs are synced up with session1 folder names...
#
session1 = ['102_rs160618','103_ac150609','110_hh160608','145_ac160621','150_mg160606',
 '151_rd160620','152_tc160422',
 '160_ek160627','161_ak160627','163_lf160707','164_sf160707','170_gm160613',
 '172_th160614','174_hs160620',
 '179_gm160701','180_zd160621','207_ah160608',
 '211_lb160617','201_gs150729',
 '203_am150831',
 '204_am150829','206_lm151119','105_bb161011','127_am161004'] 

n_subjects = len(subs)
"""
Forward model...
"""
for n, s in enumerate(session1[0:n_subjects]):
    os.chdir(os.path.join(raw_dir,session1[n]))
    
    os.chdir('inverse')
    fn = 'All_40-sss_eq_'+session1[n]+'-ave.fif'
    evoked = mne.read_evokeds(fn, condition=0, 
                              baseline=(None,0), kind='average', proj=True)
    
    info = evoked.info
    
    os.chdir('../forward')
    trans = session1[n] + '-trans.fif'
    
    # Take a look at the sensors
#    mne.viz.plot_trans(info, trans, subject=subs[n], dig=True,
#                       meg_sensors=True, subjects_dir=fs_dir)
                       
    # Create source space 
    src = mne.setup_source_space(subject=subs[n],spacing='oct6', # source spacing = 5 mm
                                 subjects_dir=fs_dir,add_dist=False,overwrite=True)
    
    mne.add_source_space_distances(src, dist_limit=np.inf, n_jobs=18, verbose=None)
    
    #import numpy as np  # noqa
    #from mayavi import mlab  # noqa
    #from surfer import Brain  # noqa
    #
    #brain = Brain('sample', 'lh', 'inflated', subjects_dir=subjects_dir)
    #surf = brain._geo
    #
    #vertidx = np.where(src[0]['inuse'])[0]
    #
    #mlab.points3d(surf.x[vertidx], surf.y[vertidx],
    #              surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)
    
    # Create BEM model
    conductivity = (0.3,)  # for single layer
    #conductivity = (0.3, 0.006, 0.3)  # for three layers
    model = mne.make_bem_model(subject=subs[n], ico=4,
                               conductivity=conductivity, 
                               subjects_dir=fs_dir)
    bem = mne.make_bem_solution(model)
    fn = session1[n] + '-bem-sol.fif'
    mne.write_bem_solution(fn,bem)
    
    # Now create forward model
    fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem,
                                    fname=None, meg=True, eeg=False,
                                    mindist=5.0, n_jobs=4)
    fn = session1[n] + '-fwd.fif'
    mne.write_forward_solution(fn,fwd,overwrite=True)
    
    
    #Inverse here
    
    os.chdir('../covariance')
    fn = session1[n] + '-40-sss-cov.fif'
    cov = mne.read_cov(fn)
    
    os.chdir('../inverse')
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, loose=0.2, depth=0.8)
    
    fn = session1[n] + '-inv.fif'
    mne.minimum_norm.write_inverse_operator(fn,inv)
