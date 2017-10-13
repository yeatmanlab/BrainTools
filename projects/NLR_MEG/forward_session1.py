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
#this_env['FREESURFER_HOME'] = '/usr/local/freesurfer'

raw_dir = '/mnt/scratch/NLR_MEG4'

os.chdir(raw_dir)

subs = ['NLR_102_RS','NLR_103_AC','NLR_110_HH','NLR_127_AM',
        'NLR_130_RW','NLR_132_WP','NLR_133_ML','NLR_145_AC','NLR_151_RD',
        'NLR_152_TC','NLR_160_EK','NLR_161_AK','NLR_163_LF','NLR_164_SF',
        'NLR_170_GM','NLR_172_TH','NLR_174_HS','NLR_179_GM','NLR_180_ZD',
        'NLR_187_NB','NLR_203_AM','NLR_204_AM','NLR_205_AC','NLR_206_LM',
        'NLR_207_AH','NLR_211_LB','NLR_150_MG',
        'NLR_GB310','NLR_KB218','NLR_JB423','NLR_GB267','NLR_JB420',
        'NLR_HB275','NLR_197_BK','NLR_GB355','NLR_GB387',
        'NLR_HB205','NLR_IB319','NLR_JB227','NLR_JB486','NLR_KB396']

#for n, s in enumerate(subs):
#    run_subprocess(['mne', 'watershed_bem', '--subject', subs[n],
#                 '--overwrite'], env=this_env)
    
#    mne.bem.make_watershed_bem(subject = subs[28],subjects_dir=fs_dir,overwrite=True,
#                               preflood=20, show=True)
"""USE above code
mri_watershed -h 3 -useSRAS -surf /mnt/diskArray/projects/avg_fsurfer/NLR_205_AC/bem/watershed/NLR_205_AC /mnt/diskArray/projects/avg_fsurfer/NLR_205_AC/mri/T1.mgz /mnt/diskArray/projects/avg_fsurfer/NLR_205_AC/bem/watershed/ws
"""

"""
Run head_surf.m
"""

# Let's take a look...
#for n, s in enumerate(subs):
#    mne.viz.plot_bem(subject=subs[n],subjects_dir=fs_dir,brain_surfaces='white', orientation='coronal')

#for n, s in enumerate(subs):
##    os.chdir(os.path.join(fs_dir,subs[n],'bem'))
#    run_subprocess(['mne', 'make_scalp_surfaces', '--subject', subs[n],
#                    '--overwrite','--no-decimate']) # Disable medium and sparse decimations (dense only)
#                                                    # otherwise, it gives errors

""" Co-register...
mne.gui.coregistration(tabbed=False,subject=subs[0],subjects_dir=fs_dir)
# Recommended way is to use mne coreg from terminal
"""

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
            '207_ah160608','211_lb160617','150_mg160606',
            'nlr_gb310170614','nlr_kb218170619','nlr_jb423170620','nlr_gb267170620',
            'nlr_jb420170621','nlr_hb275170622','197_bk170622','nlr_gb355170606','nlr_gb387170608',
            'nlr_hb205170825','nlr_ib319170825','nlr_jb227170811','nlr_jb486170803','nlr_kb396170808'] 
#subs = ['NLR_205_AC','NLR_206_LM',
#        'NLR_207_AH','NLR_210_SB','NLR_211_LB'
#        ]
#session1 = ['205_ac151208','205_ac160202',
#            '206_lm151119',
#            '206_lm160113','207_ah160608','207_ah160809','210_sb160822','211_lb160617','211_lb160823'
#            ]

#n_subjects = len(subs)
"""
Forward model...
"""
#sourceFlag = np.ones((n_subjects,1))

#%%
#for n, s in enumerate(session1):
#    os.chdir(os.path.join(raw_dir,session1[n]))
#    
#    if s[0:3] == 'nlr':
#        subject = s[0:9].upper()
#    else:
#        subject = 'NLR_' + s[0:6].upper()
#    
#    os.chdir('inverse')
#    fn = 'All_40-sss_eq_'+session1[n]+'-ave.fif'
#    evoked = mne.read_evokeds(fn, condition=0, 
#                              baseline=(None,0), kind='average', proj=True)
#    
#    info = evoked.info 
#    
#    if os.path.isdir('../forward'):
#        os.chdir('../forward')
##    else:
##        temp_src = '/mnt/scratch/NLR_MEG2/' + session1[n] + '/forward'
##        temp_dest = '/mnt/scratch/NLR_MEG3/' + session1[n] + '/forward'
##        shutil.copytree(temp_src, temp_dest)
#    trans = session1[n] + '-trans.fif'
##    Take a look at the sensors
#    mne.viz.plot_trans(info, trans, subject=subs[n], dig=True,
#                       meg_sensors=True, subjects_dir=fs_dir)

#%%
for n, s in enumerate(session1):
    os.chdir(os.path.join(raw_dir,session1[n]))
    
    if s[0:3] == 'nlr':
        subject = s[0:9].upper()
    else:
        subject = 'NLR_' + s[0:6].upper()
    
    os.chdir('inverse')
    fn = 'All_40-sss_eq_'+session1[n]+'-ave.fif'
    evoked = mne.read_evokeds(fn, condition=0, 
                              baseline=(None,0), kind='average', proj=True)
    
    info = evoked.info 
    
    if os.path.isdir('../forward'):
        os.chdir('../forward')
    else:
        temp_src = '/mnt/scratch/NLR_MEG2/' + session1[n] + '/forward'
        temp_dest = '/mnt/scratch/NLR_MEG3/' + session1[n] + '/forward'
        shutil.copytree(temp_src, temp_dest)
    trans = session1[n] + '-trans.fif'
    
    # Take a look at the sensors
#    mne.viz.plot_trans(info, trans, subject=subs[n], dig=True,
#                       meg_sensors=True, subjects_dir=fs_dir)
                       
    ### Read source space
#    spacing='oct6' #'ico5' # 10242 * 2
    fn2 = subject + '-' + 'ico-5' + '-src.fif' # ico-5
    if s == '205_ac151208' or s == '205_ac160202' or s == 'nlr_jb227170811': # NLR_205 has too small head for ico-5
        fn2 = subject + '-' + 'oct-6' + '-src.fif'

    os.chdir(os.path.join(fs_dir,subject,'bem'))
    src = mne.read_source_spaces(fn2)
    os.chdir(os.path.join(raw_dir,session1[n]))
    os.chdir('forward')
        
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
    model = mne.make_bem_model(subject=subject, ico=5, # 5=20484, 4=5120
                               conductivity=conductivity, 
                               subjects_dir=fs_dir)
    bem = mne.make_bem_solution(model)
    fn = session1[n] + '-bem-sol.fif'
    mne.write_bem_solution(fn,bem)
    
    # Now create forward model
    fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem,
                                    fname=None, meg=True, eeg=False,
                                    mindist=3.0, n_jobs=18)
    fn = session1[n] + '-fwd.fif'
    mne.write_forward_solution(fn,fwd,overwrite=True)
    
    
    #Inverse here
    
    os.chdir('../covariance')
    fn = session1[n] + '-40-sss-cov.fif'
    cov = mne.read_cov(fn)
    
    os.chdir('../inverse')
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, loose=0.2, depth=0.8)
    
    fn = session1[n] + '-inv-ico5.fif'
    mne.minimum_norm.write_inverse_operator(fn,inv)
