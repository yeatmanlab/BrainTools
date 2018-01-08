#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:46:56 2017

@author: sjjoo
"""

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

raw_dir = '/mnt/scratch/NLR_MEG4'

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

os.chdir(os.path.join(raw_dir,session1[0]))
os.chdir('lists')

plt.figure(500)
plt.clf()

all_index = []
for n in np.arange(0,6):
    fn = 'ALL_' + session1[0] + '_' + str(n+1) + '-eve.lst'
    mat = []
    for line in open(fn).readlines():
        mat.append(line.split())
    
    time_point = [float(row[0])/1200 for row in mat]
    
    time_point = np.subtract(time_point, time_point[0])
    
    event_id = [row[2] for row in mat]
    
    for ii in np.arange(0,8):
        if np.mod(n,2) == 0:
            index = ([i for i,x in enumerate(event_id) if x == str(101+ii)])
        else:
             index = [i for i,x in enumerate(event_id) if x == str(201+ii)]
        all_index.append(index)            
    
    plt.subplot(6,1,n+1)
    plt.plot(time_point, event_id,'o-')