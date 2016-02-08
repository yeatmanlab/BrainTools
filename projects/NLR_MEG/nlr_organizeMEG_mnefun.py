# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:26:17 2016

@author: jyeatman
"""
import os
import shutil
import glob

# define subjects and sessions
raw_dir = '/mnt/diskArray/projects/MEG/nlr/mnetest'
sub = ['202_dd', '203_am']
sess = [['150827', '150919', '151013', '151103'], ['150831', '150922', '151009', '151029']]
adir = '/mnt/diskArray/projects/MEG/nlr/mnetest/mneanalysis'
subjects = []
subject_run_indices = []
for n, s in enumerate(sub):
    for ss in sess[n]:
        # Make a new directory for the subject-session combo
        sdir = os.path.join(adir,s+ss)
        rfifdir = os.path.join(sdir,'raw_fif')
        subjects.append(sdir)
        if not os.path.isdir(sdir):
            os.mkdir(sdir)
        if not os.path.isdir(rfifdir):
            os.mkdir(rfifdir)
        # List all directory files
        fiflist=glob.glob(os.path.join(raw_dir,s,ss,'*_raw.fif'))
        fiflist.sort()
        for fnum, fname in enumerate(fiflist):
            fifdest = os.path.join(rfifdir,s+ss+fiflist[fnum][-10:])
            if not os.path.isfile(fifdest):
                shutil.copy(os.path.join(raw_dir,s,ss,fiflist[0]), fifdest)
            else:
                print(fifdest + ' already exists: Skpping')
                    
    



