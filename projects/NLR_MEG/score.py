# -*- coding: utf-8 -*-

"""Scores trials of VWR experiment as a function of available acq runs"""

# Authors: Ross Maddox <rkmaddox@uw.edu>
#          Kambiz Tavabi <ktavabi@gmail.com>
#
# License: BSD (3-clause)


import os
from os import path as op
from mnefun._mnefun import safe_inserter, get_raw_fnames
import mne


def score(p, subjects, run_indices):
    # run_indices = [r for ri, r in enumerate(run_indices) if ri in p.subject_indices]
    for subj, ri in zip(subjects, run_indices):
        if ri is None:
            runs = p.run_names
        else:
            runs = ['%s_%d' % (subj, r+1) for r in ri]
        print('Running subject %s...' % subj)
        # Figure out what our filenames should be
        out_dir = op.join(p.work_dir, subj, p.list_dir)
        if not op.isdir(out_dir):
            os.mkdir(out_dir)
        raw_names = get_raw_fnames(p, subj, 'raw', True, False, ri)
        # assert len(raw_names) == 8
        out_dir = op.join(p.work_dir, subj, p.list_dir)
        if not op.isdir(out_dir):
            os.mkdir(out_dir)
        for rii, fname in zip(runs, raw_names):
            print('  Run %s...' % op.basename(fname))
            
            # sjjoo_20160810: To incorporate session that has incomplete data
#            if not op.isfile(fname):
#                raise RuntimeError(' File %s not found' % fname)
            if op.isfile(fname):
            # read events, make sure we got the correct number of trials
                raw = mne.io.Raw(fname, allow_maxshield=True)
                ev = mne.find_events(raw, stim_channel='STI101', shortest_event=1)
                if op.basename(fname).__contains__('_1_raw'):
                    ev[:, 2] += 100
                elif op.basename(fname).__contains__('_2_raw'):
                    ev[:, 2] += 200
                elif op.basename(fname).__contains__('_3_raw'):
                    ev[:, 2] += 100
                elif op.basename(fname).__contains__('_4_raw'):
                    ev[:, 2] += 200
                elif op.basename(fname).__contains__('_5_raw'):
                    ev[:, 2] += 100
                elif op.basename(fname).__contains__('_6_raw'):
                    ev[:, 2] += 200
                elif op.basename(fname).__contains__('_7_raw'):
                    ev[:, 2] += 100
                elif op.basename(fname).__contains__('_8_raw'):
                    ev[:, 2] += 200
                fname_out = 'ALL_' + safe_inserter(rii,
                                                   subj) + '-eve.lst'
                fname_out = op.join(out_dir, fname_out)
                mne.write_events(fname_out, ev)
