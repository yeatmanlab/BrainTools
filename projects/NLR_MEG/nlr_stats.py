 # -*- coding: utf-8 -*-

# Author: Kambiz Tavabi <ktavabi@gmail.com>
#

"""Docsting

"""

import matplotlib
matplotlib.use('Agg')
import os
from os import path as op
import numpy as np
from functools import partial
import time
import mne
from mne import set_log_level as log
from mne.stats import ttest_1samp_no_p
from mne.minimum_norm import (read_inverse_operator, apply_inverse)
import mnefun
from mnefun import anova_time
from mnefun import get_fsaverage_medial_vertices

__copyright__ = "Copyright 2015, ILABS"
__status__ = "Development"

log(verbose='Warning')
# cd to meg directory
os.chdir('/media/ALAYA/data/ilabs/nlr/')
work_dir = os.getcwd()

# set up mnefun parameters of interest
p = mnefun.Params(lp_cut=40.)
p.analyses = ['Words_noise']
p.subjects = ['nlr01', 'nlr02', 'nlr04', 'nlr05', 'nlr06', 'nlr07', 'nlr08']
p.structurals = ['nlr01', 'nlr02', 'nlr04', 'nlr05', 'nlr06', 'nlr07', 'nlr08']

do_plots = False
reload_data = True
do_contrasts = True
do_anova = False

# Local variables
lambda2 = 1. / 9.
n_smooth = 15
fs_verts = [np.arange(10242), np.arange(10242)]
fs_medial = get_fsaverage_medial_vertices()
inv_type = 'meg-fixed'  # can be meg-eeg, meg-fixed, meg, eeg-fixed, or eeg
fname_data = op.join(work_dir, '%s_data.npz' % p.analyses[0])
sigma = 1e-3
n_jobs = 18
tmin, tmax = 0.15, 0.20  # time window for RM-ANOVA

conditions = ['epochs_word_c254_p20_010', 'epochs_word_c254_p50_010', 'epochs_word_c137_p20_010', 'epochs_word_c141_p20_010',
              'epochs_noise_010',
              'epochs_word_c254_p20_020', 'epochs_word_c254_p50_020', 'epochs_word_c137_p20_020', 'epochs_word_c141_p20_020',
              'epochs_noise_020']  # events of interest
contrasts = [[4, 0],
             [9, 5],
             [2, 0],
             [7, 5],
             [0, 5],
             [1, 6],
             [2, 7],
             [3, 8]]  # contrasts of interest

# Plot (butterfly & topographic) conditions averaged over subjects in sensor domain.
if do_plots:
    if not op.exists(work_dir + '/figures'):
        os.mkdir(work_dir + '/figures')
    for c in conditions:
        evo = []
        for subj in p.subjects:
            ev_name = 'evoked_%s' % c
            evoked_file = op.join(work_dir, subj, 'inverse',
                                  '%s_%d-sss_eq_%s-ave.fif' % (p.analyses[0], p.lp_cut, subj))
            evo.append(mne.read_evokeds(evoked_file, condition=c, baseline=(None, 0),
                                        kind='average', proj=True))
            evo_grand_ave = np.sum(evo)

        h0 = evo_grand_ave.plot_topomap(times=np.arange(0, evo_grand_ave.times[-1], 0.1))
        h0.savefig(op.join(work_dir, 'figures', ev_name + '_topomap'), dpi=96, format='png')
        h1 = evo_grand_ave.plot()
        h1.savefig(op.join(work_dir, 'figures', ev_name + '_butterfly'), dpi=96, format='png')

######################################
#  Do source imaging and handle data.#
######################################
if reload_data:
    naves = np.zeros(len(p.subjects), int)
    for si, (subj, struc) in enumerate(zip(p.subjects, p.structurals)):
        print('Loading data for subject %s...' % subj)
        inv_dir = op.join(work_dir, subj, 'inverse')

        # load the inverse
        inv = op.join(inv_dir, '%s-%d-sss-%s-inv.fif' % (subj, p.lp_cut, inv_type))
        inv = read_inverse_operator(inv)
        fname = op.join(inv_dir, '%s_%d-sss_eq_%s-ave.fif'
                        % (p.analyses[0], p.lp_cut, subj))
        aves = [mne.Evoked(fname, cond, baseline=(None, 0), proj=True,
                                kind='average') for cond in conditions]
        nave = np.unique([a.nave for a in aves])
        assert len(nave) == 1
        for ave, cond in zip(aves, conditions):
                assert ave.comment == cond
        naves[si] = nave[0]

        # apply inverse, bin, morph
        stcs = [apply_inverse(ave, inv, lambda2, 'dSPM') for ave in aves]
        stcs = [stc.bin(0.005) for stc in stcs]
        m = mne.compute_morph_matrix(struc, 'fsaverage', stcs[0].vertices,
                                     fs_verts, n_smooth)
        stcs = [stc.morph_precomputed('fsaverage', fs_verts, m)
                for stc in stcs]

        # put in big matrix
        if subj == p.subjects[0]:
            data = np.empty((len(stcs), len(p.subjects), stcs[0].shape[0],
                             stcs[0].shape[1]))
        for di, stc in enumerate(stcs):
            data[di, si, :, :] = stc.data
            times = stc.times
        print('Writing data...')
        np.savez_compressed(fname_data, data=data, times=times, naves=naves)
else:
    print('Loading saved data...')
    data = np.load(fname_data)
    data, times, naves = data['data'], data['times'], data['naves']

# 1-sample t-test in time (uncorrected) on source data for given contrasts and save results as source time course.
for s in range(len(p.subjects)):
    for cont in contrasts:
        contrast = '-'.join([conditions[c] for c in cont[::-1]])
        X = data[:, s, :, :]
        X = (np.abs(X[cont[1]]) - np.abs(X[cont[0]]))
        stc = mne.SourceEstimate(X, fs_verts, times[0],
                                 np.diff(times[:2]))
        stc.save(op.join(p.work_dir, 'stcs', 'nlr%.0f_contrast_%s' % (s + 1, contrast)))

if do_contrasts:
    if not op.exists(work_dir + '/stcs'):
        os.mkdir(work_dir + '/stcs')
    stat_fun = partial(ttest_1samp_no_p, sigma=sigma)
    for cont in contrasts:
        contrast = '-'.join([conditions[c] for c in cont[::-1]])
        print('  Running t-tests for %s' % contrast)
        X = (np.abs(data[cont[1]]) - np.abs(data[cont[0]]))
        stc = mne.SourceEstimate(stat_fun(X), fs_verts, times[0],
                                 np.diff(times[:2]))
        stc.save(op.join(work_dir, 'stcs', 'contrast_%s' % contrast))

# Compute spatiotemporal RM-ANOVA (one-way) and visualize results on surface
if do_anova:
    for cont in contrasts:
        t0 = time.time()
        contrast = '-'.join([conditions[c] for c in cont[::-1]])
        tt = '%s-%s' % (tmin, tmax)
        print('  Running spatiotemporal RM-ANOVA for %s in the interval %s ms' % (contrast, tt))
        mask = np.logical_and(times >= tmin, times <= tmax)
        X = np.swapaxes(np.swapaxes(data[cont], 0, 1), 2, 3)[:, :, mask, :]
        X = np.reshape(X, (X.shape[0], 2 * X.shape[2], X.shape[3]))
        X = np.abs(X)
        tvals, pvals, dof = anova_time(X)
        d = np.sign(tvals) * -np.log10(np.minimum(np.abs(pvals), 1))  # -np.log10(np.minimum(np.abs(p) * 20484, 1) bonferroni correction
        d[fs_medial] = 0
        stc_anova = mne.SourceEstimate(d, fs_verts, 0, 1e-3, 'fsaverage')
        stc_anova.save(op.join(work_dir, 'stcs', 'anova_%s_%s' % (contrast, tt)))
        fmin, fmid, fmax = 2, 4, 6
        colormap = mne.viz.mne_analyze_colormap(limits=[fmin, fmid, fmax])
        brain = stc_anova.plot(hemi='split', colormap=colormap, time_label=None,
                               smoothing_steps=n_smooth, transparent=True, config_opts={},
                               views=['lat', 'med'])
        brain.save_image(op.join(work_dir, 'stcs', 'anova_%s_%s.png'
                                 % (contrast, tt)))
        print('    Time: %s' % round((time.time() - t0) / 60., 2))
        brain.close()
