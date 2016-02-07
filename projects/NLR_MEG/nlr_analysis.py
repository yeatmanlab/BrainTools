# -*- coding: utf-8 -*-

"""Docstring"""

# Authors: Ross Maddox <rkmaddox@uw.edu>
#          Kambiz Tavabi <ktavabi@gmail.com>
#
# License: BSD (3-clause)

import mnefun
import numpy as np

from score import score


params = mnefun.Params(tmin=-0.05, tmax=0.5, t_adjust=-39e-3, n_jobs=18,
                       decim=2, n_jobs_mkl=1, proj_sfreq=250,
                       n_jobs_fir='cuda', n_jobs_resample='cuda',
                       filter_length='5s', epochs_type='fif', lp_cut=40.,
                       bmin=-0.05, auto_bad=10., plot_raw=False)

params.subjects = ['nlr01', 'nlr02', 'nlr04', 'nlr05', 'nlr06', 'nlr07', 'nlr08']
params.structurals = ['nlr01', 'nlr02', 'nlr04', 'nlr05', 'nlr06', 'nlr07', 'nlr08']

assert len(set(params.subjects)) == len(params.subjects)
assert len(set(params.structurals)) == len(params.structurals)

"""Shit list
nlr03 blink artifacts
"""
params.dates = [(2014, 0, 00)] * len(params.subjects)
params.subject_indices =  [0] # np.setdiff1d(np.arange(len(params.subjects)), [])  # np.arange(5, 7)
params.score = score  # scoring function to use~
params.plot_drop_logs = False

params.acq_ssh = 'kambiz@minea'  # minea - 172.28.161.8
params.acq_dir = '/sinuhe/data02/jason_words'
params.sws_ssh = 'kam@kasga'  # kasga - 172.28.161.8
params.sws_dir = '/data03/kam/'

# epoch rejection criterion
params.reject = dict(grad=3500e-13, mag=4.0e-12, eog=150e-6)
params.ssp_eog_reject = dict(grad=3500e-13, mag=4.0e-12)
params.ssp_ecg_reject = dict(grad=3500e-13, mag=4.0e-12)
params.auto_bad_reject = dict(grad=3500e-13, mag=4.0e-12)
params.auto_bad_flat = dict(grad=1e-13, mag=1e-15)
params.run_names = ['%s_1', '%s_2', '%s_3', '%s_4', '%s_5', '%s_6', '%s_7', '%s_8']
params.bem_type = '5120'
params.cov_method = 'shrunk'
params.get_projs_from = range(len(params.run_names))
params.inv_names = ['%s']
params.inv_runs = [range(0, len(params.run_names))]
params.runs_empty = ['%s_erm']
params.proj_nums = [[1, 1, 0],  # ECG: grad/mag/eeg
                    [2, 2, 0],  # EOG
                    [2, 2, 0]]  # Continuous (from ERM)

# The scoring function needs to produce an event file with these values
params.in_names = ['word_c254_p20_010_', 'word_c254_p50_010_', 'word_c137_p20_010_', 'word_c141_p20_010_',
                   'noise_010_',
                   'bigram_c254_p20_010_', 'bigram_c254_p50_010_', 'bigram_c137_p20_010_', 'bigram_c141_p20_010_',
                   'word_c254_p20_020_', 'word_c254_p50_020_', 'word_c137_p20_020_', 'word_c141_p20_020_',
                   'noise_020_',
                   'bigram_c254_p20_020_', 'bigram_c254_p50_020_', 'bigram_c137_p20_020_', 'bigram_c141_p20_020']

params.in_numbers = [101, 102, 103, 104,
                     105,
                     106, 107, 108, 109,
                     201, 202, 203, 204,
                     205,
                     206, 207, 208, 209]

# These lines define how to translate the above event types into evoked files
params.analyses = [
    'All',
    'All_words',
    'All_bigrams',
    'Words_noise',
    'Word_noise',
    'Bigrams_noise',
    'Odd_words',
    'Odd_noise',
    'Odd_bigrams',
    'Odd_word_ave',
    'Odd_bigram_ave',
    'Eve_words',
    'Eve_noise',
    'Eve_bigrams',
    'Eve_word_ave',
    'Eve_bigram_ave']

params.out_names = [
    ['All'],
    ['word_c254_p20_01', 'word_c254_p50_01', 'word_c137_p20_01', 'word_c141_p20_01',
     'word_c254_p20_02', 'word_c254_p50_02', 'word_c137_p20_02', 'word_c141_p20_02'],
    ['bigram_c254_p20_01', 'bigram_c254_p50_01', 'bigram_c137_p20_01', 'bigram_c141_p20_01',
     'bigram_c254_p20_02', 'bigram_c254_p50_02', 'bigram_c137_p20_02', 'bigram_c141_p20_02'],
    ['word_c254_p20_010', 'word_c254_p50_010', 'word_c137_p20_010', 'word_c141_p20_010',
     'noise_010',
     'word_c254_p20_020', 'word_c254_p50_020', 'word_c137_p20_020', 'word_c141_p20_020',
     'noise_020'],
    ['word_cp', 'noise'],
    ['noise_010',
     'bigram_c254_p20_010', 'bigram_c254_p50_010', 'bigram_c137_p20_010', 'bigram_c141_p20_010',
     'noise_020',
     'bigram_c254_p20_020', 'bigram_c254_p50_020', 'bigram_c137_p20_020', 'bigram_c141_p20_020'],
    ['word_c254_p20_01', 'word_c254_p50_01', 'word_c137_p20_01', 'word_c141_p20_01'],
    ['noise_01'],
    ['bigram_c254_p20_01', 'bigram_c254_p50_01', 'bigram_c137_p20_01', 'bigram_c141_p20_01'],
    ['word_cp_01'],
    ['bigram_cp_01'],
    ['word_c254_p20_02', 'word_c254_p50_02', 'word_c137_p20_02', 'word_c141_p20_02'],
    ['noise_02'],
    ['bigram_c254_p20_02', 'bigram_c254_p50_02', 'bigram_c137_p20_02', 'bigram_c141_p20_02'],
    ['word_cp_02'],
    ['bigram_cp_02']]

params.out_numbers = [
    [1] * len(params.in_numbers),  # Combine all trials
    [1, 2, 3, 4,
     -1,
     -1, -1, -1, -1,
     5, 6, 7, 8,
     -1,
     -1, -1, -1, -1],
    [-1, -1, -1, -1,
     -1,
     1, 2, 3, 4,
     -1, -1, -1, -1,
     -1,
     5, 6, 7, 8],
    [1, 2, 3, 4,
     5,
     -1, -1, -1, -1,
     6, 7, 8, 9,
     10,
     -1, -1, -1, -1],
    [1, 1, 1, 1,
     2,
     -1, -1, -1, -1,
     1, 1, 1, 1,
     2,
     -1, -1, -1, -1],
    [-1, -1, -1, -1,
     1,
     2, 3, 4, 5,
     -1, -1, -1, -1,
     6,
     7, 8, 9, 10],
    [1, 2, 3, 4,
     -1,
     -1, -1, -1, -1,
     -1, -1, -1, -1,
     -1,
     -1, -1, -1, -1],
    [-1, -1, -1, -1,
     1,
     -1, -1, -1, -1,
     -1, -1, -1, -1,
     -1,
     -1, -1, -1, -1],
    [-1, -1, -1, -1,
     -1,
     1, 2, 3, 4,
     -1, -1, -1, -1,
     -1,
     -1, -1, -1, -1],
    [1, 1, 1, 1,
     -1,
     -1, -1, -1, -1,
     -1, -1, -1, -1,
     -1,
     -1, -1, -1, -1],
    [-1, -1, -1, -1,
     -1,
     1, 1, 1, 1,
     -1, -1, -1, -1,
     -1,
     -1, -1, -1, -1],
    [-1, -1, -1, -1,
     -1,
     -1, -1, -1, -1,
     1, 2, 3, 4,
     -1,
     -1, -1, -1, -1],
    [-1, -1, -1, -1,
     -1,
     -1, -1, -1, -1,
     -1, -1, -1, -1,
     1,
     -1, -1, -1, -1],
    [-1, -1, -1, -1,
     -1,
     -1, -1, -1, -1,
     -1, -1, -1, -1,
     -1,
     1, 2, 3, 4],
    [-1, -1, -1, -1,
     -1,
     -1, -1, -1, -1,
     1, 1, 1, 1,
     -1,
     -1, -1, -1, -1],
    [-1, -1, -1, -1,
     -1,
     -1, -1, -1, -1,
     -1, -1, -1, -1,
     -1,
     1, 1, 1, 1]]

params.must_match = [
    [],
    [0, 1, 2, 3, 9, 10, 11, 12],
    [5, 6, 7, 8, 14, 15, 16, 17],
    [0, 1, 2, 3, 4, 9, 10, 11, 12, 13],
    [0, 1, 2, 3, 4, 9, 10, 11, 12, 13],
    [4, 5, 6, 7, 8, 13, 14, 15, 16, 17],
    [0, 1, 2, 3],
    [],
    [5, 6, 7, 8],
    [],
    [],
    [9, 10, 11, 12],
    [],
    [14, 15, 16, 17],
    [],
    []]

# Set what will run
mnefun.do_processing(
    params,
    fetch_raw=False,
    do_score=True,
    push_raw=False,
    do_sss=False,
    fetch_sss=False,
    do_ch_fix=False,
    gen_ssp=False,
    apply_ssp=False,
    plot_psd=True,
    write_epochs=False,
    gen_covs=False,
    gen_fwd=False,
    gen_inv=False,
    gen_report=False,
    print_status=False
)