# -*- coding: utf-8 -*-

"""Docstring

Notes: Ss 105_bb presented with issues during computation of EOG projectors despite
having valid EOG data.
Ss 102_rs No matching events found for word_c254_p50_dot (event id 102)
"""

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
#
# License: BSD (3-clause)

import numpy as np
import mnefun
from score import score

params = mnefun.Params(tmin=-0.05, tmax=1.0, t_adjust=-39e-3, n_jobs=18,
                       decim=2, proj_sfreq=250,
                       n_jobs_fir='cuda', n_jobs_resample='cuda',
                       filter_length='5s', lp_cut=40.,
                       bmin=-0.05, auto_bad=15.)

params.subjects = ['101_lg', '102_rs', '103_ac', '105_bb', '903_im', '106_km',
                   '107_tm','108_lg','109_kt','110_hh', '111_jh', '112_ar', 
                   '115_ps', '117_cg', '122_jb', '201_gs', '201_gs_b', 
                   '201_gs_c','202_dd','203_am_a','203_am_b','203_am_c', '204_am', '901_cg', '903_im']
params.structurals =[None] * len(params.subjects)
params.structurals = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,
                      'NLR_201_GS', 'NLR_201_GS', 'NLR_201_GS', 'NLR_202_dd', 'NLR_203_AM',None,None,None,None,None,]
# params.structurals = ['101_lg', '102_rs', '103_ac', '105_bb', '106_km', '903_im', '107_tm']
#assert len(set(params.subjects)) == len(params.structurals)
params.run_names = ['%s_1', '%s_2', '%s_3', '%s_4', '%s_5', '%s_6', '%s_7', '%s_8']
params.subject_run_indices = [None, [0, 1, 2, 3, 5, 6, 7],
                              None, None, None,
                              [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5],None,
                              None,None,[0, 2, 4, 6],None,None,None,None,[0, 2, 3, 4, 5, 6, 7],None,
                              [0, 2, 3, 4, 5, 6, 7],[0, 1, 2, 3],None,None,None,None,None,None]

params.dates = [(2014, 0, 00)] * len(params.subjects)
params.subject_indices = [19]
params.score = score  # scoring function to use
params.plot_drop_logs = False
# SSS options
params.sss_type = 'python'
params.sss_regularize = 'svd'
params.tsss_dur = 6.
params.st_correlation = .9
#TODO What is this about?
params.acq_ssh = 'kambiz@minea.ilabs.uw.edu'  # minea - 172.28.161.8
params.acq_dir = '/sinuhe/data03/jason_words'
params.sws_ssh = 'kam@kasga.ilabs.uw.edu'  # kasga - 172.28.161.8
params.sws_dir = '/data03/kam/nlr'
params.acq_ssh = 'jason@minea.ilabs.uw.edu'  # minea - 172.28.161.8
params.acq_dir = '/sinuhe/data03/jason_words'
params.sws_ssh = 'jason@kasga.ilabs.uw.edu'  # kasga - 172.28.161.8
params.sws_dir = '/data05/jason/NLR'
# epoch rejection criterion
params.reject = dict(grad=3000e-13, mag=4.0e-12)
params.flat = dict(grad=1e-13, mag=1e-15)
params.auto_bad_reject = params.reject
# params.auto_bad_flat = params.flat
params.ssp_eog_reject = dict(grad=3000e-13, mag=4.0e-12, eog=np.inf)
params.ssp_ecg_reject = dict(grad=3000e-13, mag=4.0e-12, eog=np.inf)
params.bem_type = '5120'
params.cov_method = 'shrunk'
params.get_projs_from = range(len(params.run_names))
params.inv_names = ['%s']
params.inv_runs = [range(0, len(params.run_names))]
params.runs_empty = []
params.proj_nums = [[2, 2, 0],  # ECG: grad/mag/eeg
                    [3, 3, 0],  # EOG
                    [0, 0, 0]]  # Continuous (from ERM)

# The scoring function needs to produce an event file with these values
params.in_names = ['word_c254_p20_dot', 'word_c254_p50_dot', 'word_c137_p20_dot',
                   'word_c254_p80_dot', 'word_c137_p80_dot',
                   'bigram_c254_p20_dot', 'bigram_c254_p50_dot', 'bigram_c137_p20_dot',
                   'word_c254_p20_word', 'word_c254_p50_word', 'word_c137_p20_word',
                   'word_c254_p80_word', 'word_c137_p80_word',
                   'bigram_c254_p20_word', 'bigram_c254_p50_word', 'bigram_c137_p20_word']

params.in_numbers = [101, 102, 103, 104, 105, 106, 107, 108,
                     201, 202, 203, 204, 205, 206, 207, 208]

# These lines define how to translate the above event types into evoked files
params.analyses = [
    'Conditions'
    ]

params.out_names = [
    ['word_c254_p20_dot', 'word_c254_p50_dot', 'word_c137_p20_dot',
     'word_c254_p80_dot', 'word_c137_p80_dot',
     'bigram_c254_p20_dot', 'bigram_c254_p50_dot', 'bigram_c137_p20_dot',
     'word_c254_p20_word', 'word_c254_p50_word', 'word_c137_p20_word',
     'word_c254_p80_word', 'word_c137_p80_word',
     'bigram_c254_p20_word', 'bigram_c254_p50_word', 'bigram_c137_p20_word']
]

params.out_numbers = [
    [101, 102, 103, 104, 105, 106, 107, 108,
     201, 202, 203, 204, 205, 206, 207, 208]
    ]

params.must_match = [
    []
    ]
# Set what will run
mnefun.do_processing(
    params,
    fetch_raw=False,
    do_score=False,
    push_raw=False,
    do_sss=False,
    fetch_sss=False,
    do_ch_fix=False,
    gen_ssp=False,
    apply_ssp=False,
    write_epochs=False,
    plot_psd=False,
    gen_covs=False,
    gen_fwd=True,
    gen_inv=True,
    print_status=False,
    gen_report=False
)
