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
import os
os.chdir('/home/jyeatman/git/BrainTools/projects/NLR_MEG')
from score import score
from nlr_organizeMEG_mnefun import nlr_organizeMEG_mnefun
out_dir = '/mnt/diskArray/scratch/NLR_MEG'
os.chdir(out_dir)

params = mnefun.Params(tmin=-0.05, tmax=1.0, t_adjust=-39e-3, n_jobs=18,
                       decim=2, n_jobs_mkl=1, proj_sfreq=250,
                       n_jobs_fir='cuda', n_jobs_resample='cuda',
                       filter_length='5s', epochs_type='fif', lp_cut=40.,
                       bmin=-0.05, auto_bad=15., plot_raw=False, 
                       bem_type = '5120')
          
params.trans_to = (0., 0., .04) # Align each subject to a common reference frame
params.sss_type = 'python'
params.auto_bad_meg_thresh = 30 # THIS SHOULD NOT BE SO HIGH!
params.subjects = nlr_organizeMEG_mnefun(out_dir=out_dir)
params.subjects.sort() # Sort the subject list
# REMOVE BAD SUBJECTS
# 103_ac150609 svd does not converge [3]
# 121_gg150904 ERROR: continuous HPI was not active in this file! [10]
# 101_lg150618 SSS error [16]
# 202_dd15091 SVD does not converge [20]
# 201_gs150824 corrupted file [24]
# 201_gs150729 chpi not active [26]
# 105_bb150713 too many bad channels
# 137_gr151201 too many bad channels
# 205_ac151123 over 100 bad channels!
badsubs = ['103_ac150609','121_gg150904','101_lg150618','202_dd150919',
           '201_gs150824','201_gs150729', '138_la151208', '205_ac160202', '205_ac151123']
for n, s in enumerate(badsubs):
    subnum = params.subjects.index(s)
    print('Removing subject ' + str(subnum) + ' ' + params.subjects[subnum])
    params.subjects.remove(s)
print("Running " + str(len(params.subjects)) + ' Subjects') 
print("\n".join(params.subjects))
params.subject_indices = np.arange(0,len(params.subjects))
# BAD subjects 
# 103_ac150609 svd does not converge [3]
# 121_gg150904 ERROR: continuous HPI was not active in this file! [10]
# 101_lg150618 SSS error [16]
# 202_dd15091 SVD does not converge [20]
# 201_gs150824 corrupted file [24]
# 201_gs150729 chpi not active [26]
#params.subject_indices = np.concatenate((np.arange(0,3), np.arange(4,10), np.arange(11,16), np.arange(17,20),
#                                         np.arange(21, 24),[25], np.arange(27,len(params.subjects)))
#                                         , axis=0)
#params.subject_indices = np.arange(27,len(params.subjects))
params.structurals =[None] * len(params.subjects)
params.run_names = ['%s_1', '%s_2', '%s_3', '%s_4', '%s_5', '%s_6', '%s_7', '%s_8']
#params.subject_run_indices = [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6, 7]
#,None,None,None,None]

params.dates = [(2014, 0, 00)] * len(params.subjects)
#params.subject_indices = [0]
params.score = score  # scoring function to use
params.plot_drop_logs = False
params.on_missing = 'warning'
params.acq_ssh = 'kambiz@minea.ilabs.uw.edu'  # minea - 172.28.161.8
params.acq_dir = '/sinuhe/data03/jason_words'
params.sws_ssh = 'kam@kasga.ilabs.uw.edu'  # kasga - 172.28.161.8
params.sws_dir = '/data03/kam/nlr'
params.acq_ssh = 'jason@minea.ilabs.uw.edu'  # minea - 172.28.161.8
params.acq_dir = '/sinuhe/data03/jason_words'
params.sws_ssh = 'jason@kasga.ilabs.uw.edu'  # kasga - 172.28.161.8
params.sws_dir = '/data05/jason/NLR'
params.tsss_dur = 6.
params.mf_args = '-hpie 30 -hpig .8 -hpicons'
# epoch rejection criterion
params.reject = dict(grad=3000e-13, mag=4.0e-12)
params.flat = dict(grad=1e-13, mag=1e-15)
params.auto_bad_reject = params.reject
# params.auto_bad_flat = params.flat
params.ssp_eog_reject = dict(grad=3000e-13, mag=4.0e-12, eog=np.inf)
params.ssp_ecg_reject = dict(grad=3000e-13, mag=4.0e-12, eog=np.inf)
# params.bem_type = '5120'
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
    gen_fwd=False,
    gen_inv=False,
    print_status=False,
    gen_report=True
)


