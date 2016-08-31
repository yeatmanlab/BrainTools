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
import glob
os.chdir('/home/sjjoo/git/BrainTools/projects/NLR_MEG')
from score import score
from nlr_organizeMEG_mnefun import nlr_organizeMEG_mnefun
import mne

mne.set_config('MNE_USE_CUDA', 'true')

# At Possum projects folder mounted in the local disk
raw_dir = '/mnt/diskArray/projects/MEG/nlr/raw'
#out_dir = '/mnt/diskArray/scratch/NLR_MEG'
#out_dir = '/mnt/scratch/NLR_MEG'

# At local hard drive
out_dir = '/mnt/scratch/NLR_MEG'
#out_dir = '/mnt/scratch/adult'

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
    
os.chdir(out_dir)

# 162_ef and 208_lh missing
# 210_sb missing first session 
subs = ['102_rs','110_hh','145_ac','150_mg','151_rd','152_tc','160_ek','161_ak',
    '163_lf','164_sf','170_gm','172_th','174_hs','179_gm','180_zd','207_ah','210_sb','211_lb']

# tmin, tmax: sets the epoch
# bmin, bmax: sets the prestim duration for baseline correction. baseline is set
# as individual as default. Refer to _mnefun.py bmax is 0.0 by default
# hp_cut, lp_cut: set cutoff frequencies for highpass and lowpass
# I found that hp_cut of 0.03 is problematic because the tansition band is set
# to 5 by default, which makes the negative stopping frequency (0.03-5).
# It appears that currently data are acquired (online) using bandpass filter
# (0.03 - 326.4 Hz), so it might be okay not doing offline highpass filtering.
# It's worth checking later though. However, I think we should do baseline 
# correction by setting bmin and bmax. I found that mnefun does baseline 
# correction by default.
# sjjoo_20160809: Commented
params = mnefun.Params(tmin=-0.2, tmax=1.0, t_adjust=-39e-3, n_jobs=18,
                       decim=2, n_jobs_mkl=1, proj_sfreq=250,
                       n_jobs_fir='cuda', n_jobs_resample='cuda',
                       filter_length='5s', epochs_type='fif', lp_cut=40.,
                       bmin=-0.2, auto_bad=20., plot_raw=False, 
                       bem_type = '5120')
          
# This sets the position of the head relative to the sensors. These values a
# A typical head position. So now in sensor space everyone is aligned. However
# We should also note that for source analysis it is better to leave this as
# the mne-fun default
          
#params.trans_to = (0., 0., .04)

params.sss_type = 'python'
params.sss_regularize = 'svd' # 'in' by default
params.tsss_dur = 16. # 60 for adults with not much head movements. This was set to 6.

params.auto_bad_meg_thresh = 30 # THIS SHOULD NOT BE SO HIGH!

# Regular subjects
#out,ind = nlr_organizeMEG_mnefun(raw_dir=raw_dir,out_dir=out_dir,subs=subs)

params.subjects = ['170_gm160613']

#params.subjects.sort() # Sort the subject list
#print("Done sorting subjects.\n")

""" Attention!!!
164_sf160707_4_raw.fif: continuous HPI was not active in this file!
170_gm160613_6_raw.fif: in _fix_raw_eog_cals...non equal eog arrays???
"""

# REMOVE BAD SUBJECTS
#badsubs = ['102_rs150716','110_hh150824','152_tc160510','152_tc160527',
#           '164_sf160707','170_gm160613']
#for n, s in enumerate(badsubs):
#    subnum = params.subjects.index(s)
#    print('Removing subject ' + str(subnum) + ' ' + params.subjects[subnum])
#    params.subjects.remove(s)
#    ind[subnum] = []
#    ind.remove([])
    
print("Running " + str(len(params.subjects)) + ' Subjects') 
print("\n\n".join(params.subjects))
params.subject_indices = np.arange(0,len(params.subjects))

#params.subject_indices = np.concatenate((np.arange(0,3), np.arange(4,10), np.arange(11,16), np.arange(17,20),
#                                         np.arange(21, 24),[25], np.arange(27,len(params.subjects)))
#                                         , axis=0)
#params.subject_indices = np.arange(27,len(params.subjects))
params.structurals =[None] * len(params.subjects)

params.run_names = ['%s_1', '%s_2', '%s_3', '%s_5', '%s_6']

#params.subject_run_indices = np.array([
#    np.arange(0,ind[0]),np.arange(0,ind[1]),np.arange(0,ind[2]),np.arange(0,ind[3]),
#    np.arange(0,ind[4]),np.arange(0,ind[5]),np.arange(0,ind[6]),np.arange(0,ind[7]),
#    np.arange(0,ind[8]),np.arange(0,ind[9])#,np.arange(0,ind[11])
##    np.arange(0,ind[12]),np.arange(0,ind[13]),np.arange(0,ind[14]),np.arange(0,ind[15]),
##    np.arange(0,ind[16]),np.arange(0,ind[17]),np.arange(0,ind[18]),np.arange(0,ind[19]),
##    np.arange(0,ind[20]),np.arange(0,ind[21]),np.arange(0,ind[22]),np.arange(0,ind[23]),
##    np.arange(0,ind[24])
#])

params.dates = [(2014, 0, 00)] * len(params.subjects)
#params.subject_indices = [0]
params.score = score  # scoring function to use
params.plot_drop_logs = False
params.on_missing = 'warning'
#params.acq_ssh = 'kambiz@minea.ilabs.uw.edu'  # minea - 172.28.161.8
#params.acq_dir = '/sinuhe/data03/jason_words'
#params.sws_ssh = 'kam@kasga.ilabs.uw.edu'  # kasga - 172.28.161.8
#params.sws_dir = '/data03/kam/nlr'
params.acq_ssh = 'jason@minea.ilabs.uw.edu'  # minea - 172.28.161.8
params.acq_dir = '/sinuhe/data03/jason_words'
params.sws_ssh = 'jason@kasga.ilabs.uw.edu'  # kasga - 172.28.161.8
params.sws_dir = '/data05/jason/NLR'

#params.mf_args = '-hpie 30 -hpig .8 -hpicons' # sjjoo-20160826: We are doing SSS using python

# epoch rejection criterion
params.reject = dict(grad=4000e-13, mag=4.0e-12)
params.flat = dict(grad=1e-13, mag=1e-15)
params.auto_bad_reject = params.reject
# params.auto_bad_flat = params.flat
params.ssp_eog_reject = dict(grad=3000e-13, mag=4.0e-12, eog=np.inf)
params.ssp_ecg_reject = dict(grad=3000e-13, mag=4.0e-12, eog=np.inf)

params.cov_method = 'shrunk'

params.get_projs_from = range(len(params.run_names))
params.inv_names = ['%s']
params.inv_runs = [range(0, len(params.run_names))]
params.runs_empty = []

params.proj_nums = [[2, 2, 0],  # ECG: grad/mag/eeg
                    [2, 2, 0],  # EOG # sjjoo-20160826: was 3
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
    'All',
    'Conditions'
    ]

params.out_names = [
    ['ALL'],
    ['word_c254_p20_dot', 'word_c254_p50_dot', 'word_c137_p20_dot',
     'word_c254_p80_dot', 'word_c137_p80_dot',
     'bigram_c254_p20_dot', 'bigram_c254_p50_dot', 'bigram_c137_p20_dot',
     'word_c254_p20_word', 'word_c254_p50_word', 'word_c137_p20_word',
     'word_c254_p80_word', 'word_c137_p80_word',
     'bigram_c254_p20_word', 'bigram_c254_p50_word', 'bigram_c137_p20_word']
]

params.out_numbers = [
    [1] * len(params.in_numbers),
    [101, 102, 103, 104, 105, 106, 107, 108,
     201, 202, 203, 204, 205, 206, 207, 208]
    ]

params.must_match = [
    [],
    [],
    ]
# Set what will run
mnefun.do_processing(
    params,
    fetch_raw=False,
    do_score=False, # True
    push_raw=False,
    do_sss=True, # True
    fetch_sss=False,
    do_ch_fix=True, # True
    gen_ssp=True, # True
    apply_ssp=True, # True
    write_epochs=True,
    plot_psd=False,
    gen_covs=False,
    gen_fwd=False,
    gen_inv=False,
    print_status=False,
    gen_report=True
)


