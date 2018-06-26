# -*- coding: utf-8 -*-

# Authors: Sung Jun Joo; Jason Yeatman; Kambiz Tavabi <ktavabi@gmail.com>
#
#
# License: BSD (3-clause)

import numpy as np
import mnefun
import os
#import glob
os.chdir('/home/sjjoo/git/BrainTools/projects/NLR_MEG')
from score import score
from nlr_organizeMEG_mnefun import nlr_organizeMEG_mnefun
import mne
import time
#import pycuda.driver
#import pycuda.autoinit

t0 = time.time()

mne.set_config('MNE_USE_CUDA', 'true')

# At Possum projects folder mounted in the local disk
raw_dir = '/mnt/diskArray/projects/MEG/nlr/raw'

# At local hard drive
out_dir = '/mnt/scratch/NLR_MEG4'

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

subs = ['NLR_102_RS','NLR_105_BB','NLR_110_HH','NLR_127_AM',
        'NLR_132_WP','NLR_145_AC','NLR_150_MG',
        'NLR_152_TC','NLR_160_EK','NLR_161_AK','NLR_163_LF', # 'NLR_162_EF',
        'NLR_164_SF','NLR_170_GM','NLR_172_TH','NLR_174_HS','NLR_179_GM',
        'NLR_180_ZD','NLR_201_GS','NLR_203_AM',
        'NLR_204_AM','NLR_205_AC','NLR_207_AH','NLR_210_SB','NLR_211_LB',
        'NLR_GB310','NLR_KB218','NLR_GB267','NLR_JB420',
        'NLR_HB275','NLR_GB355'] # 'NLR_187_NB',

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
#params = mnefun.Params(tmin=-0.1, tmax=0.9, n_jobs=18, # t_adjust was -39e-3
#                       decim=2, n_jobs_mkl=1, proj_sfreq=250,
#                       n_jobs_fir='cuda', n_jobs_resample='cuda',
#                       filter_length='5s', epochs_type='fif', lp_cut=40.,
##                       hp_cut=0.15,hp_trans=0.1,
#                       bmin=-0.1, auto_bad=20., plot_raw=False, 
#                       bem_type = '5120-5120-5120')
          
# This sets the position of the head relative to the sensors. These values a
# A typical head position. So now in sensor space everyone is aligned. However
# We should also note that for source analysis it is better to leave this as
# the mne-fun default ==> Let's put None!!!
          
""" Organize subjects """
#out,ind = nlr_organizeMEG_mnefun(raw_dir=raw_dir,out_dir=out_dir,subs=subs)
""" The directory structure is really messy -- let's not use this function. """

os.chdir(out_dir)
#
#print(out)

#params.subjects.sort() # Sort the subject list
#print("Done sorting subjects.\n")

""" Attention!!!
164_sf160707_4_raw.fif: continuous HPI was not active in this file!
170_gm160613_5_raw.fif: in _fix_raw_eog_cals...non equal eog arrays???
172_th160825_6_raw.fif: origin of head out of helmet
201_gs150729_2_raw.fif: continuous HPI was not active in this file!

174_hs160620_1_raw.fif: Too many bad channels (62 based on grad=4000e-13, mag=4.0e-12)
174_hs160829_1_raw.fif: Too many bad channels (62 based on grad=4000e-13, mag=4.0e-12)
163_lf160707          : Too many bad channels --> Use grad=5000e-13, mag=5.0e-12
163_lf160920          : : Too many bad channels --> Use grad=5000e-13, mag=5.0e-12
"""

#for n, s in enumerate(badsubs):
#    subnum = out.index(s)
#    print('Removing subject ' + str(subnum) + ' ' + out[subnum])
#    out.remove(s)
#    ind[subnum] = []
#    ind.remove([])

out = ['102_rs160815','105_bb161011','110_hh160809','127_am161004',
       '132_wp161122','145_ac160823','150_mg160825',
       '152_tc160623','160_ek160915','161_ak160916','163_lf160920', #'162_ef160829',
       '164_sf160920','170_gm160822','172_th160825','174_hs160829','179_gm160913',
       '180_zd160826','201_gs150925','203_am151029',
       '204_am151120','205_ac160202','207_ah160809','210_sb160822','211_lb160823',
       'nlr_gb310170829','nlr_kb218170829','nlr_gb267170911','nlr_jb420170828',
       'nlr_hb275170828','nlr_gb355170907'] # 187_nb161205: EOG channel number suddenly changes at run4
#%%
out = ['170_gm160822'] #'162_ef160829',
#       '164_sf160920','170_gm160822','172_th160825','174_hs160829','179_gm160913',
#       '180_zd160826','201_gs150925','203_am151029',
#       '204_am151120','205_ac160202','207_ah160809','210_sb160822','211_lb160823',
#       'nlr_gb310170829','nlr_kb218170829','nlr_gb267170911','nlr_jb420170828',
#       'nlr_hb275170828','nlr_gb355170907']

for n, s in enumerate(out):
    print(s)    
    
for n, s in enumerate(out):
    params = mnefun.Params(tmin=-0.1, tmax=0.9, n_jobs=18, # t_adjust was -39e-3
                       decim=2, n_jobs_mkl=1, proj_sfreq=250,
                       n_jobs_fir='cuda', n_jobs_resample='cuda',
                       filter_length='5s', epochs_type='fif', lp_cut=40.,
#                       hp_cut=0.15,hp_trans=0.1,
                       bmin=-0.1, auto_bad=20., plot_raw=False) 
#                       bem_type = '5120-5120-5120')
    params.subjects = [s]
    params.sss_type = 'python'
    params.sss_regularize = 'in' # 'in' by default
    params.tsss_dur = 8. # 60 for adults with not much head movements. This was set to 6.
    params.st_correlation = 0.9
    
    params.auto_bad_meg_thresh = 10 # THIS SHOULD NOT BE SO HIGH!

    params.trans_to = None #'median'

    params.t_adjust = -39e-3 # time delay from the trigger. It's due to set trigger function. I don't know why...
    
    #print("Running " + str(len(params.subjects)) + ' Subjects') 
#    print("\n\n".join(params.subjects))
    print("\n\n")
    print("Running " + str(params.subjects)) 
    print("\n\n")
    
    params.subject_indices = np.arange(0,len(params.subjects))
    
    params.structurals =[None] * len(params.subjects)
    
    if s == '187_nb161205':
        params.run_names = ['%s_1', '%s_2', '%s_3', '%s_5','%s_6']
    elif s == '172_th160825':
        params.run_names = ['%s_1', '%s_2', '%s_3', '%s_4', '%s_5']
    else:
        params.run_names = ['%s_1', '%s_2', '%s_3', '%s_4', '%s_5', '%s_6']
    
        
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

#    params.acq_ssh = 'jason@minea.ilabs.uw.edu'  # minea - 172.28.161.8
#    params.acq_dir = '/sinuhe/data03/jason_words'
    params.sws_ssh = 'jason@kasga.ilabs.uw.edu'  # kasga - 172.28.161.8
    params.sws_dir = '/data05/jason/NLR'
    
    #params.mf_args = '-hpie 30 -hpig .8 -hpicons' # sjjoo-20160826: We are doing SSS using python
    
    # epoch  
    if s == '174_hs160620':
        params.reject = dict(grad=3000e-13, mag=4.0e-12)
    else:
        params.reject = dict(grad=3000e-13, mag=4.0e-12) 
#    params.reject = dict(grad=4000e-13, mag=4.0e-12)    

    params.ssp_eog_reject = dict(grad=params.reject['grad'], mag=params.reject['mag'], eog=np.inf)
    params.ssp_ecg_reject = dict(grad=params.reject['grad'], mag=params.reject['mag'], ecg=np.inf)
        
    params.flat = dict(grad=1e-13, mag=1e-15)
    
    params.auto_bad_reject = dict(grad=2*params.reject['grad'], mag=2*params.reject['mag'])         
        
    params.auto_bad_flat = params.flat
      
    params.cov_method = 'shrunk'
    
    params.get_projs_from = range(len(params.run_names))
    params.inv_names = ['%s']
    params.inv_runs = [range(0, len(params.run_names))]
    params.runs_empty = []
    
    params.proj_nums = [[0, 0, 0],  # ECG: grad/mag/eeg
                        [1, 1, 0],  # EOG # sjjoo-20160826: was 3
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
        fetch_raw=False,     # Fetch raw recording files from acquisition machine
        do_score=False,      # Do scoring to slice data into trials
    
        # Before running SSS, make SUBJ/raw_fif/SUBJ_prebad.txt file with
        # space-separated list of bad MEG channel numbers
        push_raw=False,      # Push raw files and SSS script to SSS workstation
        do_sss=False,        # Run SSS remotely (on sws) or locally with mne-python
        fetch_sss=False,     # Fetch SSSed files from SSS workstation
        do_ch_fix=False,     # Fix channel ordering
    
        # Before running SSP, examine SSS'ed files and make
        # SUBJ/bads/bad_ch_SUBJ_post-sss.txt; usually, this should only contain EEG
        # channels.
        gen_ssp=True,       # Generate SSP vectors
        apply_ssp=True,     # Apply SSP vectors and filtering
        plot_psd=False,      # Plot raw data power spectra
        write_epochs=True,  # Write epochs to disk
        gen_covs=True,      # Generate covariances
    
        # Make SUBJ/trans/SUBJ-trans.fif using mne_analyze; needed for fwd calc.
        gen_fwd=False,       # Generate forward solutions (and src space if needed)
        gen_inv=False,       # Generate inverses
        gen_report=False,    # Write mne report html of results to disk
        print_status=False,  # Print completeness status update
    
#        params,
#        fetch_raw=False,
#        do_score=True, # True
#        push_raw=False,
#        do_sss=True, # True
#        fetch_sss=False,
#        do_ch_fix=True, # True
#        gen_ssp=True, # True
#        apply_ssp=True, # True
#        write_epochs=True, # True
#        plot_psd=False,
#        gen_covs=False,
#        gen_fwd=False,
#        gen_inv=False,
#        print_status=False,
#        gen_report=True # true
    )

print('%i sec' % (time.time() - t0))
