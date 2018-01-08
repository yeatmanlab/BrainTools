# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:05:40 2016

@author: sjjoo
"""
#%%
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
from numpy.random import randn
from scipy import stats as stats
import scipy.io as sio
import time
from functools import partial

from mne import set_config
set_config('MNE_MEMMAP_MIN_SIZE', '1M')
set_config('MNE_CACHE_DIR', '.tmp')

mne.set_config('MNE_USE_CUDA', 'true')

this_env = copy.copy(os.environ)
fs_dir = '/mnt/diskArray/projects/avg_fsurfer'
this_env['SUBJECTS_DIR'] = fs_dir

raw_dir = '/mnt/scratch/NLR_MEG3'

os.chdir(raw_dir)

subs2 = ['NLR_102_RS','NLR_110_HH',
        'NLR_145_AC',
        'NLR_152_TC','NLR_160_EK','NLR_161_AK','NLR_164_SF', # 'NLR_163_LF'
        'NLR_170_GM','NLR_172_TH','NLR_179_GM','NLR_180_ZD', # 'NLR_174_HS',
        'NLR_203_AM','NLR_204_AM','NLR_205_AC',
        'NLR_207_AH','NLR_211_LB','NLR_210_SB' # 'NLR_162_EF',
        ] 

brs2 = [98, 88, 102, 99, 91, 88, 86, 81, 88, 77, 83, 85, 86, 98, 116, 86, 110, 79, 104] #105, 89, 
brs2 = np.array(brs2)

session2 = ['102_rs160815',
            '110_hh160809','145_ac160823',
            '152_tc160623','160_ek160915',
            '161_ak160916',
            '164_sf160920','170_gm160822',
            '172_th160825','179_gm160913',
            '180_zd160826','203_am151029','204_am151120','205_ac160202',
            '207_ah160809',
            '211_lb160823','210_sb160822'
            ] 

n_subjects2 = len(subs2)

c_table = (    (0.6510,    0.8078,    0.8902), # Blue, Green, Red, Orange, Purple, yellow 
    (0.1216,    0.4706,    0.7059),
    (0.6980,    0.8745,    0.5412),
    (0.2000,    0.6275,    0.1725),
    (0.9843,    0.6039,    0.6000),
    (0.8902,    0.1020,    0.1098),
    (0.9922,    0.7490,    0.4353),
    (1.0000,    0.4980,         0),
    (0.7922,    0.6980,    0.8392),
    (0.4157,    0.2392,    0.6039),
    (1.0000,    1.0000,    0.6000),
    (0.6941,    0.3490,    0.1569))

method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2

fname_data = op.join(raw_dir, 'session2_data.npy')

m1 = np.ones(len(subs2))
m1[2] = False
m1[6] = False
m1[7] = False
m1[8] = False
all_subject = []
temp = np.where(m1)[0]
all_subject.extend(temp)

session1_loaded = False
if session1_loaded == True:
    m3 = []
    for n in np.arange(0,len(poor_subs)):
        if n != 1 and poor_subs[n] != 'NLR_204_AM':
            m3.append(subs2.index(poor_subs[n]))
    growth = np.empty(len(subs2))
    for n in np.arange(0,len(subs2)-3):
        growth[n] = np.subtract(brs2[n], brs[subs.index(subs2[n])])
    growth = np.array(growth)

fname_data = op.join(raw_dir, 'session2_data_ico5.npy')
#%%
"""
Here we do the real deal...
"""            
# Session 2
load_data = False

method = "dSPM" 
snr = 3.
lambda2 = 1. / snr ** 2
#conditions1 = [0, 2, 4, 6, 8, 16, 18, 20, 22, 24] # Lets compare word vs. scramble
conditions1 = ['word_c254_p20_dot', 'word_c254_p50_dot', 'word_c137_p20_dot',
               'word_c254_p80_dot', 'word_c137_p80_dot', #'bigram_c254_p20_dot',
#               'bigram_c254_p50_dot', 'bigram_c137_p20_dot',
               'word_c254_p20_word', 'word_c254_p50_word', 'word_c137_p20_word',
               'word_c254_p80_word', 'word_c137_p80_word', #'bigram_c254_p20_word',
#               'bigram_c254_p50_word', 'bigram_c137_p20_word'
               ]
#    conditions2 = [16, 22] # Lets compare word vs. scramble
X2 = np.empty((20484, 601, n_subjects2, len(conditions1)))
#word_data = np.empty((20484, 421, n_subjects, len(conditions1[8:])))
fs_vertices = [np.arange(10242)] * 2

n_epochs2 = np.empty((n_subjects2,len(conditions1)))
      
if load_data == False:
    for n, s in enumerate(session2):  
        os.chdir(os.path.join(raw_dir,session2[n]))
        os.chdir('inverse')
        
        fn = 'Conditions_40-sss_eq_'+session2[n]+'-ave.fif'
        fn_inv = session2[n] + '-inv-ico5.fif'
        inv = mne.minimum_norm.read_inverse_operator(fn_inv, verbose=None)
        
        for iCond in np.arange(0,len(conditions1)):
            evoked = mne.read_evokeds(fn, condition=conditions1[iCond], 
                                  baseline=(None,0), kind='average', proj=True)
            
            n_epochs2[n][iCond] = evoked.nave
            
            stc = mne.minimum_norm.apply_inverse(evoked,inv,lambda2, method=method, pick_ori="normal")
            
            stc.crop(-0.1, 0.9)
            tmin = stc.tmin
            tstep = stc.tstep
            times = stc.times
            # Average brain
            """
            One should check if morph map is current and correct. Otherwise, it will spit out and error.
            Check SUBJECTS_DIR/morph-maps
            """
            morph_mat = mne.compute_morph_matrix(subs2[n], 'fsaverage', stc.vertices,
                                                 fs_vertices, smooth=None,
                                                 subjects_dir=fs_dir)
            stc_fsaverage = stc.morph_precomputed('fsaverage', fs_vertices, morph_mat)
        #    morph_dat = mne.morph_data(subs[n], 'fsaverage', stc, n_jobs=16,
        #                        grade=fs_vertices, subjects_dir=fs_dir)
            X2[:,:,n,iCond] = stc_fsaverage.data

    os.chdir(raw_dir)
    np.save(fname_data, X2)
    np.save('session2_times.npy',times)
    np.save('session2_tstep.npy',tstep)
    np.save('session2_n_averages.npy',n_epochs2)
else:
    os.chdir(raw_dir)
    X2 = np.load(fname_data)
    times = np.load('session2_times.npy')
    tstep = np.load('session2_tstep.npy')
    n_epochs2 = np.load('session2_n_averages.npy')
    tmin = -0.1
    
#%%
""" Read HCP labels """
labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', surf_name='white', subjects_dir=fs_dir) #regexp=aparc_label_name
#aparc_label_name = 'PHT_ROI'#'_IP'#'IFSp_ROI'#'STSvp_ROI'#'STSdp_ROI'#'PH_ROI'#'TE2p_ROI' #'SFL_ROI' #'IFSp_ROI' #'TE2p_ROI' #'inferiortemporal' #'pericalcarine'
#        anat_label = mne.read_labels_from_annot('fsaverage', parc='aparc',surf_name='white',
#               subjects_dir=fs_dir, regexp=aparc_label_name)

#%%
""" Task effects """

PHT_label_lh = [label for label in labels if label.name == 'L_PHT_ROI-lh'][0]
PHT_label_rh = [label for label in labels if label.name == 'R_PHT_ROI-rh'][0]

TE1p_label_lh = [label for label in labels if label.name == 'L_TE1p_ROI-lh'][0]
TE1p_label_rh = [label for label in labels if label.name == 'R_TE1p_ROI-rh'][0]


TE2p_label_lh = [label for label in labels if label.name == 'L_TE2p_ROI-lh'][0]
TE2p_label_rh = [label for label in labels if label.name == 'R_TE2p_ROI-rh'][0]

PH_label_lh = [label for label in labels if label.name == 'L_PH_ROI-lh'][0]
PH_label_rh = [label for label in labels if label.name == 'R_PH_ROI-rh'][0]

FFC_label_lh = [label for label in labels if label.name == 'L_FFC_ROI-lh'][0]
FFC_label_rh = [label for label in labels if label.name == 'R_FFC_ROI-rh'][0]

a8C_label_lh = [label for label in labels if label.name == 'L_8C_ROI-lh'][0]
a8C_label_rh = [label for label in labels if label.name == 'R_8C_ROI-rh'][0]

p946v_label_lh = [label for label in labels if label.name == 'L_p9-46v_ROI-lh'][0]
p946v_label_rh = [label for label in labels if label.name == 'R_p9-46v_ROI-rh'][0]

IFSp_label_lh = [label for label in labels if label.name == 'L_IFSp_ROI-lh'][0]
IFSp_label_rh = [label for label in labels if label.name == 'R_IFSp_ROI-rh'][0]

IFJp_label_lh = [label for label in labels if label.name == 'L_IFJp_ROI-lh'][0]
IFJp_label_rh = [label for label in labels if label.name == 'R_IFJp_ROI-rh'][0]

IFJa_label_lh = [label for label in labels if label.name == 'L_IFJa_ROI-lh'][0]
IFJa_label_rh = [label for label in labels if label.name == 'R_IFJa_ROI-rh'][0]

a45_label_lh = [label for label in labels if label.name == 'L_45_ROI-lh'][0]
a45_label_rh = [label for label in labels if label.name == 'R_45_ROI-rh'][0]

a44_label_lh = [label for label in labels if label.name == 'L_44_ROI-lh'][0]
a44_label_rh = [label for label in labels if label.name == 'R_44_ROI-rh'][0]

PGi_label_lh = [label for label in labels if label.name == 'L_PGi_ROI-lh'][0]
PGi_label_rh = [label for label in labels if label.name == 'R_PGi_ROI-rh'][0]

PGs_label_lh = [label for label in labels if label.name == 'L_PGs_ROI-lh'][0]
PGs_label_rh = [label for label in labels if label.name == 'R_PGs_ROI-rh'][0]

STSvp_label_lh = [label for label in labels if label.name == 'L_STSvp_ROI-lh'][0]
STSvp_label_rh = [label for label in labels if label.name == 'R_STSvp_ROI-rh'][0]

STSdp_label_lh = [label for label in labels if label.name == 'L_STSdp_ROI-lh'][0]
STSdp_label_rh = [label for label in labels if label.name == 'R_STSdp_ROI-rh'][0]

TPOJ1_label_lh = [label for label in labels if label.name == 'L_TPOJ1_ROI-lh'][0]
TPOJ1_label_rh = [label for label in labels if label.name == 'R_TPOJ1_ROI-rh'][0]

V1_label_lh = [label for label in labels if label.name == 'L_V1_ROI-lh'][0]
V1_label_rh = [label for label in labels if label.name == 'R_V1_ROI-rh'][0]

LIPd_label_lh = [label for label in labels if label.name == 'L_LIPd_ROI-lh'][0]
LIPd_label_rh = [label for label in labels if label.name == 'R_LIPd_ROI-rh'][0]
LIPv_label_lh = [label for label in labels if label.name == 'L_LIPv_ROI-lh'][0]
LIPv_label_rh = [label for label in labels if label.name == 'R_LIPv_ROI-rh'][0]

new_data = X2[:,:,all_subject,:]
#data1 = np.subtract(np.mean(new_data[:,:,:,[5]],axis=3), np.mean(new_data[:,:,:,[0]],axis=3))
data1 = np.mean(new_data[:,:,:,[0,5]],axis=3)
del new_data

data11 = np.transpose(data1,[2,1,0])

stat_fun = partial(mne.stats.ttest_1samp_no_p)

temp3 = mne.SourceEstimate(np.transpose(stat_fun(data11)), fs_vertices, tmin, tstep,subject='fsaverage') # np.transpose(stat_fun(data11))

#temp3 = mne.SourceEstimate(X13[:,:,1,1], fs_vertices, tmin,
#                                 tstep,subject='fsaverage')

brain3_1 = temp3.plot(hemi='lh', subjects_dir=fs_dir, views = ['lat'], #views=['lat','ven','med'], #transparent = True,
          initial_time=0.40,clim=dict(kind='value', lims=[1.5, 1.72, np.max(temp3.data[:,:])]))
brain3_1.add_label(PHT_label_lh, borders=True, color=c_table[0])
brain3_1.add_label(TE2p_label_lh, borders=True, color=c_table[1])
brain3_1.add_label(PH_label_lh, borders=True, color=c_table[2])
brain3_1.add_label(FFC_label_lh, borders=True, color=c_table[3])
brain3_1.add_label(TE1p_label_lh, borders=True, color=c_table[4])

brain3_1.add_label(IFSp_label_lh, borders=True, color=c_table[5])
brain3_1.add_label(IFJp_label_lh, borders=True, color=c_table[6])
brain3_1.add_label(IFJa_label_lh, borders=True, color=c_table[7])
brain3_1.add_label(a45_label_lh, borders=True, color=c_table[8])
brain3_1.add_label(a44_label_lh, borders=True, color=c_table[8])
brain3_1.add_label(a8C_label_lh, borders=True, color=c_table[8])
brain3_1.add_label(p946v_label_lh, borders=True, color=c_table[8])

brain3_1.add_label(PGi_label_lh, borders=True, color=c_table[9])
brain3_1.add_label(PGs_label_lh, borders=True, color=c_table[9])
brain3_1.add_label(STSvp_label_lh, borders=True, color=c_table[11])
brain3_1.add_label(STSdp_label_lh, borders=True, color=c_table[11])
brain3_1.add_label(V1_label_lh, borders=True, color='k')
brain3_1.add_label(LIPd_label_lh, borders=True, color='k')
brain3_1.add_label(LIPv_label_lh, borders=True, color='k')

""" Frontal """
temp = temp3.in_label(a44_label_lh)
broca_vertices = temp.vertices[0]

temp = temp3.in_label(a45_label_lh)
broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))

temp = temp3.in_label(IFSp_label_lh)
broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))

#temp = temp3.in_label(IFJp_label_lh)
#broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))

#temp = temp3.in_label(IFJa_label_lh)
#broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))

""" Ventral """
temp = temp3.in_label(TE2p_label_lh)
ventral_vertices = temp.vertices[0]

temp = temp3.in_label(PH_label_lh)
ventral_vertices = np.unique(np.append(ventral_vertices, temp.vertices[0]))
#
#temp = temp3.in_label(FFC_label_lh)
#ventral_vertices = np.unique(np.append(ventral_vertices, temp.vertices[0]))

""" Temporal """
temp = temp3.in_label(STSvp_label_lh)
w_vertices = temp.vertices[0]

temp = temp3.in_label(STSdp_label_lh)
w_vertices = np.unique(np.append(w_vertices, temp.vertices[0]))

#temp = temp3.in_label(TPOJ1_label_lh)
#w_vertices = np.unique(np.append(w_vertices, temp.vertices[0]))

""" V1 """
temp = temp3.in_label(V1_label_lh)
v1_vertices = temp.vertices[0]

###############################################################################
""" Just to visualize the new ROI """
mask = np.logical_and(times >= 0.08, times <= 0.12)

lh_label = temp3.in_label(V1_label_lh)
data = np.max(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
v1_vertices = temp.vertices[0]
new_label = mne.Label(v1_vertices, hemi='lh')
brain3_1.add_label(new_label, borders=True, color='k')
###############################################################################
mask = np.logical_and(times >= 0.38, times <= 0.42)
lh_label = temp3.in_label(TE2p_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
ventral_vertices = temp.vertices[0]

lh_label = temp3.in_label(PH_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
ventral_vertices = np.unique(np.append(ventral_vertices, temp.vertices[0]))

lh_label = temp3.in_label(TE1p_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
ventral_vertices = np.unique(np.append(ventral_vertices, temp.vertices[0]))

new_label = mne.Label(ventral_vertices, hemi='lh')
brain3_1.add_label(new_label, borders=True, color='k')
###############################################################################
mask = np.logical_and(times >= 0.22, times <= 0.26)
lh_label = temp3.in_label(STSvp_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
w_vertices = temp.vertices[0]

lh_label = temp3.in_label(STSdp_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
w_vertices = np.unique(np.append(w_vertices, temp.vertices[0]))

new_label = mne.Label(w_vertices, hemi='lh')
brain3_1.add_label(new_label, borders=True, color='k')
###############################################################################
mask = np.logical_and(times >= 0.12, times <= 0.16)
lh_label = temp3.in_label(a44_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
broca_vertices = temp.vertices[0]

lh_label = temp3.in_label(IFJa_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))

lh_label = temp3.in_label(p946v_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))

new_label = mne.Label(broca_vertices, hemi='lh')
brain3_1.add_label(new_label, borders=True, color='k')
###############################################################################
mask = np.logical_and(times >= 0.12, times <= 0.16)
lh_label = temp3.in_label(PGi_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
STGp_vertices = temp.vertices[0]

lh_label = temp3.in_label(PGs_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
STGp_vertices = np.unique(np.append(STGp_vertices, temp.vertices[0]))

new_label = mne.Label(STGp_vertices, hemi='lh')
brain3_1.add_label(new_label, borders=True, color='k')
###############################################################################
mask = np.logical_and(times >= 0.12, times <= 0.16)
lh_label = temp3.in_label(LIPv_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
LIP_vertices = temp.vertices[0]

lh_label = temp3.in_label(LIPd_label_lh)
data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.72] = 0.

temp_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)
temp = temp3.in_label(temp_labels)
LIP_vertices = np.unique(np.append(LIP_vertices, temp.vertices[0]))

new_label = mne.Label(LIP_vertices, hemi='lh')
brain3_1.add_label(new_label, borders=True, color='k')

#%% 
""" Dot task: V1 """
plt.figure(1)
plt.clf()

#X11 = np.mean(X2[v1_vertices,:,:,:], axis=0)
#M = np.mean(X11[:,all_subject,:],axis=1)
#errM = np.std(X11[:,all_subject,:],axis=1) / np.sqrt(X2.shape[2])
X11 = X2[v1_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))

plt.subplot(2,3,1)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1,3])
plt.legend()
plt.title('Dot task: V1')

plt.subplot(2,3,4)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='High contrast')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,2],'-',color=c_table[7],label='Low contrast')
plt.fill_between(times, M[:,2]-errM[:,2], M[:,2]+errM[:,2], facecolor=c_table[7], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1,3])
plt.legend()

#%%
""" Dot task: All subjects """
plt.figure(2)
plt.clf()

X11 = X2[ventral_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))

plt.subplot(1,3,1)
plt.hold(True)

plt.plot(times, M[:,5],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,6],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,8],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Dot task: VWFA')

X11 = X2[broca_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)

errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))

plt.subplot(1,3,2)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Dot task: Frontal')

X11 = X2[w_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
errM = np.std(np.mean(X11[:,:,all_subject,:],axis=0),axis=1) / np.sqrt(len(all_subject))

plt.subplot(1,3,3)
plt.hold(True)

plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,1],'-',color=c_table[3],label='Med noise')
plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, M[:,3],'-',color=c_table[1],label='High noise')
plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([-1, 3])
plt.title('Dot task: STS')

#for iSub in np.arange(0,len(all_subject)):
#    plt.figure(100+iSub)
#    plt.clf()
#    plt.subplot(1,2,1)
#    plt.hold(True)
#    plt.plot(times, np.mean(X2[ventral_vertices,:,all_subject[iSub],0],axis=0), '-', color=c_table[5])
#    plt.plot(times, np.mean(X2[ventral_vertices,:,all_subject[iSub],1],axis=0), '-', color=c_table[3])
#    plt.plot(times, np.mean(X2[ventral_vertices,:,all_subject[iSub],3],axis=0), '-', color=c_table[1])
##    plt.plot([0.1, 0.1],[0, 8],'-',color='k')
#    plt.title(subs2[all_subject[iSub]])
#    plt.subplot(1,2,2)
#    plt.hold(True)
#    plt.plot(times, np.mean(X2[ventral_vertices,:,all_subject[iSub],5],axis=0), '-', color=c_table[5])
#    plt.plot(times, np.mean(X2[ventral_vertices,:,all_subject[iSub],6],axis=0), '-', color=c_table[3])
#    plt.plot(times, np.mean(X2[ventral_vertices,:,all_subject[iSub],8],axis=0), '-', color=c_table[1])
##    plt.plot([0.1, 0.1],[0, 8],'-',color='k')
#    plt.title(subs2[all_subject[iSub]])
