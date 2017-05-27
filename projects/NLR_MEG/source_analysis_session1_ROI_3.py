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
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)

from mne import set_config

set_config('MNE_MEMMAP_MIN_SIZE', '1M')
set_config('MNE_CACHE_DIR', '.tmp')

mne.set_config('MNE_USE_CUDA', 'true')

this_env = copy.copy(os.environ)
fs_dir = '/mnt/diskArray/projects/avg_fsurfer'
this_env['SUBJECTS_DIR'] = fs_dir

raw_dir = '/mnt/scratch/NLR_MEG3'

os.chdir(raw_dir)

subs = ['NLR_102_RS','NLR_103_AC','NLR_110_HH','NLR_127_AM',
        'NLR_130_RW','NLR_132_WP','NLR_133_ML','NLR_145_AC','NLR_151_RD',
        'NLR_152_TC','NLR_160_EK','NLR_161_AK','NLR_163_LF','NLR_164_SF',
        'NLR_170_GM','NLR_172_TH','NLR_174_HS','NLR_179_GM','NLR_180_ZD',
        'NLR_187_NB','NLR_203_AM','NLR_204_AM','NLR_205_AC','NLR_206_LM',
        'NLR_207_AH','NLR_211_LB','NLR_150_MG'
        ] # 'NLR_202_DD','NLR_105_BB','NLR_150_MG','NLR_201_GS',

brs = [87, 102, 78, 115,
       91, 121, 77, 91, 93,
       88, 75, 90, 66, 59,
       81, 84, 81, 72, 71, 
       121,75, 66, 90, 93, 
       101, 56, 93] #75 101, 93,
brs = np.array(brs)

age = [125, 132, 138, 109,
       138, 108, 98, 105, 87,
       131, 123, 95, 112, 133,
       152, 103, 89, 138, 93,
       117, 122, 109, 90, 111,
       86, 147]
age = np.divide(age, 12)
# Session 1
# subs are synced up with session1 folder names...
#
session1 = ['102_rs160618','103_ac150609',
            '110_hh160608','127_am161004','130_rw151221',
            '132_wp160919','133_ml151124','145_ac160621',
            '151_rd160620','152_tc160422','160_ek160627',
            '161_ak160627','163_lf160707',
            '164_sf160707','170_gm160613','172_th160614',
            '174_hs160620','179_gm160701','180_zd160621',
            '187_nb161017','203_am150831',
            '204_am150829','205_ac151208','206_lm151119',
            '207_ah160608','211_lb160617','150_mg160606'
            ] #'202_dd150919'(# of average is zero) '105_bb150713'(# of average is less than 10)
              #,(# of average is less than 20) '201_gs150729'(# of average is less than 10)

n_subjects = len(subs)

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

fname_data = op.join(raw_dir, 'session1_data.npy')

m1 = np.transpose(brs) >= 90

m2 = np.logical_not(m1)

m1[12] = False
m2[12] = False
m1[16] = False
m2[16] = False
m1[26] = False
m2[26] = False

good_readers = np.where(m1)[0]
poor_readers = np.where(m2)[0]
all_subject = []
all_subject.extend(good_readers)
all_subject.extend(poor_readers)
all_subject.sort()

poor_subs = []
for n in np.arange(0,len(poor_readers)):
    poor_subs.append(subs[poor_readers[n]])

#m1 = np.transpose(age) > 9
#
#m2 = np.logical_not(m1)
#
#m2[12] = False
#m2[16] = False
#m2[26] = False
#old_readers = np.where(m1)[0]
#young_readers = np.where(m2)[0]
#
#all_readers = []
#all_readers.extend(good_readers)
#all_readers.extend(poor_readers)
#all_readers.sort()

#%%
"""
Here we do the real deal...
"""            
# Session 1
load_data = True

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
X13 = np.empty((20484, 601, n_subjects, len(conditions1)))
#word_data = np.empty((20484, 421, n_subjects, len(conditions1[8:])))
fs_vertices = [np.arange(10242)] * 2

n_epochs = np.empty((n_subjects,len(conditions1)))
      
if load_data == False:
    for n, s in enumerate(session1):  
        os.chdir(os.path.join(raw_dir,session1[n]))
        os.chdir('inverse')
        
        fn = 'Conditions_40-sss_eq_'+session1[n]+'-ave.fif'
        fn_inv = session1[n] + '-inv.fif'
        inv = mne.minimum_norm.read_inverse_operator(fn_inv, verbose=None)
        
        for iCond in np.arange(0,len(conditions1)):
            evoked = mne.read_evokeds(fn, condition=conditions1[iCond], 
                                  baseline=(None,0), kind='average', proj=True)
            
            n_epochs[n][iCond] = evoked.nave
            
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
            morph_mat = mne.compute_morph_matrix(subs[n], 'fsaverage', stc.vertices,
                                                 fs_vertices, smooth=None,
                                                 subjects_dir=fs_dir)
            stc_fsaverage = stc.morph_precomputed('fsaverage', fs_vertices, morph_mat)
        #    morph_dat = mne.morph_data(subs[n], 'fsaverage', stc, n_jobs=16,
        #                        grade=fs_vertices, subjects_dir=fs_dir)
            X13[:,:,n,iCond] = stc_fsaverage.data

    os.chdir(raw_dir)
    np.save(fname_data, X13)
    np.save('session1_times.npy',times)
    np.save('session1_tstep.npy',tstep)
    np.save('session1_n_averages.npy',n_epochs)
else:
    os.chdir(raw_dir)
    X13 = np.load(fname_data)
    times = np.load('session1_times.npy')
    tstep = np.load('session1_tstep.npy')
    n_epochs = np.load('session1_n_averages.npy')
    tmin = -0.1

#%%
"""
Let's just add or modify subjects
"""
data = np.load(fname_data)
X1, times, tstep = data['X1'], data['times'], data['tstep']

add_subject = ['127_am151022']

n = session1.index(add_subject[0])
os.chdir(os.path.join(raw_dir,session1[n]))
os.chdir('inverse')

fn = 'Conditions_40-sss_eq_'+session1[n]+'-ave.fif'

###############################################################################
""" Different sources """
fn_inv = session1[n] + '-inv-ico5.fif'
fn_inv = session1[n] + '-inv-oct6.fif'
###############################################################################

inv = mne.minimum_norm.read_inverse_operator(fn_inv, verbose=None)

for iCond in np.arange(0,len(conditions1)):
    evoked = mne.read_evokeds(fn, condition=conditions1[iCond], 
                          baseline=(None,0), kind='average', proj=True)
    
    stc = mne.minimum_norm.apply_inverse(evoked,inv,lambda2, method=method, pick_ori=None)
    
    stc.crop(0, 0.5) # only to include 0 ~ 500 ms after stim onset
    tmin = stc.tmin
    tstep = stc.tstep
    times = stc.times
    # Average brain
    fs_vertices = [np.arange(10242)] * 2
    morph_mat = mne.compute_morph_matrix(subs[n], 'fsaverage', stc.vertices,
                                         fs_vertices, smooth=None,
                                         subjects_dir=fs_dir)
    stc_fsaverage = stc.morph_precomputed('fsaverage', fs_vertices, morph_mat)
    
    X1[:,:,n,iCond] = stc_fsaverage.data

os.chdir(raw_dir)
np.savez_compressed(fname_data, X1=X1, times=times, tstep=tstep)
    
#%%
""" Read HCP labels """
labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', surf_name='white', subjects_dir=fs_dir) #regexp=aparc_label_name
#aparc_label_name = 'PHT_ROI'#'_IP'#'IFSp_ROI'#'STSvp_ROI'#'STSdp_ROI'#'PH_ROI'#'TE2p_ROI' #'SFL_ROI' #'IFSp_ROI' #'TE2p_ROI' #'inferiortemporal' #'pericalcarine'
#        anat_label = mne.read_labels_from_annot('fsaverage', parc='aparc',surf_name='white',
#               subjects_dir=fs_dir, regexp=aparc_label_name)
#%%
#labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'lh', subjects_dir=fs_dir)
#aud_label = [label for label in labels if label.name == 'L_A1_ROI-lh'][0]
#brain.add_label(aud_label, borders=False)
""" Task effects """

#TE2p_mask_lh = mne.Label.get_vertices_used(TE2p_label[0])
#TE2p_mask_rh = mne.Label.get_vertices_used(TE2p_label[1])
k = 1
#tmin = 0

tp1 = [0.08, 0.13, 0.15, 0.20, 0.30]
tp2 = [0.12, 0.17, 0.19, 0.24, 0.35]
mask = times == 0.15

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

new_data = X13[:,:,all_subject,:]
data1 = np.subtract(np.mean(new_data[:,:,:,[5]],axis=3), np.mean(new_data[:,:,:,[8]],axis=3))
data1 = np.mean(new_data[:,:,:,[0,5]],axis=3)
del new_data

data11 = np.transpose(data1,[2,1,0])

stat_fun = partial(mne.stats.ttest_1samp_no_p)

temp3 = mne.SourceEstimate(np.transpose(stat_fun(data11)), fs_vertices, tmin, tstep,subject='fsaverage') # np.transpose(stat_fun(data11))

#temp3 = mne.SourceEstimate(X13[:,:,1,1], fs_vertices, tmin,
#                                 tstep,subject='fsaverage')

brain3_1 = temp3.plot(hemi='lh', subjects_dir=fs_dir, views = ['ven'], #views=['lat','ven','med'], #transparent = True,
          initial_time=0.18, clim=dict(kind='value', lims=[1.5, 1.72, np.max(temp3.data[:,:])])) #pos_lims=[0, 4, 4.5] #np.max(temp3.data[:,:])]))
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

brain3_1.add_label(PGi_label_lh, borders=True, color=c_table[9])
brain3_1.add_label(PGs_label_lh, borders=True, color=c_table[9])
brain3_1.add_label(STSvp_label_lh, borders=True, color=c_table[11])
brain3_1.add_label(STSdp_label_lh, borders=True, color=c_table[11])
brain3_1.add_label(V1_label_lh, borders=True, color='k')

brain3_1.save_movie('V1_DotTask_HighC-LowC.mp4',time_dilation = 4.0,framerate = 30)

brain3_2 = temp3.plot(hemi='rh', subjects_dir=fs_dir,  views='lat',
          clim=dict(kind='value', lims=[2.9, 3, np.max(temp3.data[:,:])]),
          initial_time=0.15)
brain3_2.add_label(PHT_label_rh, borders=True, color=c_table[0])
brain3_2.add_label(TE2p_label_rh, borders=True, color=c_table[1])
brain3_2.add_label(PH_label_rh, borders=True, color=c_table[2])
brain3_2.add_label(FFC_label_rh, borders=True, color=c_table[3])
brain3_2.add_label(TE1p_label_rh, borders=True, color=c_table[4])
brain3_2.add_label(IFSp_label_rh, borders=True, color=c_table[5])
brain3_2.add_label(IFJp_label_rh, borders=True, color=c_table[6])
brain3_2.add_label(IFJa_label_rh, borders=True, color=c_table[7])
brain3_2.add_label(a45_label_rh, borders=True, color=c_table[8])

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
temp = temp3.in_label(PGi_label_lh)
w_vertices = temp.vertices[0]

temp = temp3.in_label(PGs_label_lh)
w_vertices = np.unique(np.append(w_vertices, temp.vertices[0]))

#temp = temp3.in_label(TPOJ1_label_lh)
#w_vertices = np.unique(np.append(w_vertices, temp.vertices[0]))

""" V1 """
temp = temp3.in_label(V1_label_lh)
v1_vertices = temp.vertices[0]


""" Just to visualize the new ROI """
mask = np.logical_and(times >= 0.06, times <= 0.09)

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
mask = np.logical_and(times >= 0.17, times <= 0.19)
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
#
#""" Overwrite functional with anatomical ROIs """
#lh_label = temp3.in_label(PHT_label_lh)
#
#data = np.mean(lh_label.data[:,:],axis=1)
##lh_label.data[data < 1.5] = 0.
#
#func_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
#                      subjects_dir=fs_dir, connected=False)
#ven_vertices = func_labels.vertices

# Figures
#%%
""" All subjects """
plt.figure(1)
plt.clf()

X11 = X13[ventral_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)

plt.subplot(3,2,1)
plt.hold(True)

plt.plot(times, np.mean(M[:,[0]],axis=1),'--',color=c_table[5],label='Word-No noise')
yerr = np.std(np.mean(M[:,[0]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[0]],axis=1)-yerr, np.mean(M[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')


plt.plot(times, np.mean(M[:,[1]],axis=1),'--',color=c_table[3],label='Word-Med noise')
yerr = np.std(np.mean(M[:,[1]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[1]],axis=1)-yerr, np.mean(M[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')


plt.plot(times, np.mean(M[:,[3]],axis=1),'--',color=c_table[1],label='Noise')
yerr = np.std(np.mean(M[:,[3]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[3]],axis=1)-yerr, np.mean(M[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.title('Dot task: Ventral')

plt.subplot(3,2,2)
plt.hold(True)
plt.plot(times, np.mean(M[:,[5]],axis=1),'-',color=c_table[5],label='Word-No noise')
yerr = np.std(np.mean(M[:,[5]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[5]],axis=1)-yerr, np.mean(M[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M[:,[6]],axis=1),'-',color=c_table[3],label='Word-Med noise')
yerr = np.std(np.mean(M[:,[6]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[6]],axis=1)-yerr, np.mean(M[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M[:,[8]],axis=1),'-',color=c_table[1],label= 'Noise')
yerr = np.std(np.mean(M[:,[8]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[8]],axis=1)-yerr, np.mean(M[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.title('Lexical task: Ventral')

X11 = X13[broca_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)

plt.subplot(3,2,3)
plt.hold(True)

plt.plot(times, np.mean(M[:,[0]],axis=1),'--',color=c_table[5],label='Word-No noise')
yerr = np.std(np.mean(M[:,[0]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[0]],axis=1)-yerr, np.mean(M[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')


plt.plot(times, np.mean(M[:,[1]],axis=1),'--',color=c_table[3],label='Word-Med noise')
yerr = np.std(np.mean(M[:,[1]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[1]],axis=1)-yerr, np.mean(M[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')


plt.plot(times, np.mean(M[:,[3]],axis=1),'--',color=c_table[1],label='Noise')
yerr = np.std(np.mean(M[:,[3]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[3]],axis=1)-yerr, np.mean(M[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
#plt.ylim([0, 3])
plt.title('Dot task: Frontal')

plt.subplot(3,2,4)
plt.hold(True)
plt.plot(times, np.mean(M[:,[5]],axis=1),'-',color=c_table[5],label='Word-No noise')
yerr = np.std(np.mean(M[:,[5]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[5]],axis=1)-yerr, np.mean(M[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M[:,[6]],axis=1),'-',color=c_table[3],label='Word-Med noise')
yerr = np.std(np.mean(M[:,[6]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[6]],axis=1)-yerr, np.mean(M[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M[:,[8]],axis=1),'-',color=c_table[1],label='Noise')
yerr = np.std(np.mean(M[:,[8]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[8]],axis=1)-yerr, np.mean(M[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
#plt.ylim([0, 3])
plt.title('Lexical task: Frontal')

X11 = X13[w_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)

plt.subplot(3,2,5)
plt.hold(True)

plt.plot(times, np.mean(M[:,[0]],axis=1),'--',color=c_table[5],label='Word-No noise')
yerr = np.std(np.mean(M[:,[0]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[0]],axis=1)-yerr, np.mean(M[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')


plt.plot(times, np.mean(M[:,[1]],axis=1),'--',color=c_table[3],label='Word-Med noise')
yerr = np.std(np.mean(M[:,[1]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[1]],axis=1)-yerr, np.mean(M[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')


plt.plot(times, np.mean(M[:,[3]],axis=1),'--',color=c_table[1],label='Noise')
yerr = np.std(np.mean(M[:,[3]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[3]],axis=1)-yerr, np.mean(M[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
#plt.ylim([0, 4])
plt.title('Dot task: Temporal')

plt.subplot(3,2,6)
plt.hold(True)
plt.plot(times, np.mean(M[:,[5]],axis=1),'-',color=c_table[5],label='Word-No noise')
yerr = np.std(np.mean(M[:,[5]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[5]],axis=1)-yerr, np.mean(M[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M[:,[6]],axis=1),'-',color=c_table[3],label='Word-Med noise')
yerr = np.std(np.mean(M[:,[6]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[6]],axis=1)-yerr, np.mean(M[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M[:,[8]],axis=1),'-',color=c_table[1],label='Noise')
yerr = np.std(np.mean(M[:,[8]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[8]],axis=1)-yerr, np.mean(M[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
#plt.ylim([0, 4])
plt.title('Lexical task: Temporal')

#%% 
""" V1 """
plt.figure(2)
plt.clf()

X11 = X13[v1_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)

plt.subplot(3,2,1)
plt.hold(True)

plt.plot(times, np.mean(M[:,[0]],axis=1),'--',color=c_table[5],label='Word-No noise')
yerr = np.std(np.mean(M[:,[0]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[0]],axis=1)-yerr, np.mean(M[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')


plt.plot(times, np.mean(M[:,[1]],axis=1),'--',color=c_table[3],label='Word-Med noise')
yerr = np.std(np.mean(M[:,[1]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[1]],axis=1)-yerr, np.mean(M[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')


plt.plot(times, np.mean(M[:,[3]],axis=1),'--',color=c_table[1],label='Noise')
yerr = np.std(np.mean(M[:,[3]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[3]],axis=1)-yerr, np.mean(M[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
#plt.ylim([0, 4])
plt.title('Dot task: V1')

plt.subplot(3,2,2)
plt.hold(True)
plt.plot(times, np.mean(M[:,[5]],axis=1),'-',color=c_table[5],label='Word-No noise')
yerr = np.std(np.mean(M[:,[5]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[5]],axis=1)-yerr, np.mean(M[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M[:,[6]],axis=1),'-',color=c_table[3],label='Word-Med noise')
yerr = np.std(np.mean(M[:,[6]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[6]],axis=1)-yerr, np.mean(M[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M[:,[8]],axis=1),'-',color=c_table[1],label='Noise')
yerr = np.std(np.mean(M[:,[8]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[8]],axis=1)-yerr, np.mean(M[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
#plt.ylim([0, 4])
plt.title('Lexical task: V1')

M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)
plt.subplot(3, 2, 3)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M1[:,[0]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[0]],axis=1)-yerr, np.mean(M1[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M1[:,[1]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[1]],axis=1)-yerr, np.mean(M1[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M1[:,[3]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[3]],axis=1)-yerr, np.mean(M1[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
#plt.ylim([0, 4])
plt.title('Dot task (GR): V1')

plt.subplot(3, 2, 4)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M1[:,[5]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[5]],axis=1)-yerr, np.mean(M1[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M1[:,[6]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[6]],axis=1)-yerr, np.mean(M1[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M1[:,[8]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[8]],axis=1)-yerr, np.mean(M1[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
#plt.ylim([0, 4])
plt.title('Lexical task (GR): V1')

plt.subplot(3, 2, 5)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M2[:,[0]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[0]],axis=1)-yerr, np.mean(M2[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M2[:,[1]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[1]],axis=1)-yerr, np.mean(M2[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M2[:,[3]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[3]],axis=1)-yerr, np.mean(M2[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
#plt.ylim([0, 4])
plt.title('Dot task (PR): V1')

plt.subplot(3, 2, 6)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M2[:,[5]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[5]],axis=1)-yerr, np.mean(M2[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M2[:,[6]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[6]],axis=1)-yerr, np.mean(M2[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M2[:,[8]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[8]],axis=1)-yerr, np.mean(M2[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
#plt.ylim([0, 4])
plt.title('Lexical task (PR): V1')

""" Plot individual V1 responses """
#for iSub in np.arange(0,len(poor_readers)):
#    plt.figure(100+iSub)
#    plt.clf()
#    plt.subplot(1,2,1)
#    plt.hold(True)
#    plt.plot(times, np.mean(X1[v1_vertices,:,poor_readers[iSub],0],axis=0), '--', color=c_table[5])
#    plt.plot(times, np.mean(X1[v1_vertices,:,poor_readers[iSub],1],axis=0), '--', color=c_table[3])
#    plt.plot(times, np.mean(X1[v1_vertices,:,poor_readers[iSub],3],axis=0), '--', color=c_table[1])
#    plt.plot([0.1, 0.1],[0, 8],'-',color='k')
#    plt.title(subs[poor_readers[iSub]])
#    plt.subplot(1,2,2)
#    plt.hold(True)
#    plt.plot(times, np.mean(X1[v1_vertices,:,poor_readers[iSub],5],axis=0), '-', color=c_table[5])
#    plt.plot(times, np.mean(X1[v1_vertices,:,poor_readers[iSub],6],axis=0), '-', color=c_table[3])
#    plt.plot(times, np.mean(X1[v1_vertices,:,poor_readers[iSub],8],axis=0), '-', color=c_table[1])
#    plt.plot([0.1, 0.1],[0, 8],'-',color='k')
#    plt.title(subs[poor_readers[iSub]])

#%%
""" Task effects in """
plt.figure(22)
plt.clf()

X11 = X13[ventral_vertices,:,:,:]
M = np.mean(np.mean(X11[:,:,all_subject,:],axis=0),axis=1)
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3,3,1)
plt.hold(True)
plt.plot(times, np.mean(M[:,[0]],axis=1),'--',color=c_table[5],label='Word-No noise')
yerr = np.std(np.mean(M[:,[0]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[0]],axis=1)-yerr, np.mean(M[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M[:,[5]],axis=1),'-',color=c_table[1],label='Word-No noise')
yerr = np.std(np.mean(M[:,[5]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[5]],axis=1)-yerr, np.mean(M[:,[5]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
#plt.ylim([0, 4])
plt.title('Low Noise - all')

plt.subplot(3,3,2)
plt.hold(True)
plt.plot(times, np.mean(M[:,[1]],axis=1),'--',color=c_table[5],label='Word-Med noise')
yerr = np.std(np.mean(M[:,[1]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[1]],axis=1)-yerr, np.mean(M[:,[1]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M[:,[6]],axis=1),'-',color=c_table[1],label='Word-Med noise')
yerr = np.std(np.mean(M[:,[6]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[6]],axis=1)-yerr, np.mean(M[:,[6]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
#plt.ylim([0, 4])
plt.title('Med Noise - all')

plt.subplot(3,3,3)
plt.hold(True)
plt.plot(times, np.mean(M[:,[3]],axis=1),'--',color=c_table[5],label='Noise')
yerr = np.std(np.mean(M[:,[3]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[3]],axis=1)-yerr, np.mean(M[:,[3]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M[:,[8]],axis=1),'-',color=c_table[1],label='Noise')
yerr = np.std(np.mean(M[:,[8]],axis=1)) / np.sqrt(len(all_subject))
plt.fill_between(times, np.mean(M[:,[8]],axis=1)-yerr, np.mean(M[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
#plt.ylim([0, 4])
plt.title('High Noise - all')

plt.subplot(3,3,4)
plt.hold(True)
plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5],label='Word-No noise')
yerr = np.std(np.mean(M1[:,[0]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[0]],axis=1)-yerr, np.mean(M1[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[1],label='Word-No noise')
yerr = np.std(np.mean(M1[:,[5]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[5]],axis=1)-yerr, np.mean(M1[:,[5]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
#plt.ylim([0, 4])
plt.title('Low Noise - good')

plt.subplot(3,3,5)
plt.hold(True)
plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[5],label='Word-Med noise')
yerr = np.std(np.mean(M1[:,[1]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[1]],axis=1)-yerr, np.mean(M1[:,[1]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[1],label='Word-Med noise')
yerr = np.std(np.mean(M1[:,[6]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[6]],axis=1)-yerr, np.mean(M1[:,[6]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
#plt.ylim([0, 4])
plt.title('Med Noise - good')

plt.subplot(3,3,6)
plt.hold(True)
plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[5],label='Noise')
yerr = np.std(np.mean(M1[:,[3]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[3]],axis=1)-yerr, np.mean(M1[:,[3]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1],label='Noise')
yerr = np.std(np.mean(M1[:,[8]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[8]],axis=1)-yerr, np.mean(M1[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
#plt.ylim([0, 4])
plt.title('High Noise - good')

plt.subplot(3,3,7)
plt.hold(True)
plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[5],label='Word-No noise')
yerr = np.std(np.mean(M2[:,[0]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[0]],axis=1)-yerr, np.mean(M2[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[1],label='Word-No noise')
yerr = np.std(np.mean(M2[:,[5]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[5]],axis=1)-yerr, np.mean(M2[:,[5]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
#plt.ylim([0, 4])
plt.title('Low Noise - poor')

plt.subplot(3,3,8)
plt.hold(True)
plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[5],label='Word-Med noise')
yerr = np.std(np.mean(M2[:,[1]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[1]],axis=1)-yerr, np.mean(M2[:,[1]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[1],label='Word-Med noise')
yerr = np.std(np.mean(M2[:,[6]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[6]],axis=1)-yerr, np.mean(M2[:,[6]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
#plt.ylim([0, 4])
plt.title('Med Noise - poor')

plt.subplot(3,3,9)
plt.hold(True)
plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[5],label='Noise')
yerr = np.std(np.mean(M2[:,[3]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[3]],axis=1)-yerr, np.mean(M2[:,[3]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[1],label='Noise')
yerr = np.std(np.mean(M2[:,[8]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[8]],axis=1)-yerr, np.mean(M2[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
#plt.ylim([0, 4])
plt.title('High Noise - poor')
#%%
""" Good readers """
plt.figure(3)
plt.clf()

X11 = X13[ventral_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 2, 1)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M1[:,[0]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[0]],axis=1)-yerr, np.mean(M1[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M1[:,[1]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[1]],axis=1)-yerr, np.mean(M1[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M1[:,[3]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[3]],axis=1)-yerr, np.mean(M1[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
#plt.ylim([0, 4])
plt.title('Dot task (GR): Ventral')

plt.subplot(3, 2, 2)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M1[:,[5]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[5]],axis=1)-yerr, np.mean(M1[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M1[:,[6]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[6]],axis=1)-yerr, np.mean(M1[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M1[:,[8]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[8]],axis=1)-yerr, np.mean(M1[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
#plt.ylim([0, 4])
plt.title('Lexical task (GR): Ventral')

X11 = X13[broca_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 2, 3)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M1[:,[0]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[0]],axis=1)-yerr, np.mean(M1[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M1[:,[1]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[1]],axis=1)-yerr, np.mean(M1[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M1[:,[3]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[3]],axis=1)-yerr, np.mean(M1[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Dot task (GR): Frontal')

plt.subplot(3, 2, 4)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M1[:,[5]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[5]],axis=1)-yerr, np.mean(M1[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M1[:,[6]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[6]],axis=1)-yerr, np.mean(M1[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M1[:,[8]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[8]],axis=1)-yerr, np.mean(M1[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Lexical task (GR): Frontal')

X11 = X13[w_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 2, 5)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M1[:,[0]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[0]],axis=1)-yerr, np.mean(M1[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M1[:,[1]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[1]],axis=1)-yerr, np.mean(M1[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M1[:,[3]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[3]],axis=1)-yerr, np.mean(M1[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Dot task (GR): Temporal')

plt.subplot(3, 2, 6)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M1[:,[5]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[5]],axis=1)-yerr, np.mean(M1[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M1[:,[6]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[6]],axis=1)-yerr, np.mean(M1[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M1[:,[8]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[8]],axis=1)-yerr, np.mean(M1[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Lexical task (GR): Temporal')

#%%
""" Poor readers """
plt.figure(4)
plt.clf()

X11 = X13[ventral_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 2, 1)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M2[:,[0]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[0]],axis=1)-yerr, np.mean(M2[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M2[:,[1]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[1]],axis=1)-yerr, np.mean(M2[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M2[:,[3]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[3]],axis=1)-yerr, np.mean(M2[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
#plt.ylim([0, 4])
plt.title('Dot task (PR): Ventral')

plt.subplot(3, 2, 2)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M2[:,[5]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[5]],axis=1)-yerr, np.mean(M2[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M2[:,[6]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[6]],axis=1)-yerr, np.mean(M2[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M2[:,[8]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[8]],axis=1)-yerr, np.mean(M2[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
#plt.ylim([0, 4])
plt.title('Lexical task (PR): Ventral')

X11 = X13[broca_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 2, 3)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M2[:,[0]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[0]],axis=1)-yerr, np.mean(M2[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M2[:,[1]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[1]],axis=1)-yerr, np.mean(M2[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M2[:,[3]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[3]],axis=1)-yerr, np.mean(M2[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Dot task (PR): Frontal')

plt.subplot(3, 2, 4)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M2[:,[5]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[5]],axis=1)-yerr, np.mean(M2[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M2[:,[6]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[6]],axis=1)-yerr, np.mean(M2[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M2[:,[8]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[8]],axis=1)-yerr, np.mean(M2[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Lexical task (PR): Frontal')

X11 = X13[w_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 2, 5)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M2[:,[0]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[0]],axis=1)-yerr, np.mean(M2[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M2[:,[1]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[1]],axis=1)-yerr, np.mean(M2[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M2[:,[3]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[3]],axis=1)-yerr, np.mean(M2[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Dot task (PR): Temporal')

plt.subplot(3, 2, 6)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M2[:,[5]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[5]],axis=1)-yerr, np.mean(M2[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M2[:,[6]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[6]],axis=1)-yerr, np.mean(M2[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M2[:,[8]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[8]],axis=1)-yerr, np.mean(M2[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Lexical task (PR): Temporal')

#%%
""" Dot task: Good vs. Poor """
plt.figure(5)
plt.clf()

X11 = X1[ventral_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 1)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[4])
yerr = np.std(np.mean(M2[:,[0]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[0]],axis=1)-yerr, np.mean(M2[:,[0]],axis=1)+yerr, facecolor=c_table[4], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M1[:,[0]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[0]],axis=1)-yerr, np.mean(M1[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('No Noise (GP vs. PR): Ventral')

plt.subplot(3, 3, 2)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[2])
yerr = np.std(np.mean(M2[:,[1]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[1]],axis=1)-yerr, np.mean(M2[:,[1]],axis=1)+yerr, facecolor=c_table[2], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M1[:,[1]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[1]],axis=1)-yerr, np.mean(M1[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Med Noise (GP vs. PR): Ventral')

plt.subplot(3, 3, 3)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[0])
yerr = np.std(np.mean(M2[:,[3]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[3]],axis=1)-yerr, np.mean(M2[:,[3]],axis=1)+yerr, facecolor=c_table[0], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M1[:,[3]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[3]],axis=1)-yerr, np.mean(M1[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Noise (GP vs. PR): Ventral')

X11 = X1[broca_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 4)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[4])
yerr = np.std(np.mean(M2[:,[0]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[0]],axis=1)-yerr, np.mean(M2[:,[0]],axis=1)+yerr, facecolor=c_table[4], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M1[:,[0]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[0]],axis=1)-yerr, np.mean(M1[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('No Noise (GP vs. PR): Frontal')

plt.subplot(3, 3, 5)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[2])
yerr = np.std(np.mean(M2[:,[1]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[1]],axis=1)-yerr, np.mean(M2[:,[1]],axis=1)+yerr, facecolor=c_table[2], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M1[:,[1]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[1]],axis=1)-yerr, np.mean(M1[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Med Noise (GP vs. PR): Frontal')

plt.subplot(3, 3, 6)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[0])
yerr = np.std(np.mean(M2[:,[3]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[3]],axis=1)-yerr, np.mean(M2[:,[3]],axis=1)+yerr, facecolor=c_table[0], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M1[:,[3]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[3]],axis=1)-yerr, np.mean(M1[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Noise (GP vs. PR): Frontal')

X11 = X1[w_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 7)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[4])
yerr = np.std(np.mean(M2[:,[0]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[0]],axis=1)-yerr, np.mean(M2[:,[0]],axis=1)+yerr, facecolor=c_table[4], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M1[:,[0]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[0]],axis=1)-yerr, np.mean(M1[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('No Noise (GP vs. PR): Frontal')

plt.subplot(3, 3, 8)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[2])
yerr = np.std(np.mean(M2[:,[1]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[1]],axis=1)-yerr, np.mean(M2[:,[1]],axis=1)+yerr, facecolor=c_table[2], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M1[:,[1]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[1]],axis=1)-yerr, np.mean(M1[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Med Noise (GP vs. PR): Frontal')

plt.subplot(3, 3, 9)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[0])
yerr = np.std(np.mean(M2[:,[3]],axis=1)) / np.sqrt(len(poor_readers))
plt.fill_between(times, np.mean(M2[:,[3]],axis=1)-yerr, np.mean(M2[:,[3]],axis=1)+yerr, facecolor=c_table[0], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M1[:,[3]],axis=1)) / np.sqrt(len(good_readers))
plt.fill_between(times, np.mean(M1[:,[3]],axis=1)-yerr, np.mean(M1[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Noise (GP vs. PR): Frontal')

#%%
""" Lexical task: Good vs. Poor """
plt.figure(6)
plt.clf()

X11 = X1[ventral_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 1)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[4])
yerr = np.std(np.mean(M2[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[5]],axis=1)-yerr, np.mean(M2[:,[5]],axis=1)+yerr, facecolor=c_table[4], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M1[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[5]],axis=1)-yerr, np.mean(M1[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('No Noise (GP vs. PR): Ventral')

plt.subplot(3, 3, 2)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[2])
yerr = np.std(np.mean(M2[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[6]],axis=1)-yerr, np.mean(M2[:,[6]],axis=1)+yerr, facecolor=c_table[2], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M1[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[6]],axis=1)-yerr, np.mean(M1[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Med Noise (GP vs. PR): Ventral')

plt.subplot(3, 3, 3)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[0])
yerr = np.std(np.mean(M2[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[8]],axis=1)-yerr, np.mean(M2[:,[8]],axis=1)+yerr, facecolor=c_table[0], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M1[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[8]],axis=1)-yerr, np.mean(M1[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Noise (GP vs. PR): Ventral')

X11 = X1[broca_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 4)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[4])
yerr = np.std(np.mean(M2[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[5]],axis=1)-yerr, np.mean(M2[:,[5]],axis=1)+yerr, facecolor=c_table[4], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M1[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[5]],axis=1)-yerr, np.mean(M1[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('No Noise (GP vs. PR): Frontal')

plt.subplot(3, 3, 5)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[2])
yerr = np.std(np.mean(M2[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[6]],axis=1)-yerr, np.mean(M2[:,[6]],axis=1)+yerr, facecolor=c_table[2], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M1[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[6]],axis=1)-yerr, np.mean(M1[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Med Noise (GP vs. PR): Frontal')

plt.subplot(3, 3, 6)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[0])
yerr = np.std(np.mean(M2[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[8]],axis=1)-yerr, np.mean(M2[:,[8]],axis=1)+yerr, facecolor=c_table[0], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M1[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[8]],axis=1)-yerr, np.mean(M1[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Noise (GP vs. PR): Frontal')

X11 = X1[w_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 7)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[4])
yerr = np.std(np.mean(M2[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[5]],axis=1)-yerr, np.mean(M2[:,[5]],axis=1)+yerr, facecolor=c_table[4], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M1[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[5]],axis=1)-yerr, np.mean(M1[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('No Noise (GP vs. PR): Temporal')

plt.subplot(3, 3, 8)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[2])
yerr = np.std(np.mean(M2[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[6]],axis=1)-yerr, np.mean(M2[:,[6]],axis=1)+yerr, facecolor=c_table[2], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M1[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[6]],axis=1)-yerr, np.mean(M1[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Med Noise (GP vs. PR): Temporal')

plt.subplot(3, 3, 9)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[0])
yerr = np.std(np.mean(M2[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[8]],axis=1)-yerr, np.mean(M2[:,[8]],axis=1)+yerr, facecolor=c_table[0], alpha=0.3, edgecolor='none')

plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M1[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[8]],axis=1)-yerr, np.mean(M1[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')
plt.grid(b=True)
plt.ylim([0, 4])
plt.title('Noise (GP vs. PR): Temporal')

#%%
""" Task effects """
plt.figure(7)
plt.clf()

X11 = X1[ventral_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 1)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M1[:,[0]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[0]],axis=1)-yerr, np.mean(M1[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M1[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[5]],axis=1)-yerr, np.mean(M1[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Ventral')

plt.subplot(3, 3, 2)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M1[:,[1]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[1]],axis=1)-yerr, np.mean(M1[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M1[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[6]],axis=1)-yerr, np.mean(M1[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Ventral')

plt.subplot(3, 3, 3)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M1[:,[3]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[3]],axis=1)-yerr, np.mean(M1[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M1[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[8]],axis=1)-yerr, np.mean(M1[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Ventral')

X11 = X1[broca_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 4)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M1[:,[0]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[0]],axis=1)-yerr, np.mean(M1[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M1[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[5]],axis=1)-yerr, np.mean(M1[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Frontal')

plt.subplot(3, 3, 5)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M1[:,[1]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[1]],axis=1)-yerr, np.mean(M1[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M1[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[6]],axis=1)-yerr, np.mean(M1[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Frontal')

plt.subplot(3, 3, 6)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M1[:,[3]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[3]],axis=1)-yerr, np.mean(M1[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M1[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[8]],axis=1)-yerr, np.mean(M1[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Frontal')

X11 = X1[w_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 7)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M1[:,[0]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[0]],axis=1)-yerr, np.mean(M1[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M1[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[5]],axis=1)-yerr, np.mean(M1[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Temporal')

plt.subplot(3, 3, 8)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M1[:,[1]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[1]],axis=1)-yerr, np.mean(M1[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M1[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[6]],axis=1)-yerr, np.mean(M1[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Temporal')

plt.subplot(3, 3, 9)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M1[:,[3]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[3]],axis=1)-yerr, np.mean(M1[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M1[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M1[:,[8]],axis=1)-yerr, np.mean(M1[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Temporal')

#%%
""" Task effects """
plt.figure(8)
plt.clf()

X11 = X1[ventral_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 1)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M2[:,[0]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[0]],axis=1)-yerr, np.mean(M2[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M2[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[5]],axis=1)-yerr, np.mean(M2[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Ventral')

plt.subplot(3, 3, 2)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M2[:,[1]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[1]],axis=1)-yerr, np.mean(M2[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M2[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[6]],axis=1)-yerr, np.mean(M2[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Ventral')

plt.subplot(3, 3, 3)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M2[:,[3]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[3]],axis=1)-yerr, np.mean(M2[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M2[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[8]],axis=1)-yerr, np.mean(M2[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Ventral')

X11 = X1[broca_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 4)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M2[:,[0]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[0]],axis=1)-yerr, np.mean(M2[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M2[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[5]],axis=1)-yerr, np.mean(M2[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Frontal')

plt.subplot(3, 3, 5)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M2[:,[1]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[1]],axis=1)-yerr, np.mean(M2[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M2[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[6]],axis=1)-yerr, np.mean(M2[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Frontal')

plt.subplot(3, 3, 6)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M2[:,[3]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[3]],axis=1)-yerr, np.mean(M2[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M2[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[8]],axis=1)-yerr, np.mean(M2[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Frontal')

X11 = X1[w_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.subplot(3, 3, 7)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[5])
yerr = np.std(np.mean(M2[:,[0]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[0]],axis=1)-yerr, np.mean(M2[:,[0]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[5])
yerr = np.std(np.mean(M2[:,[5]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[5]],axis=1)-yerr, np.mean(M2[:,[5]],axis=1)+yerr, facecolor=c_table[5], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Temporal')

plt.subplot(3, 3, 8)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[3])
yerr = np.std(np.mean(M2[:,[1]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[1]],axis=1)-yerr, np.mean(M2[:,[1]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[3])
yerr = np.std(np.mean(M2[:,[6]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[6]],axis=1)-yerr, np.mean(M2[:,[6]],axis=1)+yerr, facecolor=c_table[3], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Temporal')

plt.subplot(3, 3, 9)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[1])
yerr = np.std(np.mean(M2[:,[3]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[3]],axis=1)-yerr, np.mean(M2[:,[3]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[1])
yerr = np.std(np.mean(M2[:,[8]],axis=1)) / np.sqrt(M.shape[1])
plt.fill_between(times, np.mean(M2[:,[8]],axis=1)-yerr, np.mean(M2[:,[8]],axis=1)+yerr, facecolor=c_table[1], alpha=0.2, edgecolor='none')

plt.grid(b=True)
plt.ylim([0, 4])
plt.title('GR: Temporal')


#%%
""" Right h """
temp = temp3.in_label(a44_label_rh)

broca_vertices = temp.vertices[1]

temp = temp3.in_label(a45_label_rh)

broca_vertices = np.unique(np.append(broca_vertices, temp.vertices[0]))

temp = temp3.in_label(TE2p_label_rh)

ventral_vertices1 = temp.vertices[1]

temp = temp3.in_label(PH_label_rh)

ventral_vertices1 = np.unique(np.append(ventral_vertices1, temp.vertices[0]))

temp = temp3.in_label(PHT_label_rh)

ventral_vertices1 = np.unique(np.append(ventral_vertices1, temp.vertices[0]))

temp = temp3.in_label(TE1p_label_rh)

ventral_vertices1 = np.unique(np.append(ventral_vertices1, temp.vertices[0]))


plt.figure(20)
plt.clf()

plt.subplot(2,1,1)
plt.hold(True)
X11 = X1[ventral_vertices1,:,:,:]

M = np.mean(np.mean(X11[:,:,:,:],axis=0),axis=1)
plt.plot(times, np.mean(M[:,[0]],axis=1),'--',color=c_table[5])
plt.plot(times, np.mean(M[:,[5]],axis=1),'-',color=c_table[5],label='Word-No noise')

plt.plot(times, np.mean(M[:,[1]],axis=1),'--',color=c_table[3])
plt.plot(times, np.mean(M[:,[6]],axis=1),'-',color=c_table[3],label='Word-Med noise')

plt.plot(times, np.mean(M[:,[3]],axis=1),'--',color=c_table[1])
plt.plot(times, np.mean(M[:,[8]],axis=1),'-',color=c_table[1],label='Noise')

plt.subplot(2,1,2)
plt.hold(True)
X11 = X1[broca_vertices,:,:,:]

M = np.mean(np.mean(X11[:,:,:,:],axis=0),axis=1)
plt.plot(times, np.mean(M[:,[0]],axis=1),'--',color=c_table[5])
plt.plot(times, np.mean(M[:,[5]],axis=1),'-',color=c_table[5],label='Word-No noise')

plt.plot(times, np.mean(M[:,[1]],axis=1),'--',color=c_table[3])
plt.plot(times, np.mean(M[:,[6]],axis=1),'-',color=c_table[3],label='Word-Med noise')

plt.plot(times, np.mean(M[:,[3]],axis=1),'--',color=c_table[1])
plt.plot(times, np.mean(M[:,[8]],axis=1),'-',color=c_table[1],label='Noise')


X11 = X1[ventral_vertices1,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.figure(30)
plt.clf()

plt.subplot(2, 1, 1)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5],label='Word-No noise')

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3],label='Word-Med noise')

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1],label='Noise')

plt.ylabel('%s value' % method)
plt.title('Good readers')
plt.legend(loc='upper left', shadow=True)

plt.subplot(2, 1, 2)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[5])
plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[5],label='Word-No noise')

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[3])
plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[3],label='Word-Med noise')

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[1])
plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[1],label='Noise')

#plt.ylim(0, 4)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.title('Poor readers')
plt.legend(loc='upper left', shadow=True)

plt.show()

X11 = X1[broca_vertices,:,:,:]
M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.figure(40)
plt.clf()

plt.subplot(2, 1, 1)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5],label='Word-No noise')

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3],label='Word-Med noise')

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1],label='Noise')

plt.ylabel('%s value' % method)
plt.title('Good readers')
plt.legend(loc='upper left', shadow=True)

plt.subplot(2, 1, 2)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[5])
plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[5],label='Word-No noise')

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[3])
plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[3],label='Word-Med noise')

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[1])
plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[1],label='Noise')

#plt.ylim(0, 4)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.title('Poor readers')
plt.legend(loc='upper left', shadow=True)

plt.show()

#%%
#plt.plot([0.15, 0.15],[0, 4],'-',color='k')

#plt.ylim(0, 4)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
#plt.title('High contrast')
plt.legend(loc='upper left', shadow=True)

plt.show()


M1 = np.mean(np.mean(X11[:,:,good_readers,:],axis=0),axis=1)   
M2 = np.mean(np.mean(X11[:,:,poor_readers,:],axis=0),axis=1)

plt.figure(3)
plt.clf()

plt.subplot(2, 1, 1)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[0]],axis=1),'--',color=c_table[5])
plt.plot(times, np.mean(M1[:,[5]],axis=1),'-',color=c_table[5],label='Word-No noise')

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3],label='Word-Med noise')

plt.plot(times, np.mean(M1[:,[3]],axis=1),'--',color=c_table[1])
plt.plot(times, np.mean(M1[:,[8]],axis=1),'-',color=c_table[1],label='Noise')

plt.ylabel('%s value' % method)
plt.title('Good readers')
plt.legend(loc='upper left', shadow=True)

plt.subplot(2, 1, 2)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[0]],axis=1),'--',color=c_table[5])
plt.plot(times, np.mean(M2[:,[5]],axis=1),'-',color=c_table[5],label='Word-No noise')

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[3])
plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[3],label='Word-Med noise')

plt.plot(times, np.mean(M2[:,[3]],axis=1),'--',color=c_table[1])
plt.plot(times, np.mean(M2[:,[8]],axis=1),'-',color=c_table[1],label='Noise')

#plt.ylim(0, 4)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.title('Poor readers')
plt.legend(loc='upper left', shadow=True)

plt.show()

#%%
k = 1

tp1 = [0.08, 0.13, 0.15, 0.20, 0.30]
tp2 = [0.12, 0.17, 0.19, 0.24, 0.35]
mask = np.logical_and(times >= tp1[k], times <= tp2[k])

""" Plot scatter """
temp_diff = np.subtract(np.mean(X1[:,:,:,[0,1]],axis=3), np.mean(X1[:,:,:,[3]],axis=3))
data_diff = np.mean(temp_diff[vwfa_labels,:,:], axis = 0)
#data1 = data1.reshape((data1.shape[1],data1.shape[0],data1.shape[2]))
vwfa_response = np.mean(data_diff[mask,:],axis=0)

plt.figure(4)
plt.clf()
ax = plt.subplot()
ax.scatter(brs, vwfa_response, s=30, c='k', alpha=0.5)
for i, txt in enumerate(subs):
    ax.annotate(txt, (brs[i],vwfa_response[i]))

np.corrcoef(vwfa_response,brs)

#%%
""" Individual """
for iSub, s in enumerate(subs):
    plt.figure(100+iSub)
    plt.clf()
    plt.subplot(2,2,1)
    plt.hold(True)
    plt.plot(times, np.mean(X1[ven_vertices,:,iSub,0],axis=0), '--', color=c_table[5])
    plt.plot(times, np.mean(X1[ven_vertices,:,iSub,3],axis=0), '--', color=c_table[1])
    
    plt.subplot(2,2,2)
    plt.hold(True)
    plt.plot(times, np.mean(X1[ven_vertices,:,iSub,5],axis=0), '-', color=c_table[5])
    plt.plot(times, np.mean(X1[ven_vertices,:,iSub,8],axis=0), '-', color=c_table[1])
    
    plt.subplot(2,2,3)
    plt.hold(True)
    plt.plot(times, np.mean(X1[ven_vertices,:,iSub,5],axis=0), '-', color=c_table[5])
    plt.plot(times, np.mean(X1[ven_vertices,:,iSub,0],axis=0), '--', color=c_table[5])
    
    plt.subplot(2,2,4)
    plt.hold(True)
    plt.plot(times, np.mean(X1[ven_vertices,:,iSub,8],axis=0), '-', color=c_table[1])
    plt.plot(times, np.mean(X1[ven_vertices,:,iSub,3],axis=0), '--', color=c_table[1])
    plt.title(s)

#%%
""" Good readers vs. poor readers """
k = 1
#tmin = 0

tp1 = [0.08, 0.13, 0.15, 0.20, 0.30]
tp2 = [0.12, 0.17, 0.19, 0.24, 0.35]
mask = times == 0.15

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

stv_label_lh = [label for label in labels if label.name == 'L_STV_ROI-lh'][0]
stv_label_rh = [label for label in labels if label.name == 'R_STV_ROI-rh'][0]

#new_data = X1[:,:,all_subject,:]
good_data = X1[:,:,good_readers,:]
poor_data = X1[:,:,poor_readers,:]
data1 = np.subtract(np.mean(good_data[:,:,:,[6]],axis=3), np.mean(poor_data[:,:,:,[6]],axis=3))
#del new_data
data11 = data1[:,:,:]
del data1, good_data, poor_data

#stat_fun = partial(mne.stats.ttest_1samp_no_p,sigma=1e-3)
data11 = np.transpose(data11,[2,1,0])
stat_fun = partial(mne.stats.ttest_1samp_no_p)

temp3 = mne.SourceEstimate(np.transpose(stat_fun(data11)), fs_vertices, tmin,
                                 tstep,subject='fsaverage')
     
brain3_1 = temp3.plot(hemi='lh', subjects_dir=fs_dir, views = ['lat','ven','med'], #views=['lat','ven','med'], #transparent = True,
          initial_time=0.15) # clim=dict(kind='value', lims=[2.0, 2.1, np.max(temp3.data[:,:])])
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
brain3_1.add_label(stv_label_lh, borders=True, color=c_table[9])

brain3_1.save_movie('GoodPoor_Lexical_MedNoise_lh.mp4',time_dilation = 4.0,framerate = 24)

#%%
""" Single condition """
k = 1
#tmin = 0

tp1 = [0.08, 0.13, 0.15, 0.20, 0.30]
tp2 = [0.12, 0.17, 0.19, 0.24, 0.35]
mask = times == 0.15

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

stv_label_lh = [label for label in labels if label.name == 'L_STV_ROI-lh'][0]
stv_label_rh = [label for label in labels if label.name == 'R_STV_ROI-rh'][0]

#new_data = X1[:,:,all_subject,:]
good_data = X1[:,:,good_readers,:]
poor_data = X1[:,:,poor_readers,:]
data1 = np.mean(good_data[:,:,:,[6]],axis=3)
#del new_data
data11 = data1[:,:,:]
del data1, good_data, poor_data

#stat_fun = partial(mne.stats.ttest_1samp_no_p,sigma=1e-3)
data11 = np.transpose(data11,[2,1,0])
stat_fun = partial(mne.stats.ttest_1samp_no_p)

temp3 = mne.SourceEstimate(np.transpose(stat_fun(data11)), fs_vertices, tmin,
                                 tstep,subject='fsaverage')
     
brain3_1 = temp3.plot(hemi='lh', subjects_dir=fs_dir, views = ['lat','ven','med'], #views=['lat','ven','med'], #transparent = True,
          initial_time=0.15) # clim=dict(kind='value', lims=[2.0, 2.1, np.max(temp3.data[:,:])])
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
brain3_1.add_label(stv_label_lh, borders=True, color=c_table[9])

brain3_1.save_movie('Single_Lexical_MedNoise_GR_lh.mp4',time_dilation = 4.0,framerate = 24)


#%%
""" VWFA TE2p """
aparc_label_name = 'TE2p_ROI'#'TE2p_ROI' #'TE2p_ROI' #'inferiortemporal' #'pericalcarine'
#anat_label1 = mne.read_labels_from_annot('fsaverage', parc='aparc',surf_name='white',
#       subjects_dir=fs_dir, regexp=aparc_label_name)
anat_label1 = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', surf_name='white', 
                                        subjects_dir=fs_dir, regexp=aparc_label_name)
vertices_mask_lh = mne.Label.get_vertices_used(anat_label1[0])
vertices_mask_rh = mne.Label.get_vertices_used(anat_label1[1])

#aparc_label_name = 'PH_ROI' #'TE2p_ROI' #'inferiortemporal' #'pericalcarine'
##        anat_label = mne.read_labels_from_annot('fsaverage', parc='aparc',surf_name='white',
##               subjects_dir=fs_dir, regexp=aparc_label_name)
#anat_label2 = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', surf_name='white', 
#                                        subjects_dir=fs_dir, regexp=aparc_label_name)
#vertices_mask_lh2 = mne.Label.get_vertices_used(anat_label2[0])
#vertices_mask_rh2 = mne.Label.get_vertices_used(anat_label2[1])
#
#vertices_mask_lh = np.append(vertices_mask_lh1,vertices_mask_lh2)
#vertices_mask_rh = np.append(vertices_mask_rh1,vertices_mask_rh2)

data1 = np.subtract(np.mean(X1[:,:,:,[1,6]],axis=3), np.mean(X1[:,:,:,[3,8]],axis=3))
data11 = data1[:,:,:]

#stat_fun = partial(mne.stats.ttest_1samp_no_p,sigma=1e-3)
data11 = np.transpose(data11,[2,1,0])
stat_fun = partial(mne.stats.ttest_1samp_no_p)

temp3 = mne.SourceEstimate(np.transpose(stat_fun(data11)), fs_vertices, tmin,
                                 tstep,subject='fsaverage')

##        
brain3_1 = temp3.plot(hemi='lh', subjects_dir=fs_dir,  views='lat', #transparent = True,
          clim=dict(kind='value', lims=[0, 1.5, np.max(temp3.data[vertices_mask_lh,:])]),
          initial_time=0.15) # background='white', size=(800, 600)
brain3_1.add_label(anat_label1[0], borders=True, color='k')
#brain3_1.add_label(anat_label2[0], borders=True, color='k')

brain3_2 = temp3.plot(hemi='rh', subjects_dir=fs_dir,  views='lat',
          clim=dict(kind='value', lims=[0, 1.5, np.max(temp3.data[vertices_mask_rh,:])]),
          initial_time=0.15)
brain3_2.add_label(anat_label1[1], borders=True, color='k')
#brain3_2.add_label(anat_label2[1], borders=True, color='k')

k = 1
#tmin = 0

tp1 = [0.08, 0.13, 0.15, 0.20, 0.30]
tp2 = [0.12, 0.17, 0.19, 0.24, 0.35]
#mask = np.logical_and(times >= tp1[k], times <= tp2[k])
mask = times == 0.15
""" Left """
lh_label = temp3.in_label(anat_label1[0])

data = np.mean(lh_label.data[:,mask],axis=1)
lh_label.data[data < 1.5] = 0.

func_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)

brain3_1.add_label(func_labels, borders=True, color='b')
#brain3_1.save_image('l_TE2p.png')

""" Right """
rh_label = temp3.in_label(anat_label1[1])

#data2 = rh_label.data
#rh_label.data[data2 < 1.5] = 0.

_, func_labels2 = mne.stc_to_label(rh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)

brain3_2.add_label(func_labels2, borders=True, color='b')

vwfa_labels = func_labels.vertices


X11 = X1[vwfa_labels,:,:,:]
X11 = X11[:,:,:,:]
#X11 = np.delete(X11,16,2)
#M = np.mean(np.mean(X1[vwfa_labels,:,:,:],axis=0),axis=1)
M = np.mean(np.mean(X11[:,:,:,:],axis=0),axis=1)

plt.figure(2)
plt.clf()

plt.subplot(2,1,1)
plt.hold(True)

#plt.plot(times, np.mean(M[:,[0]],axis=1),'--',color=c_table[5])
plt.plot(times, np.mean(M[:,[5]],axis=1),'-',color=c_table[5],label='Word')

#plt.plot(times, np.mean(M[:,[1]],axis=1),'--',color=c_table[3])
plt.plot(times, np.mean(M[:,[6]],axis=1),'-',color=c_table[3],label='Word')

#plt.plot(times, np.mean(M[:,[3]],axis=1),'--',color=c_table[1])
plt.plot(times, np.mean(M[:,[8]],axis=1),'-',color=c_table[1],label='Noise')

plt.plot([0.15, 0.15],[0, 4],'-',color='k')

#plt.ylim(0, 4)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
#plt.title('High contrast')
plt.legend(loc='upper left', shadow=True)

plt.subplot(2,1,2)
plt.hold(True)

plt.plot(times, np.mean(M[:,[0]],axis=1),'--',color=c_table[5],label='Word')
#plt.plot(times, np.mean(M[:,[5]],axis=1),'-',color=c_table[5],label='Word')

plt.plot(times, np.mean(M[:,[1]],axis=1),'--',color=c_table[3],label='Word')
#plt.plot(times, np.mean(M[:,[6]],axis=1),'-',color=c_table[3],label='Word')

plt.plot(times, np.mean(M[:,[3]],axis=1),'--',color=c_table[1],label='Noise')
#plt.plot(times, np.mean(M[:,[8]],axis=1),'-',color=c_table[1],label='Noise')

#plt.plot(times, np.mean(M[:,[2]],axis=1),'--',color=c_table[5])
#plt.plot(times, np.mean(M[:,[7]],axis=1),'-',color=c_table[5],label='Word')
#
#plt.plot(times, np.mean(M[:,[4]],axis=1),'--',color=c_table[1])
#plt.plot(times, np.mean(M[:,[9]],axis=1),'-',color=c_table[1],label='Noise')

plt.plot([0.15, 0.15],[0, 4],'-',color='k')

#plt.ylim(0, 4)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
#plt.title('Low contrast')
plt.legend(loc='upper left', shadow=True)

plt.show()



m1 = np.transpose(brs) >= 85

M1 = np.mean(np.mean(X11[:,:,m1,:],axis=0),axis=1)

m2 = np.logical_not(m1)

M2 = np.mean(np.mean(X11[:,:,m2,:],axis=0),axis=1)

 

plt.figure(4)
plt.clf()

plt.subplot(2, 1, 1)
plt.hold(True)

plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[5])
plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[5],label='Word')

#plt.plot(times, np.mean(M1[:,[1]],axis=1),'--',color=c_table[3])
#plt.plot(times, np.mean(M1[:,[6]],axis=1),'-',color=c_table[3],label='Word')

plt.plot(times, M1[:,3],'--',color=c_table[1])
plt.plot(times, M1[:,8],'-',color=c_table[1],label='Noise')

plt.ylabel('%s value' % method)
plt.title('Good readers')
plt.legend(loc='upper left', shadow=True)

plt.subplot(2, 1, 2)
plt.hold(True)

plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[5])
plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[5],label='Word')

#plt.plot(times, np.mean(M2[:,[1]],axis=1),'--',color=c_table[3])
#plt.plot(times, np.mean(M2[:,[6]],axis=1),'-',color=c_table[3],label='Word')

plt.plot(times, M2[:,3],'--',color=c_table[1])
plt.plot(times, M2[:,8],'-',color=c_table[1],label='Noise')

#plt.ylim(0, 4)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.title('Poor readers')
plt.legend(loc='upper left', shadow=True)

plt.show()


#data = {
#        'X11': X11,
#        'brs': brs
#    }
#sio.savemat('R21.mat',{'data':data})

#%%
k = 1
#tmin = 0

tp1 = [0.08, 0.13, 0.15, 0.20, 0.30]
tp2 = [0.12, 0.17, 0.19, 0.24, 0.35]
mask = np.logical_and(times >= tp1[k], times <= tp2[k])

""" Plot scatter """
temp_diff = np.subtract(np.mean(X1[:,:,:,[0,1]],axis=3), np.mean(X1[:,:,:,[3]],axis=3))
data_diff = np.mean(temp_diff[vwfa_labels,:,:], axis = 0)
#data1 = data1.reshape((data1.shape[1],data1.shape[0],data1.shape[2]))
vwfa_response = np.mean(data_diff[mask,:],axis=0)

plt.figure(4)
plt.clf()
ax = plt.subplot()
ax.scatter(brs, vwfa_response, s=30, c='k', alpha=0.5)
for i, txt in enumerate(subs):
    ax.annotate(txt, (brs[i],vwfa_response[i]))

np.corrcoef(vwfa_response,brs)

for iSub, s in enumerate(subs):
    plt.figure(100+iSub)
    plt.clf()
    plt.subplot(1,2,1)
    plt.hold(True)
    plt.plot(times, np.mean(X1[vwfa_labels,:,iSub,0],axis=0), '--', color=c_table[5])
    plt.plot(times, np.mean(X1[vwfa_labels,:,iSub,3],axis=0), '--', color=c_table[1])
    plt.subplot(1,2,2)
    plt.hold(True)
    plt.plot(times, np.mean(X1[vwfa_labels,:,iSub,5],axis=0), '-', color=c_table[5])
    plt.plot(times, np.mean(X1[vwfa_labels,:,iSub,8],axis=0), '-', color=c_table[1])
    plt.title(s)

#%%

#temp3 = mne.SourceEstimate(np.mean(X1[:,:,:,0],axis=2), fs_vertices, tmin,
#                                         tstep,subject='fsaverage')
##vertno_max, time_max = temp1.get_peak(hemi='lh',mode='pos')
#temp3.plot(hemi='lh', subjects_dir=fs_dir,  views='lat', #transparent = True,
#          clim=dict(kind='value', lims=[0.5, 2, 4]),
#          initial_time=0.1)

""" V1 V1 """
#k = 0
#tmin = 0
#
#tp1 = [0.08, 0.13, 0.15, 0.20, 0.30]
#tp2 = [0.12, 0.17, 0.19, 0.24, 0.35]

aparc_label_name = '_V1_ROI' #'inferiortemporal' #'pericalcarine'
#        anat_label = mne.read_labels_from_annot('fsaverage', parc='aparc',surf_name='white',
#               subjects_dir=fs_dir, regexp=aparc_label_name)
anat_label = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', surf_name='white', 
                                        subjects_dir=fs_dir, regexp=aparc_label_name)
vertices_mask_lh = mne.Label.get_vertices_used(anat_label[0])
vertices_mask_rh = mne.Label.get_vertices_used(anat_label[1])

#mask = np.logical_and(times >= tp1[k], times <= tp2[k])
#data1 = temp_avg1[:,mask,:]
#data1 = np.mean(data1, axis = 2)

data1 = np.subtract(np.mean(X1[:,:,:,[0,5]],axis=3), np.mean(X1[:,:,:,[2,7]],axis=3))
data11 = data1[:,:,:]
#data11 = data11[:,mask,:]

#data1 = np.mean(data1,axis=2)

#stat_fun = partial(mne.stats.ttest_1samp_no_p,sigma=1e-3)
data11 = np.transpose(data11,[2,1,0])
stat_fun = partial(mne.stats.ttest_1samp_no_p)

temp3 = mne.SourceEstimate(np.transpose(stat_fun(data11)), fs_vertices, tmin,
                                 tstep,subject='fsaverage')

#vertno_max, time_max = temp3.get_peak(hemi='lh',mode='pos')

##
      
brain3_1 = temp3.plot(hemi='lh', subjects_dir=fs_dir,  views=['lat','ven','med'], #transparent = True,
          clim=dict(kind='value', lims=[2, 3, np.max(temp3.data[vertices_mask_lh,:])]),
          initial_time=0.10) # background='white', size=(800, 600)
#brain3_1.save_movie('test.mp4',time_dilation =8.0,framerate = 30)
brain3_1.add_label(anat_label[0], borders=True, color='k')

brain3_2 = temp3.plot(hemi='rh', subjects_dir=fs_dir,  views='lat',
          clim=dict(kind='value', lims=[2, np.max(temp3.data[vertices_mask_lh,:])*.7, np.max(temp3.data[vertices_mask_lh,:])]),
          initial_time=0.1)
brain3_2.add_label(anat_label[1], borders=True, color='k')

""" Left """
lh_label = temp3.in_label(anat_label[0])

#data = lh_label.data
#
#lh_label.data[data < np.max(data)*0.9] = 0.

func_labels, _ = mne.stc_to_label(lh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)

brain3_1.add_label(func_labels, borders=True, color='b')
#brain3_1.save_image('l_TE2p.png')

""" Right """
rh_label = temp3.in_label(anat_label[1])

#data2 = rh_label.data
#
#rh_label.data[data2 < np.max(data2)*0.9] = 0.

_, func_labels2 = mne.stc_to_label(rh_label, src='fsaverage', smooth=False,
                      subjects_dir=fs_dir, connected=False)

brain3_2.add_label(func_labels2, borders=True, color='b')

v1_labels = func_labels.vertices
v1_labels2 = func_labels2.vertices

M = np.mean(np.mean(X1[v1_labels,:,:,:],axis=0),axis=1)

plt.figure(20)
plt.clf()

plt.hold(True)

plt.plot(times, M[:,0],'--',color=c_table[5])
plt.plot(times, M[:,5],'-',color=c_table[5],label='High')

plt.plot(times, M[:,2],'--',color=c_table[1])
plt.plot(times, M[:,7],'-',color=c_table[1],label='Low')

plt.legend(loc='upper left', shadow=True)

plt.show()

plt.figure(300)
plt.clf()

plt.hold(True)

plt.plot(times, M[:,0],'--',color=c_table[5])
plt.plot(times, M[:,5],'-',color=c_table[5],label='High')

plt.plot(times, M[:,1],'--',color=c_table[1])
plt.plot(times, M[:,6],'-',color=c_table[1],label='med')

plt.plot(times, M[:,3],'--',color=c_table[3])
plt.plot(times, M[:,8],'-',color=c_table[3],label='Noise')

plt.legend(loc='down right', shadow=True)


k = 0
tp1 = [0.08, 0.13, 0.15, 0.20, 0.30]
tp2 = [0.12, 0.17, 0.19, 0.24, 0.35]
mask = np.logical_and(times >= tp1[k], times <= tp2[k])

""" Plot scatter """
temp_diff = np.subtract(np.mean(X1[:,:,:,[0,5]],axis=3), np.mean(X1[:,:,:,[2,7]],axis=3))
data_diff = np.mean(temp_diff[v1_labels,:,:], axis = 0)

#data1 = np.subtract(X1[v1_labels,:,:,[0,5]], X1[v1_labels,:,:,[3,7]])
#data1 = data1.reshape((data1.shape[1],data1.shape[0],data1.shape[2]))
v1_response = np.mean(data_diff[mask,:],axis=0)

fig = plt.figure(40)
plt.clf()
ax1 = plt.subplot()
ax1.scatter(brs, v1_response, s=30, c='k', alpha=0.5)
for i, txt in enumerate(subs):
    ax1.annotate(txt, (brs[i],v1_response[i]))

np.corrcoef(v1_response,brs)

""" Plot individual V1 responses """
for iSub, s in enumerate(subs):
    plt.figure(100+iSub)
    plt.clf()
    plt.subplot(1,2,1)
    plt.hold(True)
    plt.plot(times, np.mean(X1[v1_labels,:,iSub,0],axis=0), '--', color=c_table[5])
    plt.plot(times, np.mean(X1[v1_labels,:,iSub,1],axis=0), '--', color=c_table[3])
    plt.plot(times, np.mean(X1[v1_labels,:,iSub,3],axis=0), '--', color=c_table[1])
    plt.plot([0.1, 0.1],[0, 8],'-',color='k')
    plt.title(s)
    plt.subplot(1,2,2)
    plt.hold(True)
    plt.plot(times, np.mean(X1[v1_labels,:,iSub,5],axis=0), '-', color=c_table[5])
    plt.plot(times, np.mean(X1[v1_labels,:,iSub,6],axis=0), '-', color=c_table[3])
    plt.plot(times, np.mean(X1[v1_labels,:,iSub,8],axis=0), '-', color=c_table[1])
    plt.plot([0.1, 0.1],[0, 8],'-',color='k')
    plt.title(s)


#%%

#connectivity = mne.spatial_tris_connectivity(mne.grade_to_tris(5))
#p_threshold = 0.02
#t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
#
#T_obs, clusters, cluster_p_values, H0 = clu = \
#    mne.stats.spatio_temporal_cluster_1samp_test(data1, connectivity=connectivity, n_jobs=18,
#                                       threshold=t_threshold)
#good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
#
#stc_all_cluster_vis1 = mne.stats.summarize_clusters_stc(clu, tstep=tstep,
#                                             vertices=fs_vertices,
#                                             subject='fsaverage')
#brain1_1 = stc_all_cluster_vis1.plot(hemi='lh', views='lateral',
#                                 subjects_dir=fs_dir,
#                                 time_label='Duration significant (ms)') 
#brain1_2 = stc_all_cluster_vis1.plot(hemi='rh', views='lateral',
#                                 subjects_dir=fs_dir,
#                                 time_label='Duration significant (ms)') 

#""
#aparc_label_name = 'lateraloccipital'#'inferiortemporal'#'fusiform'#'lingual' #fusiform'#'pericalcarine' # lateraloccipital
##    tmin, tmax = 0.080, 0.12
#tmin, tmax = 0.13, 0.18 
##    tmin, tmax = 0.10, 0.15
#
#stc_mean = stc.crop(tmin, tmax).mean()
#label = mne.read_labels_from_annot(subs[0], parc='aparc',surf_name='white',
#               subjects_dir=fs_dir,
#               regexp=aparc_label_name)
#               
#stc_mean_label = stc_mean.in_label(label[0])
#data = np.abs(stc_mean_label.data)
#stc_mean_label.data[data < 0.6 * np.max(data)] = 0.
#               
#func_labels, _ = mne.stc_to_label(stc_mean_label, src=src, smooth=True,
#                              subjects_dir=fs_dir, connected=True)
#func_label = func_labels[0]
#
#anat_label = mne.read_labels_from_annot(subs[0], parc='aparc',
#                                    subjects_dir=fs_dir,
#                                    regexp=aparc_label_name)
#
## extract the anatomical time course for each label
#stc_anat_label = stc.in_label(anat_label[0])
#pca_anat = stc.extract_label_time_course(anat_label[0], src, mode='pca_flip')[0]
#
#stc_func_label = stc.in_label(func_label)
#pca_func = stc.extract_label_time_course(func_label, src, mode='pca_flip')[0]
#
## flip the pca so that the max power between tmin and tmax is positive
#pca_anat *= np.sign(pca_anat[np.argmax(np.abs(pca_anat))])
#pca_func *= np.sign(pca_func[np.argmax(np.abs(pca_anat))])
#
#plt.figure()
#plt.plot(1e3 * stc_anat_label.times, pca_anat, 'k',
#         label='Anatomical %s' % aparc_label_name)
#plt.plot(1e3 * stc_func_label.times, pca_func, 'b',
#         label='Functional %s' % aparc_label_name)
#plt.legend()
#plt.show()
#
#brain = stc_mean.plot(hemi='lh', subjects_dir=fs_dir,
#                      clim=dict(kind='value', lims=[3, 5, 10]))
#brain.show_view('lateral')
#
## show both labels
#brain.add_label(anat_label[0], borders=True, color='k')
#brain.add_label(func_label, borders=True, color='b')
