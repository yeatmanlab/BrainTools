#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 17:33:40 2020

@author: caffarra
"""



import os
import numpy as np
import mne
import glob
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import f_threshold_mway_rm, f_mway_rm, fdr_correction

root='/mnt/scratch/NLR_MEG'

session1 = ['102_rs160618','103_ac150609','105_bb150713','110_hh160608','127_am151022',
       '130_rw151221','132_wp160919','133_ml151124','145_ac160621','150_mg160606',
       '151_rd160620','152_tc160422','160_ek160627','161_ak160627','163_lf160707',
       '164_sf160707','170_gm160613','172_th160614','174_hs160620','179_gm160701',
       '180_zd160621','187_nb161017','201_gs150818','203_am150831',
       '204_am150829','205_ac151123','206_lm151119','207_ah160608','211_lb160617',
       'nlr_gb310170614','nlr_kb218170619','nlr_jb423170620','nlr_gb267170620','nlr_jb420170621',
       'nlr_hb275170622','197_bk170622','nlr_gb355170606','nlr_gb387170608','nlr_hb205170825',
       'nlr_ib217170831','nlr_ib319170825','nlr_jb227170811','nlr_jb486170803','nlr_kb396170808',
       'nlr_ib357170912']
twre_index = [87,93,108,66,116,85,110,71,84,92,87,86,63,81,60,55,71,63,68,67,64,127,79,
              73,59,84,79,91,57,67,77,57,80,53,72,58,85,79,116,117,107,78,66,101,67]
#%%
All_R = [session1[i] for i in all_subject]
Good_R = [session1[i] for i in good_readers]
Poor_R = [session1[i] for i in poor_readers]
read_score = [twre_index[i] for i in all_subject]
read_score_G = [twre_index[i] for i in good_readers]
read_score_P = [twre_index[i] for i in poor_readers]

#All_R.sort()
#Good_R.sort()
#Poor_R.sort()

picks_F= ['MEG0123', 'MEG0122', 'MEG0343','MEG0342', 'MEG0212', 'MEG0213', 'MEG0133', 'MEG0132', 'MEG1513', 'MEG1512', 'MEG0242','MEG0243']
picks_T= ['MEG1513', 'MEG1512', 'MEG0242','MEG0243', 'MEG1523', 'MEG1522', 'MEG1612', 'MEG1613', 'MEG1722','MEG1723', 'MEG1642', 'MEG1643']
picks_V= ['MEG0132', 'MEG0133', 'MEG0142', 'MEG0143', 'MEG0242', 'MEG0243', 'MEG1512', 'MEG1513', \
          'MEG1522', 'MEG1523', 'MEG1542', 'MEG1543', 'MEG0212', 'MEG0213', \
          'MEG0232', 'MEG0233', 'MEG0442', 'MEG0443', 'MEG0742', 'MEG0743', 'MEG1822', 'MEG1823', \
          'MEG1812', 'MEG1813', 'MEG1622', 'MEG1623', 'MEG1612', 'MEG1613']
picks_L= ['MEG0132', 'MEG0133', 'MEG0142', 'MEG0143', 'MEG0232', 'MEG0233','MEG0242', 'MEG0243', \
          'MEG0442', 'MEG0443', 'MEG1512', 'MEG1513', 'MEG1522', 'MEG1523','MEG1542', 'MEG1543', \
          'MEG1612', 'MEG1613']
picks_R = ['MEG1322','MEG1323','MEG1332','MEG1333','MEG1342','MEG1343','MEG1432','MEG1433', \
           'MEG1442','MEG1443','MEG2612','MEG2613','MEG2622','MEG2623','MEG2632','MEG2633']
#picks_fix= ['MEG0112', 'MEG0113', 'MEG0132', 'MEG0133', 'MEG0142', 'MEG0143','MEG1512', 'MEG1513', \
#            'MEG1542', 'MEG1543','MEG1522', 'MEG1523','MEG0212', 'MEG0213','MEG0242', 'MEG0243', \
#            'MEG0232', 'MEG0233','MEG1612', 'MEG1613']
#%% Select the participant sample
Exp = All_R
colors = {"Word": "crimson", "Noise": "black"}

Ep_path= '/mnt/scratch/NLR_MEG/'
os.chdir(Ep_path)  # go into the tsss dir 
Att_LN_evoked_list=list()
Att_HN_evoked_list=list()
Lex_LN_evoked_list=list()
Lex_HN_evoked_list=list()

# read in evoked responses
for x in range(0,len(Exp)):
    Ep_path= '/mnt/scratch/NLR_MEG/'+Exp[x]+'/epochs/'
    os.chdir(Ep_path)  # go into the tsss dir 
    file = 'All_40-sss_'+Exp[x]+'-epo.fif'
    epochs = mne.read_epochs(file, verbose=True)
    Att_LN_evoked = epochs['word_c254_p20_dot'].average().pick(['grad']) # n Preproc 5 no need of i .shift_time(-0.239,relative=False) # select epochs from a group of conditions
    Att_HN_evoked = epochs['word_c254_p80_dot'].average().pick(['grad']) 
    Lex_LN_evoked = epochs['word_c254_p20_word'].average().pick(['grad']) 
    Lex_HN_evoked = epochs['word_c254_p80_word'].average().pick(['grad']) 
 
    Att_LN_evoked_list.append(Att_LN_evoked)
    Att_HN_evoked_list.append(Att_HN_evoked)
    Lex_LN_evoked_list.append(Lex_LN_evoked)
    Lex_HN_evoked_list.append(Lex_HN_evoked)
    
# Figure 2
mne.viz.plot_compare_evokeds(dict(Word=Lex_LN_evoked_list, Noise=Lex_HN_evoked_list), picks=picks_L, colors=colors, ylim=dict(grad=[-15, 15]), combine= 'mean', ci= True, legend='lower right', show_sensors='upper right', title="Left frontal sensors")
mne.viz.plot_compare_evokeds(dict(Word=Lex_LN_evoked_list, Noise=Lex_HN_evoked_list), picks=picks_R, colors=colors, ylim=dict(grad=[-15, 15]), combine= 'mean', ci= True, legend='lower right', show_sensors='upper right', title="Left temporal sensors")
mne.viz.plot_compare_evokeds(dict(Word=Att_LN_evoked_list, Noise=Att_HN_evoked_list), picks=picks_L, colors=colors, ylim=dict(grad=[-15, 15]), combine= 'mean', ci= True, legend='lower right', show_sensors='upper right', title="Left temporal sensors")
mne.viz.plot_compare_evokeds(dict(Word=Att_LN_evoked_list, Noise=Att_HN_evoked_list), picks=picks_R, colors=colors, ylim=dict(grad=[-15, 15]), combine= 'mean', ci= True, legend='lower right', show_sensors='upper right', title="Left temporal sensors")

t = mne.combine_evoked(Att_LN_evoked_list, weights = 'nave')
mne.viz.plot_evoked(t)

#%%
sensors = picks_L

tmin = 0.283
tmax = 0.383
Att_LN_rms = np.empty((int(len(sensors)/2), 61,len(Exp)))
Att_HN_rms = np.empty((int(len(sensors)/2), 61,len(Exp)))
Lex_LN_rms = np.empty((int(len(sensors)/2), 61,len(Exp)))
Lex_HN_rms = np.empty((int(len(sensors)/2), 61,len(Exp)))

for x in range(0,len(Exp)):
    Ep_path= '/mnt/scratch/NLR_MEG/'+Exp[x]+'/epochs/'
    os.chdir(Ep_path)  # go into the tsss dir 
    file = 'All_40-sss_'+Exp[x]+'-epo.fif'
    epochs = mne.read_epochs(file, verbose=True)
    Att_LN_evoked = epochs['word_c254_p20_dot'].average().pick(['grad']) # n Preproc 5 no need of i .shift_time(-0.239,relative=False) # select epochs from a group of conditions
    Att_HN_evoked = epochs['word_c254_p80_dot'].average().pick(['grad']) 
    Lex_LN_evoked = epochs['word_c254_p20_word'].average().pick(['grad']) 
    Lex_HN_evoked = epochs['word_c254_p80_word'].average().pick(['grad'])
    Att_LN_cropped = Att_LN_evoked.pick(sensors).crop(tmin, tmax)
    Att_HN_cropped = Att_HN_evoked.pick(sensors).crop(tmin, tmax)
    Lex_LN_cropped = Lex_LN_evoked.pick(sensors).crop(tmin, tmax)
    Lex_HN_cropped = Lex_HN_evoked.pick(sensors).crop(tmin, tmax)
    
    for iii in range(0,int(len(sensors)/2)):
        Att_LN_rms[iii,:,x] = np.sqrt((np.square(Att_LN_cropped.data[2*(iii-1),:])+np.square(Att_LN_cropped.data[2*(iii-1)+1,:]))/2)
#        Att_LN_rms[1,:,x] = np.sqrt((np.square(Att_LN_cropped.data[2,:])+np.square(Att_LN_cropped.data[3,:]))/2)
#        Att_LN_rms[2,:,x] = np.sqrt((np.square(Att_LN_cropped.data[4,:])+np.square(Att_LN_cropped.data[5,:]))/2)
#        Att_LN_rms[3,:,x] = np.sqrt((np.square(Att_LN_cropped.data[6,:])+np.square(Att_LN_cropped.data[7,:]))/2)
        Att_HN_rms[iii,:,x] = np.sqrt((np.square(Att_HN_cropped.data[2*(iii-1),:])+np.square(Att_HN_cropped.data[2*(iii-1),:]))/2)
#        Att_HN_rms[1,:,x] = np.sqrt((np.square(Att_HN_cropped.data[2,:])+np.square(Att_HN_cropped.data[3,:]))/2)
#        Att_HN_rms[2,:,x] = np.sqrt((np.square(Att_HN_cropped.data[4,:])+np.square(Att_HN_cropped.data[5,:]))/2)
#        Att_HN_rms[3,:,x] = np.sqrt((np.square(Att_HN_cropped.data[6,:])+np.square(Att_HN_cropped.data[7,:]))/2)
        Lex_LN_rms[iii,:,x] = np.sqrt((np.square(Lex_LN_cropped.data[2*(iii-1),:])+np.square(Lex_LN_cropped.data[2*(iii-1),:]))/2)
#        Lex_LN_rms[1,:,x] = np.sqrt((np.square(Lex_LN_cropped.data[2,:])+np.square(Lex_LN_cropped.data[3,:]))/2)
#        Lex_LN_rms[2,:,x] = np.sqrt((np.square(Lex_LN_cropped.data[4,:])+np.square(Lex_LN_cropped.data[5,:]))/2)
#        Lex_LN_rms[3,:,x] = np.sqrt((np.square(Lex_LN_cropped.data[6,:])+np.square(Lex_LN_cropped.data[7,:]))/2)
        Lex_HN_rms[iii,:,x] = np.sqrt((np.square(Lex_HN_cropped.data[2*(iii-1),:])+np.square(Lex_HN_cropped.data[2*(iii-1),:]))/2)
#        Lex_HN_rms[1,:,x] = np.sqrt((np.square(Lex_HN_cropped.data[2,:])+np.square(Lex_HN_cropped.data[3,:]))/2)
#        Lex_HN_rms[2,:,x] = np.sqrt((np.square(Lex_HN_cropped.data[4,:])+np.square(Lex_HN_cropped.data[5,:]))/2)
#        Lex_HN_rms[3,:,x] = np.sqrt((np.square(Lex_HN_cropped.data[6,:])+np.square(Lex_HN_cropped.data[7,:]))/2)
    
plt.figure()
plt.clf()

ax = plt.subplot()

lex = np.mean(np.mean(Lex_LN_rms,axis=1),axis=0) - np.mean(np.mean(Lex_HN_rms,axis=1),axis=0)
fix = np.mean(np.mean(Att_LN_rms,axis=1),axis=0) - np.mean(np.mean(Att_HN_rms,axis=1),axis=0)

fit = np.polyfit(lex, read_score, deg=1)
ax.plot(lex, fit[0] * lex + fit[1], color=[0,0,0])
ax.plot(lex, read_score, 'o', markerfacecolor=[.5, .5, .5], markeredgecolor=[1,1,1], markersize=10)

r, p = stats.pearsonr(read_score,fix)
print (r, p)

r, p = stats.pearsonr(lex,fix)
print (r, p)

#%%
os.chdir(raw_dir)
os.chdir('figures')
plt.savefig('Sensor_corr_lex400.png',dpi=600,papertype='letter',format='png')
plt.savefig('Sensor_corr_lex400.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')  
#%%
HN= mne.combine_evoked(Att_HN_evoked_list, weights = 'nave')
LN= mne.combine_evoked(Att_LN_evoked_list, weights = 'nave')
evoked = mne.combine_evoked([LN,HN], weights = [1, -1])
all_times = np.arange(0.1, 0.8, 0.1)
evoked.plot_topomap(0.3, ch_type='grad', show_names=False, colorbar=True, time_unit='s',
                    ncols=8, nrows='auto')
#%% Good readers
Exp = Good_R
Ep_path= '/mnt/scratch/NLR_MEG/'
os.chdir(Ep_path)  # go into the tsss dir 
Att_LN_evoked_list_GR=list()
Att_HN_evoked_list_GR=list()
Lex_LN_evoked_list_GR=list()
Lex_HN_evoked_list_GR=list()

# read in evoked responses
for x in range(0,len(Exp)):
    Ep_path= '/mnt/scratch/NLR_MEG/'+Exp[x]+'/epochs/'
    os.chdir(Ep_path)  # go into the tsss dir 
    file = 'All_40-sss_'+Exp[x]+'-epo.fif'
    epochs = mne.read_epochs(file, verbose=True)
    Att_LN_evoked = epochs['word_c254_p20_dot'].average().pick(['grad'])  # n Preproc 5 no need of i .shift_time(-0.239,relative=False) # select epochs from a group of conditions
    Att_HN_evoked = epochs['word_c254_p80_dot'].average().pick(['grad']) 
    Lex_LN_evoked = epochs['word_c254_p20_word'].average().pick(['grad']) 
    Lex_HN_evoked = epochs['word_c254_p80_word'].average().pick(['grad']) 
 
    Att_LN_evoked_list_GR.append(Att_LN_evoked)
    Att_HN_evoked_list_GR.append(Att_HN_evoked)
    Lex_LN_evoked_list_GR.append(Lex_LN_evoked)
    Lex_HN_evoked_list_GR.append(Lex_HN_evoked)
    
# Lexical: Group
mne.viz.plot_compare_evokeds(dict(Word=Lex_LN_evoked_list_GR, Noise=Lex_HN_evoked_list_GR), picks=picks_L, colors=colors, ylim=dict(grad=[-15, 15]), combine= 'mean', ci= True, legend='lower right', show_sensors='upper right', title="Typical readers")
#mne.viz.plot_compare_evokeds(dict(Word=Lex_LN_evoked_list_GR, Noise=Lex_HN_evoked_list_GR), picks=picks_R, colors=colors, ylim=dict(grad=[-15, 15]), combine= 'mean', ci= True, legend='lower right', show_sensors='upper right', title="Typical readers")
mne.viz.plot_compare_evokeds(dict(Word=Att_LN_evoked_list_GR, Noise=Att_HN_evoked_list_GR), picks=picks_L, colors=colors, ylim=dict(grad=[-15, 15]), combine= 'mean', ci= True, legend='lower right', show_sensors='upper right', title="Left temporal sensors")
#mne.viz.plot_compare_evokeds(dict(Word=Att_LN_evoked_list_GR, Noise=Att_HN_evoked_list_GR), picks=picks_R, colors=colors, ylim=dict(grad=[-15, 15]), combine= 'mean', ci= True, legend='lower right', show_sensors='upper right', title="Typical readers")

t = mne.combine_evoked(Att_LN_evoked_list_GR, weights = 'nave')
mne.viz.plot_evoked(t)

from mne.stats import permutation_cluster_1samp_test
# condition to be compared
erf_1_list = Lex_LN_evoked_list_GR
erf_2_list = Lex_HN_evoked_list_GR
# get starting empty matrices
erf_1= np.zeros((len(Exp),481)) 
erf_2= np.zeros((len(Exp),481))

for s in range(0,len(Exp)):
    erf_1_l= erf_1_list[s].copy().pick_channels(sensors).crop(tmin= 0, tmax= 0.8) # get the channel subset
    erf_2_l= erf_2_list[s].copy().pick_channels(sensors).crop(tmin= 0, tmax= 0.8) # get the channel subset 
    erf_1[s]= np.mean(erf_1_l._data, axis =0) # by column avg for the first column -chn
    erf_2[s]= np.mean(erf_2_l._data, axis =0) # by column avg for the first column -chn

XX = erf_1 - erf_2
threshold =5.0
p_accept = 0.01
t_threshold = -stats.distributions.t.ppf(p_accept / 2., len(All_R) - 1)
T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_1samp_test(XX,
                             n_permutations=5000, threshold=t_threshold, tail=0)

# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(range(0,len(clusters)), cluster_p_values):
    if p_val <= p_accept:
        print (clusters[c]) # get the bounderies of the ssignificant cluster
        T_obs_plot[c] = T_obs[c]
        
# the bounderies can be translated into time using this vector (e.g. times[149])
times = 1e3 * erf_1_l.times  # change unit to ms
        
mne.viz.plot_compare_evokeds(dict(Word=erf_1_list, Noise=erf_2_list), picks=sensors, colors=colors, \
                             ylim=dict(grad=[-15, 10]), combine= 'mean', ci= True, legend='lower right', title="Left temporal sensors") #show_sensors='upper right', 
plt.plot([0.226,0.396], [-14,-14])
#plt.plot([0.445,0.558], [-14,-14])
#%%
os.chdir(raw_dir)
os.chdir('figures')
plt.savefig('Sensor_left_lex_G.png',dpi=600,papertype='letter',format='png')
plt.savefig('Sensor_left_lex_G.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..')  
#%% Poor readers
Exp = Poor_R
Ep_path= '/mnt/scratch/NLR_MEG/'
os.chdir(Ep_path)  # go into the tsss dir 
Att_LN_evoked_list_BR=list()
Att_HN_evoked_list_BR=list()
Lex_LN_evoked_list_BR=list()
Lex_HN_evoked_list_BR=list()

# read in evoked responses
for x in range(0,len(Exp)):
    Ep_path= '/mnt/scratch/NLR_MEG/'+Exp[x]+'/epochs/'
    os.chdir(Ep_path)  # go into the tsss dir 
    file = 'All_40-sss_'+Exp[x]+'-epo.fif'
    epochs = mne.read_epochs(file, verbose=True)
    Att_LN_evoked = epochs['word_c254_p20_dot'].average().pick(['grad'])  # n Preproc 5 no need of i .shift_time(-0.239,relative=False) # select epochs from a group of conditions
    Att_HN_evoked = epochs['word_c254_p80_dot'].average().pick(['grad']) 
    Lex_LN_evoked = epochs['word_c254_p20_word'].average().pick(['grad']) 
    Lex_HN_evoked = epochs['word_c254_p80_word'].average().pick(['grad']) 
 
    Att_LN_evoked_list_BR.append(Att_LN_evoked)
    Att_HN_evoked_list_BR.append(Att_HN_evoked)
    Lex_LN_evoked_list_BR.append(Lex_LN_evoked)
    Lex_HN_evoked_list_BR.append(Lex_HN_evoked)
    
mne.viz.plot_compare_evokeds(dict(Word=Lex_LN_evoked_list_BR, Noise=Lex_HN_evoked_list_BR), picks=picks_L, colors=colors, ylim=dict(grad=[-15, 15]), combine= 'mean', ci= True, legend='lower right', show_sensors='upper right', title="Struggling readers")
#mne.viz.plot_compare_evokeds(dict(Word=Lex_LN_evoked_list_BR, Noise=Lex_HN_evoked_list_BR), picks=picks_R, colors=colors, ylim=dict(grad=[-15, 15]), combine= 'mean', ci= True, legend='lower right', show_sensors='upper right', title="Struggling readers")
mne.viz.plot_compare_evokeds(dict(Word=Att_LN_evoked_list_BR, Noise=Att_HN_evoked_list_BR), picks=picks_L, colors=colors, ylim=dict(grad=[-15, 15]), combine= 'mean', ci= True, legend='lower right', show_sensors='upper right', title="Left temporal sensors")
#mne.viz.plot_compare_evokeds(dict(Word=Att_LN_evoked_list_BR, Noise=Att_HN_evoked_list_BR), picks=picks_R, colors=colors, ylim=dict(grad=[-15, 15]), combine= 'mean', ci= True, legend='lower right', show_sensors='upper right', title="Struggling readers")

#t = mne.combine_evoked(Att_LN_evoked_list_BR, weights = 'nave')
#mne.viz.plot_evoked(t)

from mne.stats import permutation_cluster_1samp_test
# condition to be compared
erf_1_list = Att_LN_evoked_list_BR
erf_2_list = Att_HN_evoked_list_BR
# get starting empty matrices
erf_1= np.zeros((len(Exp),481)) 
erf_2= np.zeros((len(Exp),481))

for s in range(0,len(Exp)):
    erf_1_l= erf_1_list[s].copy().pick_channels(sensors).crop(tmin= 0, tmax= 0.8) # get the channel subset
    erf_2_l= erf_2_list[s].copy().pick_channels(sensors).crop(tmin= 0, tmax= 0.8) # get the channel subset 
    erf_1[s]= np.mean(erf_1_l._data, axis =0) # by column avg for the first column -chn
    erf_2[s]= np.mean(erf_2_l._data, axis =0) # by column avg for the first column -chn

XX = erf_1 - erf_2
threshold =5.0
p_accept = 0.01
t_threshold = -stats.distributions.t.ppf(p_accept / 2., len(All_R) - 1)
T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_1samp_test(XX,
                             n_permutations=5000, threshold=t_threshold, tail=0)

# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(range(0,len(clusters)), cluster_p_values):
    if p_val <= p_accept:
        print (clusters[c]) # get the bounderies of the ssignificant cluster
        T_obs_plot[c] = T_obs[c]
        
# the bounderies can be translated into time using this vector (e.g. times[149])
times = 1e3 * erf_1_l.times  # change unit to ms
        
mne.viz.plot_compare_evokeds(dict(Word=erf_1_list, Noise=erf_2_list), picks=sensors, colors=colors, \
                             ylim=dict(grad=[-15, 10]), combine= 'mean', ci= True, legend='lower right', title="Left temporal sensors") #show_sensors='upper right', 
#plt.plot([0.310,0.376], [-14,-14])
#plt.plot([0.445,0.558], [-14,-14])
#%%
os.chdir(raw_dir)
os.chdir('figures')
plt.savefig('Sensor_left_fix_P.png',dpi=600,papertype='letter',format='png')
plt.savefig('Sensor_left_fix_P.pdf',dpi=600,papertype='letter',format='pdf')
os.chdir('..') 
#%%
"""Compare between tasks"""
mne.viz.plot_compare_evokeds(dict(Lex=Lex_LN_evoked_list_BR, Att=Att_LN_evoked_list_BR), picks=picks_V, ylim=dict(grad=[-15, 15]), combine= 'mean', ci= True, legend='lower right', show_sensors='upper right', title="Struggling readers")
mne.viz.plot_compare_evokeds(dict(Lex=Lex_LN_evoked_list_GR, Att=Att_LN_evoked_list_GR), picks=picks_V, ylim=dict(grad=[-15, 15]), combine= 'mean', ci= True, legend='lower right', show_sensors='upper right', title="Typical readers")
#%%
# get the ERF effects
HN= mne.grand_average(Lex_HN_evoked_list)
LN= mne.grand_average(Lex_LN_evoked_list)
word_diff = mne.combine_evoked([LN, HN], weights = [1, -1])

# plot topo
word_diff.plot_topo()

times = np.linspace(0.1, 0.8, 8)
times = 0.32
word_diff.plot_topomap(ch_type='grad', times=times, colorbar=True, show_names=True)
word_diff.plot_joint()

#word_diff.pick_types(meg='grad')
#max_t = word_diff.get_peak()[1]
#
#tstep = np.load('session1_tstep.npy')
#tmin = -0.1
#fs_vertices = [np.arange(10242)] * 2
#temp3 = mne.SourceEstimate(np.mean(X11[:,:,:,5]-X11[:,:,:,8]),axis=2), fs_vertices, tmin, tstep, subject='fsaverage')
#
#brain3_1 = temp3.plot(hemi='lh', subjects_dir=fs_dir, views = ['lat','ven','med'], initial_time=0.26, #['lat','ven','med']
#           clim=dict(kind='value', lims=[2, 2.5, 5])) #clim=dict(kind='value', lims=[2, t_threshold, 7]), size=(800,800))

