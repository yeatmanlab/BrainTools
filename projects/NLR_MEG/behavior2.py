#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:46:56 2017

@author: sjjoo
"""

#import sys
#import mne
import matplotlib.pyplot as plt
#import imageio
#from mne.utils import run_subprocess, logger
import os
#from os import path as op
#import copy
#import shutil
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from scipy import stats as stats

raw_dir = '/mnt/scratch/NLR_MEG3'

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
session2 = ['102_rs160815','110_hh160809','145_ac160823','150_mg160825',
       '152_tc160623','160_ek160915','161_ak160916','162_ef160829','163_lf160920',
       '164_sf160920','170_gm160822','172_th160825','174_hs160829','179_gm160913',
       '180_zd160826','201_gs150925',
       '204_am151120','205_ac160202','207_ah160809','210_sb160822','211_lb160823',
       'nlr_gb310170829','nlr_kb218170829','nlr_gb267170911','nlr_jb420170828','nlr_hb275170828','nlr_gb355170907']

#subIndex1 = np.nonzero(np.in1d(subs,subs2))[0]
#subIndex2 = np.empty([1,len(subIndex1)],dtype=int)[0]
#for i in range(0,len(subIndex1)):
#    subIndex2[i] = np.nonzero(np.in1d(subs2,subs[subIndex1[i]]))[0]

twre_index = [87,93,108,66,116,85,110,71,84,92,87,86,63,81,60,55,71,63,68,67,64,127,79,
              73,59,84,79,91,57,67,77,57,80,53,72,58,85,79,116,117,107,78,66,101,67]
twre_index = np.array(twre_index)

brs = [87,102,108,78,122,91,121,77,91,93,93,88,75,90,66,59,81,84,81,72,71,121,
       81,75,66,90,93,101,56,78,83,69,88,60,88,73,82,81,115,127,124,88,68,110,96]
brs = np.array(brs)

#twre_index1 = twre_index[subIndex1]
twre_index2_all = [90,76,94,115,
               85,75,82,64,75,
               63,83,77,84,75,
               68,79,
               62,90,105,75,71,
               69,83,76,62,73,94]

twre_index2_all = np.array(twre_index2_all)
#twre_index2 = twre_index2_all[subIndex2]

#brs1 = brs[subIndex1]
brs2_all = [98,88,102,110,99,91,88,79,105,86,81,88,89,77,83,81,86,98,116,104,86,90,91,97,57,99,102]
brs2_all = np.array(brs2_all)
#brs2 = brs2_all[subIndex2]

#twre_diff = np.subtract(twre_index2,twre_index1)
#brs_diff = np.subtract(brs2,brs1)

swe_raw = [62, 76, 74, 42, 75, 67, 76, 21, 54, 35, 21, 61, 45, 48, 17, 11, 70, 19, 10, 57,
           12, 86, 53, 51, 13, 28, 54, 25, 27, 10, 66, 18, 18, 20, 37, 23, 17, 36, 79, 82,
           74, 64, 42, 78, 35]
swe_raw = np.array(swe_raw)

age = [125.6885, 132.9501, 122.0434, 138.4349, 97.6347, 138.1420, 108.2457, 98.0631, 105.8147, 89.9132,
       87.6465, 131.8660, 123.7174, 95.959, 112.416, 133.8042, 152.4639, 103.4823, 89.8475, 138.4020,
       93.8568, 117.0814, 123.6202, 122.9304, 109.1656, 90.6058,
       111.9593,86.0381,147.2063,95.8699,148.0802,122.5896,88.7162,123.0495,110.6645,105.3069,88.9143,95.2879,106.2852,
       122.2915,114.4389,136.1496,128.6246,137.9216,122.7528]
age = np.divide(age, 12)

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

reading_thresh = 85 #np.median(swe_raw) #90
#m1 = np.logical_and(np.transpose(twre_index) >= 80, np.transpose(age) <= 13)
#m2 = np.logical_and(np.transpose(twre_index) < 80, np.transpose(age) <= 13)

m1 = np.logical_and(np.transpose(twre_index) >= reading_thresh, np.transpose(age) <= 13)
m2 = np.logical_and(np.transpose(twre_index) < reading_thresh, np.transpose(age) <= 13)

#m1 = np.logical_and(np.transpose(swe_raw) >= np.mean(swe_raw), np.transpose(age) <= 13)
#m2 = np.logical_and(np.transpose(swe_raw) < np.mean(swe_raw), np.transpose(age) <= 13)

#m4 = np.logical_and(np.transpose(twre_index) >= 80, np.transpose(twre_index) < 90)


good_readers = np.where(m1)[0]
poor_readers = np.where(m2)[0]

#%%
"""
No Button Responses:
174_hs160620
nlr_hb205170825
nlr_jb227170811
"""

d_prime = np.empty((1,len(session1)))[0]
hit_rate = np.empty((1,len(session1)))[0]
falsealarm_rate = np.empty((1,len(session1)))[0]
overall_pseudo_rt = np.empty((1,len(session1)))[0]
overall_pseudo_rt1 = np.empty((1,len(session1)))[0]
overall_pseudo_rt2 = np.empty((1,len(session1)))[0]
overall_pseudo_rt3 = np.empty((1,len(session1)))[0]
overall_word_rt = np.empty((1,len(session1)))[0]
for ttt, s in enumerate(session1):
    os.chdir(os.path.join(raw_dir,s))
    os.chdir('lists')
    
    if s == '164_sf160707':
        run_names = ['1', '2', '3', '5', '6']
        n_lexical = 2
    elif s == '170_gm160613':
        run_names = ['1', '2', '3', '5', '6']
        n_lexical = 2
    else:
        run_names = ['1', '2', '3', '4', '5', '6']
        n_lexical = 3
    
    if s == '103_ac150609':
        key_string = '232'
    elif s == '130_rw151221' or s == 'nlr_gb310170614' or s == '211_lb160617' or s == 'nlr_gb387170608' \
        or s == 'nlr_kb218170619' or s == 'nlr_jb423170620' or s == 'nlr_gb267170620' \
        or s == 'nlr_jb420170621' or s == 'nlr_hb275170622' or s == '197_bk170622' \
        or s == 'nlr_gb355170606' or s == 'nlr_ib217170831' or s == 'nlr_ib319170825' \
        or s == 'nlr_jb486170803' or s == 'nlr_kb396170808' or s == 'nlr_ib357170912':
        key_string = '328'
    else:
        key_string = '216'
    
    if s != '174_hs160620' and s != 'nlr_hb205170825' and s != 'nlr_jb227170811':
        all_cond = []
        all_rt = []
        for n, r_name in enumerate(run_names):
            if np.mod(int(r_name),2) == 0:
                fn = 'ALL_' + s + '_' + r_name + '-eve.lst'
                mat = []
                for line in open(fn).readlines():
                    mat.append(line.split())
                
                time_point = [float(row[0])/1200 for row in mat]
                time_point = np.subtract(time_point, time_point[0])
                time_point = np.array(time_point)
                
                event_id = [row[2] for row in mat]
                event_id = np.array(event_id)
                
                temp_press = np.where(event_id == key_string)[0]
    
                mask = np.zeros((1,len(event_id)))[0]
                for iii in range(1,len(event_id)):
                    if event_id[iii] == key_string and event_id[iii-1] == key_string:
                        mask[iii] = 1
                button_press = []
                for kkk,fjdsjflkdsjlf in enumerate(temp_press):
                    if mask[temp_press[kkk]] == 0.0:
                        button_press.append(temp_press[kkk])       
                button_time = time_point[button_press]
                cond = event_id[np.subtract(button_press,np.ones((1,len(button_press)),dtype=int))][0]
                all_cond.append(cond)
                cond_time = time_point[np.subtract(button_press,np.ones((1,len(button_press)),dtype=int))]
                rt = button_time - cond_time
                all_rt.append(rt[0])
        pseudo_rt = []
        pseudo_rt1 = []
        pseudo_rt2 = []
        pseudo_rt3 = []
        word_rt = []
        for n in np.arange(0,len(all_cond)):
            for k in np.arange(0,len(all_cond[n])):
                if all_cond[n][k] == '206':
                    pseudo_rt1.append(all_rt[n][k])
                elif all_cond[n][k] == '207':
                    pseudo_rt2.append(all_rt[n][k])
                elif all_cond[n][k] == '208':
                    pseudo_rt3.append(all_rt[n][k])
                else:
                    word_rt.append(all_rt[n][k])
        pseudo_rt.extend(pseudo_rt1)
        pseudo_rt.extend(pseudo_rt2)
        pseudo_rt.extend(pseudo_rt3)
        
        overall_pseudo_rt[ttt] = np.median(pseudo_rt)
        overall_pseudo_rt1[ttt] = np.median(pseudo_rt1)
        overall_pseudo_rt2[ttt] = np.median(pseudo_rt2)
        overall_pseudo_rt3[ttt] = np.median(pseudo_rt3)
        overall_word_rt[ttt] = np.median(word_rt)
        
        hit_rate[ttt] = len(pseudo_rt) / (21.*len(all_cond)) # 7 repetitions 3 different pseudoword conditions
        falsealarm_rate[ttt] = len(word_rt) / (20.*5*n_lexical)
        d_prime[ttt] = norm.ppf(hit_rate[ttt]) - norm.ppf( [ttt])
    
    if s == '174_hs160620' or s == 'nlr_hb205170825' or s == 'nlr_jb227170811':
        d_prime[ttt] = np.nan
        hit_rate[ttt] = np.nan
        falsealarm_rate[ttt] = np.nan
        overall_pseudo_rt[ttt] = np.nan
        overall_pseudo_rt1[ttt] = np.nan
        overall_pseudo_rt2[ttt] = np.nan
        overall_pseudo_rt3[ttt] = np.nan

#%%        
plt.figure(1)
plt.clf()
plt.hold(True)
plt.bar(0.65, np.nanmean(overall_pseudo_rt1[good_readers]), width=0.2, \
        yerr = np.nanstd(overall_pseudo_rt1[good_readers])/np.sqrt(np.sum(~np.isnan(overall_pseudo_rt1[good_readers]))), \
        color=c_table[5], ecolor = [0,0,0])
plt.bar(0.9, np.nanmean(overall_pseudo_rt2[good_readers]), width=0.2, \
        yerr = np.nanstd(overall_pseudo_rt2[good_readers])/np.sqrt(np.sum(~np.isnan(overall_pseudo_rt2[good_readers]))), \
        color=c_table[3], ecolor = [0,0,0])
plt.bar(1.15, np.nanmean(overall_pseudo_rt3[good_readers]), width=0.2, \
        yerr = np.nanstd(overall_pseudo_rt3[good_readers])/np.sqrt(np.sum(~np.isnan(overall_pseudo_rt3[good_readers]))), \
        color=[0.4, 0.4, 0.4], ecolor = [0,0,0])

plt.bar(1.65, np.nanmean(overall_pseudo_rt1[poor_readers]), width=0.2, \
        yerr = np.nanstd(overall_pseudo_rt1[poor_readers])/np.sqrt(np.sum(~np.isnan(overall_pseudo_rt1[poor_readers]))), \
        color=c_table[5], ecolor = [0,0,0])
plt.bar(1.9, np.nanmean(overall_pseudo_rt2[poor_readers]), width=0.2, \
        yerr = np.nanstd(overall_pseudo_rt2[poor_readers])/np.sqrt(np.sum(~np.isnan(overall_pseudo_rt2[poor_readers]))), \
        color=c_table[3], ecolor = [0,0,0])
plt.bar(2.15, np.nanmean(overall_pseudo_rt3[poor_readers]), width=0.2, \
        yerr = np.nanstd(overall_pseudo_rt3[poor_readers])/np.sqrt(np.sum(~np.isnan(overall_pseudo_rt3[poor_readers]))), \
        color=[0.4, 0.4, 0.4], ecolor = [0,0,0])
plt.xticks([1,2],('Typical readers','Struggling readers'))
plt.xlim([0.5,2.5])
plt.ylim([0.5,1])
plt.ylabel('Reaction time (s)')
plt.show()

plt.figure(2)
plt.clf()
plt.hold(True)
plt.bar(0.9, np.nanmean(d_prime[good_readers]), width=0.2, \
        yerr = np.nanstd(d_prime[good_readers])/np.sqrt(np.sum(~np.isnan(d_prime[good_readers]))), \
        color=[0.4, 0.4, 0.4], ecolor = [0,0,0])
plt.bar(1.9, np.nanmean(d_prime[poor_readers]), width=0.2, \
        yerr = np.nanstd(d_prime[poor_readers])/np.sqrt(np.sum(~np.isnan(d_prime[poor_readers]))), \
        color=[0.4, 0.4, 0.4], ecolor = [0,0,0])
plt.xticks([1,2],('Typical readers','Struggling readers'))
plt.xlim([0.5,2.5])
plt.ylim([0,2])
plt.ylabel('D-prime')
plt.show()
#print(hit_rate)
#print(overall_pseudo_rt)

temp = d_prime[good_readers]
a = temp[np.where(~np.isnan(temp))[0]]
temp = d_prime[poor_readers]
b = temp[np.where(~np.isnan(temp))[0]]

stats.ttest_ind(a,b)

plt.figure(3)
plt.clf()
plt.hold(True)
plt.bar(0.9, np.nanmean(hit_rate[good_readers]), width=0.2, \
        yerr = np.nanstd(hit_rate[good_readers])/np.sqrt(np.sum(~np.isnan(hit_rate[good_readers]))), \
        color=[0.4, 0.4, 0.4], ecolor = [0,0,0])
plt.bar(1.9, np.nanmean(hit_rate[poor_readers]), width=0.2, \
        yerr = np.nanstd(hit_rate[poor_readers])/np.sqrt(np.sum(~np.isnan(hit_rate[poor_readers]))), \
        color=[0.4, 0.4, 0.4], ecolor = [0,0,0])
plt.xticks([1,2],('Typical readers','Struggling readers'))
plt.xlim([0.5,2.5])
plt.ylim([0,1])
plt.ylabel('Hit rate')
plt.show()
#print(hit_rate)
#print(overall_pseudo_rt)

temp = hit_rate[good_readers]
a = temp[np.where(~np.isnan(temp))[0]]
temp = hit_rate[poor_readers]
b = temp[np.where(~np.isnan(temp))[0]]

stats.ttest_ind(a,b)
    
#%%
for k, s in enumerate(session2):
    os.chdir(os.path.join(raw_dir,s))
    os.chdir('lists')
    
    if s == '172_th160825':
        run_names = ['1', '2', '3', '4', '5']
    elif s == '170_gm160613':
        run_names = ['1', '2', '3', '5', '6']
    else:
        run_names = ['1', '2', '3', '4', '5', '6']
       
    plt.figure(500)
    plt.clf()
    
    all_index = []
    for n, r_name in enumerate(run_names):
        fn = 'ALL_' + s + '_' + r_name + '-eve.lst'
        mat = []
        for line in open(fn).readlines():
            mat.append(line.split())
        
        time_point = [float(row[0])/1200 for row in mat]
        
        time_point = np.subtract(time_point, time_point[0])
        
        event_id = [row[2] for row in mat]
        
        for ii in np.arange(0,8):
            if np.mod(n,2) == 0:
                index = ([i for i,x in enumerate(event_id) if x == str(101+ii)])
            else:
                 index = [i for i,x in enumerate(event_id) if x == str(201+ii)]
            all_index.append(index)            
        
        plt.subplot(6,1,n+1)
        plt.plot(time_point, event_id,'o-')
        plt.title(s)
    
    os.chdir(raw_dir)
    os.chdir('figures')
    os.chdir('events2')
    figure_name = s + '.png'
    plt.savefig(figure_name)
    