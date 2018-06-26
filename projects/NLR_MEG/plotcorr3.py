#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:01:39 2018

@author: sjjoo
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy import stats as stats

corrcolor = (    (1.0000,    1.0000,    0.8000),
    (1.0000,    0.9294,    0.6275),
    (0.9961,    0.8510,    0.4627),
    (0.9961,   0.6980,    0.2980),
    (0.9922,    0.5529,    0.2353),
    (0.9882,    0.3059,    0.1647),
    (0.8902,    0.1020,    0.1098),
    (0.7412,         0,    0.1490),
    (0.5020,         0,    0.1490) )
corrcolor2 = (    (1.0000,    1.0000,    0.8980),
    (0.9686,    0.9882,    0.7255),
    (0.8510,    0.9412,    0.6392),
    (0.6784,    0.8667,    0.5569),
    (0.4706,    0.7765,    0.4745),
    (0.2549,    0.6706,   0.3647),
    (0.1373,    0.5176,   0.2627),
         (0,    0.4078,    0.2157),
         (0,    0.2706,    0.1608) )

def plotcorr3(times, C, readingscore, ypos = 2.3):
    t_window = 4 # 12 ms

    corr = np.empty((1,C.shape[0]-t_window))[0]
    corr_p = np.empty((1,C.shape[0]-t_window))[0]
    
    for t in range(0,301-t_window):
        temp_p = stats.pearsonr(readingscore,np.mean(C[t:t+t_window,:], axis=0))
        corr[t] = temp_p[0]
        corr_p[t] = temp_p[1]

    for t in range(0,len(times)-t_window):
        if corr_p[t] >= 0.05:
            plt.plot(times[t], ypos, 's',markerfacecolor=[0.5,0.5,0.5],alpha=0.5)
        elif corr[t]>=0.15 and corr[t]<0.2:
            plt.plot(times[t], ypos, 's',markerfacecolor=corrcolor[8])
        elif corr[t]>=0.2 and corr[t]<0.25:
            plt.plot(times[t], ypos, 's',markerfacecolor=corrcolor[7])
        elif corr[t]>=0.25 and corr[t]<0.3:
            plt.plot(times[t], ypos, 's',markerfacecolor=corrcolor[6])
        elif corr[t]>=0.3 and corr[t]<0.35:
            plt.plot(times[t], ypos, 's',markerfacecolor=corrcolor[5])
        elif corr[t]>=0.35 and corr[t]<0.4:
            plt.plot(times[t], ypos, 's',markerfacecolor=corrcolor[4])
        elif corr[t]>=0.4 and corr[t]<0.45:
            plt.plot(times[t], ypos, 's',markerfacecolor=corrcolor[3])
        elif corr[t]>=0.45 and corr[t]<0.5:
            plt.plot(times[t], ypos, 's',markerfacecolor=corrcolor[2])
        elif corr[t]>=0.5 and corr[t]<0.55:
            plt.plot(times[t], ypos, 's',markerfacecolor=corrcolor[1])
        elif corr[t]>=0.55 and corr[t]<0.6:
            plt.plot(times[t], ypos, 's',markerfacecolor=corrcolor[0])
        elif corr[t]>=0.6:
            plt.plot(times[t], ypos, 's',markerfacecolor=[1,1,1])
        else:
            plt.plot(times[t], ypos, 's',markerfacecolor=[0.5,0.5,0.5],alpha=0.5)
    
    return corr
    