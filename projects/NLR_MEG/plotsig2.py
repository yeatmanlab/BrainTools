#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:00:57 2018

@author: sjjoo
"""
import matplotlib.pyplot as plt
import numpy as np

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

def plotsig2(times,nReps,X,task1, task2, subject, p_value):
    
    t_window = 4 # 12 ms
    temp_range = [100*(0.5-(1-p_value)/2), 100*(0.5 + (1-p_value)/2)]
    """ Lexical task: all subjects """
    result = np.zeros((X.shape[0],1))
    diff = np.empty((nReps,1))
    p = np.empty((X.shape[0],2))
    for i in range(0,X.shape[0]-t_window):
        X1 = np.mean(X[i:i+t_window,subject,task1], axis=0)
        X2 = np.mean(X[i:i+t_window,subject,task2], axis=0)
#        true_diff = np.mean(X1) - np.mean(X2)
        for j in range(0,nReps):
            aaa = np.random.choice(range(X2.shape[0]),X2.shape)
            diff[j] = np.mean(X1[aaa]) - np.mean(X2[aaa])
        p[i:] = np.percentile(diff,temp_range)
        if p[i,0] > 0.0:
            result[i] = 1
    k = 0    
    for i in range(0,len(times)-t_window):
        if result[i]:
            if times[i] > 0.2 and k == 0:
                k = 1
#                plt.plot([times[i],times[i]],[0,2.7],'--',color=[0,0,0])
                plt.text(times[i],2.0,np.str(times[i]))
            plt.plot(times[i],result[i]*0.2,'o',markerfacecolor=c_table[5],markeredgecolor=c_table[5])
            
#    for i in range(0,len(times)-t_window):
#        if result[i]:
#            plt.plot(times[i],result[i]*0.2,'o',markerfacecolor=c_table[5],markeredgecolor=c_table[5])
