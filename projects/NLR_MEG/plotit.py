#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:01:39 2018

@author: sjjoo
"""

import matplotlib.pyplot as plt

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
    (0.6941,    0.3490,    0.1569),
    (0,0,0) )

def plotit(times, M, errM, yMin=-1, yMax=5, title = 'abc', task=0, color_num = 12):
    plt.figure(title)
    plt.clf()    
    
    plt.hold(True)
    
    plt.plot(times, M[:,task],'-',color=c_table[color_num],label='Low noise')
    plt.fill_between(times, M[:,task]-errM[:,task], M[:,task]+errM[:,task], facecolor=c_table[color_num], alpha=0.2, edgecolor='none')
    
#    plt.plot(times, M[:,task+1],'-',color=c_table[3],label='Med noise')
#    plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')
#    
#    plt.plot(times, M[:,task+3],'-',color=[0, 0, 0],label='High noise')
#    plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=[0, 0, 0], alpha=0.2, edgecolor='none')
    
    plt.grid(b=True)
    plt.ylim([yMin, yMax])
#    if task == 0:
#        fn = 'Dot:' + subject
#        plt.title(fn)
#    elif task == 5:
#        fn = 'Lexcial:' + subject
#        plt.title(fn)
