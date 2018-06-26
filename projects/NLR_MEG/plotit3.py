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
    (0.6941,    0.3490,    0.1569))

def plotit3(times, M, errM, task1, task2, task3, yMin=-1, yMax=5, subject = 'all'):
    plt.figure()
    plt.clf()    
    
    plt.hold(True)
    
    plt.plot(times, M[:,task3],'-',color=[0, 0, 0],label='High noise')
    plt.fill_between(times, M[:,task3]-errM[:,task3], M[:,task3]+errM[:,task3], facecolor=[0, 0, 0], alpha=0.2, edgecolor='none')
    
    plt.plot(times, M[:,task2],'-',color=c_table[1],label='Med noise')
    plt.fill_between(times, M[:,task2]-errM[:,task2], M[:,task2]+errM[:,task2], facecolor=c_table[1], alpha=0.2, edgecolor='none')
    
    plt.plot(times, M[:,task1],'-',color=c_table[5],label='Low noise')
    plt.fill_between(times, M[:,task1]-errM[:,task1], M[:,task1]+errM[:,task1], facecolor=c_table[5], alpha=0.2, edgecolor='none')
    
    plt.grid(b=True)
    plt.ylim([yMin, yMax])
    if task1 < 5:
        fn = 'Dot:' + ' ' + subject
        plt.title(fn)
    elif task1 >= 5:
        fn = 'Lexcial:' + ' ' + subject
        plt.title(fn)
