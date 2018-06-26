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

def plotit(times, M, M1, M2, errM, errM1, errM2, yMin=-1, yMax=5, title = 'abc', task=0):
    plt.figure(title)
    plt.clf()    
    
    plt.subplot(2,3,1)
    plt.hold(True)
    
    plt.plot(times, M[:,0],'-',color=c_table[5],label='Low noise')
    plt.fill_between(times, M[:,0]-errM[:,0], M[:,0]+errM[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')
    
    plt.plot(times, M[:,1],'-',color=c_table[3],label='Med noise')
    plt.fill_between(times, M[:,1]-errM[:,1], M[:,1]+errM[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')
    
    plt.plot(times, M[:,3],'-',color=c_table[1],label='High noise')
    plt.fill_between(times, M[:,3]-errM[:,3], M[:,3]+errM[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')
    
    plt.grid(b=True)
    plt.ylim([yMin, yMax])
    plt.title('Dot task')
    
    plt.subplot(2,3,2)
    plt.hold(True)
    
    plt.plot(times, M1[:,0],'-',color=c_table[5],label='Low noise')
    plt.fill_between(times, M1[:,0]-errM1[:,0], M1[:,0]+errM1[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')
    
    plt.plot(times, M1[:,1],'-',color=c_table[3],label='Med noise')
    plt.fill_between(times, M1[:,1]-errM1[:,1], M1[:,1]+errM1[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')
    
    plt.plot(times, M1[:,3],'-',color=c_table[1],label='High noise')
    plt.fill_between(times, M1[:,3]-errM1[:,3], M1[:,3]+errM1[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')
    
    plt.grid(b=True)
    plt.ylim([yMin, yMax])
    plt.title('Good Readers')
    
    plt.subplot(2,3,3)
    plt.hold(True)
    
    plt.plot(times, M2[:,0],'-',color=c_table[5],label='Low noise')
    plt.fill_between(times, M2[:,0]-errM2[:,0], M2[:,0]+errM2[:,0], facecolor=c_table[5], alpha=0.2, edgecolor='none')
    
    plt.plot(times, M2[:,1],'-',color=c_table[3],label='Med noise')
    plt.fill_between(times, M2[:,1]-errM2[:,1], M2[:,1]+errM2[:,1], facecolor=c_table[3], alpha=0.2, edgecolor='none')
    
    plt.plot(times, M2[:,3],'-',color=c_table[1],label='High noise')
    plt.fill_between(times, M2[:,3]-errM2[:,3], M2[:,3]+errM2[:,3], facecolor=c_table[1], alpha=0.2, edgecolor='none')
    
    plt.grid(b=True)
    plt.ylim([yMin, yMax])
    plt.title('Poor Readers')
    
    plt.subplot(2,3,4)
    plt.hold(True)
    
    plt.plot(times, M[:,5],'-',color=c_table[5],label='Low noise')
    plt.fill_between(times, M[:,5]-errM[:,5], M[:,5]+errM[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')
    
    plt.plot(times, M[:,6],'-',color=c_table[3],label='Med noise')
    plt.fill_between(times, M[:,6]-errM[:,6], M[:,6]+errM[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')
    
    plt.plot(times, M[:,8],'-',color=c_table[1],label='High noise')
    plt.fill_between(times, M[:,8]-errM[:,8], M[:,8]+errM[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')
    
    plt.grid(b=True)
    plt.ylim([yMin, yMax])
    plt.title('Lexical task')
    
    plt.subplot(2,3,5)
    plt.hold(True)
    
    plt.plot(times, M1[:,5],'-',color=c_table[5],label='Low noise')
    plt.fill_between(times, M1[:,5]-errM1[:,5], M1[:,5]+errM1[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')
    
    plt.plot(times, M1[:,6],'-',color=c_table[3],label='Med noise')
    plt.fill_between(times, M1[:,6]-errM1[:,6], M1[:,6]+errM1[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')
    
    plt.plot(times, M1[:,8],'-',color=c_table[1],label='High noise')
    plt.fill_between(times, M1[:,8]-errM1[:,8], M1[:,8]+errM1[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')
    
    plt.grid(b=True)
    plt.ylim([yMin, yMax])
    plt.title('Good Readers')
    
    plt.subplot(2,3,6)
    plt.hold(True)
    
    plt.plot(times, M2[:,5],'-',color=c_table[5],label='Low noise')
    plt.fill_between(times, M2[:,5]-errM2[:,5], M2[:,5]+errM2[:,5], facecolor=c_table[5], alpha=0.2, edgecolor='none')
    
    plt.plot(times, M2[:,6],'-',color=c_table[3],label='Med noise')
    plt.fill_between(times, M2[:,6]-errM2[:,6], M2[:,6]+errM2[:,6], facecolor=c_table[3], alpha=0.2, edgecolor='none')
    
    plt.plot(times, M2[:,8],'-',color=c_table[1],label='High noise')
    plt.fill_between(times, M2[:,8]-errM2[:,8], M2[:,8]+errM2[:,8], facecolor=c_table[1], alpha=0.2, edgecolor='none')
    
    plt.grid(b=True)
    plt.ylim([yMin, yMax])
    plt.title('Poor Readers')
    
    plt.show()