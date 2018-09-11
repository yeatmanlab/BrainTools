#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 13:04:46 2018

@author: sjjoo
"""

#%%
import matplotlib.pyplot as plt
import numpy as np

corrcolor = (    (1.0000,    1.0000,    0.8000),
    (1.0000,    0.9294,    0.6275),
    (0.9961,    0.8510,    0.4627),
    (0.9961,   0.6980,    0.2980),
    (0.9922,    0.5529,    0.2353),
    (0.9882,    0.3059,    0.1647),
    (0.8902,    0.1020,    0.1098),
    (0.7412,         0,    0.1490),
    (0.5020,         0,    0.1490) )

for i in range(0,len(corrcolor)+1):
    plt.bar(1, 0.9-0.1*i, width=0.4, color=corrcolor[8-i], edgecolor = corrcolor[8-i], align='center')
    if i != 8:
        plt.bar(1, 0.9-0.1*i-0.05, width=0.4, color=np.add(corrcolor[8-i],corrcolor[8-i-1])/2, \
                edgecolor = np.add(corrcolor[8-i],corrcolor[8-i-1])/2, align='center')
    plt.xlim([0.8,1.2])
    plt.ylim([0.05,0.9])
    plt.grid(b=False)
