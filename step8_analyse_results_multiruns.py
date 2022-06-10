# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:13:03 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt
from step7_testing import fixedHist
from shutil import copy

def CopyImagesByRange(runNum, rng, vs, imageNums, cribNums, rows, cols):
    for i in np.nonzero(np.logical_and(vs > rng[0], vs < rng[1]))[0]:
        fn = 'Run%d_Image%05d_Crib%04d_Row%d_Col%d.png' % (runNum, imageNums[i], cribNums[i], rows[i], cols[i])
        copy('Output\\Blocks\\Run_%d\\%s' % (runNum, fn), 'Diagnostics\\')


'''Load in the ranking done by human'''
'''Load in the labelling by the CNN and recalculate the labels based on a new threshold'''
imageNum1s, cribNum1s, row1s, col1s, v1s, temp, labels = np.loadtxt('Output\\Run132_result.csv', delimiter = ',', skiprows = 1, unpack = True)
imageNum2s, cribNum2s, row2s, col2s, v2s, temp, labels = np.loadtxt('Output\\Run354_result.csv', delimiter = ',', skiprows = 1, unpack = True)
imageNum3s, cribNum3s, row3s, col3s, v3s, temp, labels = np.loadtxt('Output\\Run364_result.csv', delimiter = ',', skiprows = 1, unpack = True)

print('Copy problematic images')
rng = [10, 10.01]
CopyImagesByRange(132, rng, v1s, imageNum1s, cribNum1s, row1s, col1s)
CopyImagesByRange(354, rng, v2s, imageNum2s, cribNum2s, row2s, col2s)
CopyImagesByRange(364, rng, v3s, imageNum3s, cribNum3s, row3s, col3s)
    
binSize = 0.2
hx1, hy1 = fixedHist(v1s, binSize)
hx2, hy2 = fixedHist(v2s, binSize)
hx3, hy3 = fixedHist(v3s, binSize)
plt.plot(hx1, hy1, label = 'Run 132')
plt.plot(hx2, hy2, label = 'Run 354')
plt.plot(hx3, hy3, label = 'Run 364')
plt.grid(True)
plt.legend(loc = 0)
         

    
    
         
 
