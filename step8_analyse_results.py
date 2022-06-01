# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:13:03 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt
from step7_testing import fixedHist

'''Load in the ranking done by human'''
runStr = 'Run_354-20200216@032951_08000-13966'
csvfile = 'C:\\Personal\\Mudspots\\%s\\%s_Rankings.csv' % (runStr, runStr)   
imgNums, smi = np.loadtxt(csvfile, delimiter = ',', skiprows = 1, usecols = (0, 3), unpack = True)
imgNums = imgNums[:5500]
smi = smi[:5500]

'''Load in the labelling by the CNN and recalculate the labels based on a new threshold'''
imageNums, CribNums, Rows, Cols, v1s, v2s, labels = np.loadtxt('Output\\Run354_result.csv', delimiter = ',', skiprows = 1, unpack = True)
labels = v1s < 0

'''Count the fraction of missclassification in clean'''
counts = np.array([np.sum(labels[imageNums == imgNum]) for imgNum in imgNums])
correctCleanLabelCount = np.count_nonzero(np.logical_and(counts == 0, smi == 1))
totalCleanLabelCount = np.count_nonzero(counts == 0)
cleanFraction = correctCleanLabelCount / totalCleanLabelCount
print('Fraction of missclassifiction in clean (mud missclassified as clean) = %f' % (1 - cleanFraction))

binSize = 0.2
hx1, hy1 = fixedHist(v1s, binSize)
hx2, hy2 = fixedHist(v2s, binSize)
plt.plot(hx1, hy1, label = 'Classifer 1 output')
plt.plot(hx2, hy2, label = 'Classifer 2 output')
plt.grid(True)
plt.legend(loc = 0)
         
         
 
