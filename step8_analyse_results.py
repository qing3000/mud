# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:13:03 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt
from step7_testing import fixedHist
from shutil import copy

'''Load in the ranking done by human'''
runStr = 'Run_354-20200216@032951_08000-13966'
csvfile = 'C:\\Personal\\Mudspots\\%s\\%s_Rankings.csv' % (runStr, runStr)   
imgNums, smi = np.loadtxt(csvfile, delimiter = ',', skiprows = 1, usecols = (0, 3), unpack = True)
imgNums = imgNums[:5500]
smi = smi[:5500]

'''Load in the labelling by the CNN and recalculate the labels based on a new threshold'''
imageNums, CribNums, Rows, Cols, v1s, v2s, labels = np.loadtxt('Output\\Run354_result.csv', delimiter = ',', skiprows = 1, unpack = True)

'''Count the fraction of missclassification in clean'''
counts = np.array([np.sum(labels[imageNums == imgNum]) for imgNum in imgNums])
correctCleanLabelCount = np.count_nonzero(np.logical_and(counts == 0, smi == 1))
totalCleanLabelCount = np.count_nonzero(counts == 0)
cleanFraction = correctCleanLabelCount / totalCleanLabelCount
print('Fraction of missclassifiction in clean (mud missclassified as clean) = %f' % (1 - cleanFraction))

wrongCleanLabelIndices = np.nonzero(np.logical_and(counts == 0, np.logical_or(smi == 2, smi == 3)))[0]

   
# print('Copy problematic images')
# for i in wrongCleanLabelIndices:
#     copy('Output\\Stitched\\Run_354\\%05d.png' % imgNums[i], 'Output\\Test\\Run_354\\')
#     jj = np.nonzero(imageNums == imgNums[i])[0]
#     for j in jj:
#         fn = 'Run354_Image%05d_Crib%04d_Row%d_Col%d.png' % (imageNums[j], CribNums[j], Rows[j], Cols[j])
#         copy('Output\\Blocks\\Run_354\\%s' % fn, 'Output\\Test\\Run_354\\')
        


binSize = 0.2
hx1, hy1 = fixedHist(v1s, binSize)
hx2, hy2 = fixedHist(v2s, binSize)
plt.plot(hx1, hy1, label = 'Classifer 1 output')
plt.plot(hx2, hy2, label = 'Classifer 2 output')
plt.plot(v1s[wrongCleanLabelIndices], 0.2 * np.ones(len(wrongCleanLabelIndices)), '.', label = 'Wrongly labeled as clean')
plt.grid(True)
plt.legend(loc = 0)
         

    
    
         
 
