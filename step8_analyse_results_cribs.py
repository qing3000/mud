# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:13:03 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt
from step7_testing import fixedHist
from shutil import copy
from imageio import imread, imwrite

'''Load in the ranking done by human'''
runStr = 'Run_132-20190424@105356_38000-48000'
runNum = '132'
runStr = 'Run_354-20200216@032951_08000-13966'
runNum = '354'
runStr = 'Run_364-20200424@011547_52000-62000'
runNum = '364_batch1'
runStr = 'Run_364-20200424@011547_70000-80000'
runNum = '364_batch2'
runStr = 'Run_364-20200424@011547_85000-95000'
runNum = '364_batch3'
csvfile = 'C:\\Personal\\Mudspots\\%s\\%s_Rankings.csv' % (runStr, runStr)   
imgNums, smi = np.loadtxt(csvfile, delimiter = ',', skiprows = 1, usecols = (0, 3), unpack = True)
'''Run354 has some missing data'''
#imgNums = imgNums[:5500]
#smi = smi[:5500]

'''Load in the labelling by the CNN and recalculate the labels based on a new threshold'''
imageNums, CribNums, v1s, v2s, labels = np.loadtxt('Output\\Run%s_Cribs_result.csv' % runNum, delimiter = ',', skiprows = 1, unpack = True)
labels = v1s < -1

'''Count the number of mud blocks in each image'''
counts = np.array([np.sum(labels[imageNums == imgNum]) for imgNum in imgNums])

'''Calculate the fraction of false negatives (mud images classified as clean or zero mud blocks)'''
totalCleanLabelCount = np.count_nonzero(counts == 0)
wrongCleanLabelCount = np.count_nonzero(np.logical_and(counts == 0, smi == 3))
print('Fraction of false negatives=%d/%d=%.2f%%' % (wrongCleanLabelCount, totalCleanLabelCount, wrongCleanLabelCount * 100 / totalCleanLabelCount))

'''Calculate the fraction of false positives (clean images classified as mud (nonzero mud blocks)'''
totalMudLabelCount = np.count_nonzero(counts > 0)
wrongMudLabelCount = np.count_nonzero(np.logical_and(counts > 0, smi == 1))
print('Fraction of false positives=%d/%d=%.2f%%' % (wrongMudLabelCount, totalMudLabelCount, wrongMudLabelCount * 100 / totalMudLabelCount))

# print('Copy problematic images')
# wrongCleanLabelIndices = np.nonzero(np.logical_and(counts == 0, smi != 1))[0]
# for i in wrongCleanLabelIndices:
#     img = imread('Output\\Drawn\\Run_%s\\%05d_%05d.png' % (runNum, imgNums[i], imgNums[i] + 1))
#     M, N, K = np.shape(img)
#     imwrite('Output\\FalseNegatives\\Run%s\\%05d.jpg' % (runNum, imgNums[i]), img[:int(M / 2), :, :3])

# wrongMudLabelIndices = np.nonzero(np.logical_and(counts > 0, smi == 1))[0]
# for i in wrongMudLabelIndices:
#     img = imread('Output\\Drawn\\Run_%s\\%05d_%05d.png' % (runNum, imgNums[i], imgNums[i] + 1))
#     M, N, K = np.shape(img)
#     imwrite('Output\\FalsePositives\\Run%s\\%05d.jpg' % (runNum, imgNums[i]), img[:int(M / 2), :, :3])
    
    
#     jj = np.nonzero(imageNums == imgNums[i])[0]
#     for j in jj:
#         fn = 'Run%d_Image%05d_Crib%04d_Row%d_Col%d.png' % (runNum, imageNums[j], CribNums[j], Rows[j], Cols[j])
#         copy('Output\\Blocks\\Run_%d\\%s' % (runNum, fn), 'Output\\Misses\\Run%d_Blocks\\' % runNum)
        


binSize = 0.2
hx1, hy1 = fixedHist(v1s, binSize)
hx2, hy2 = fixedHist(v2s, binSize)
plt.plot(hx1, hy1, label = 'Classifer 1 output')
plt.plot(hx2, hy2, label = 'Classifer 2 output')
#plt.plot(v1s[wrongCleanLabelIndices], 0.2 * np.ones(len(wrongCleanLabelIndices)), '.', label = 'Wrongly labeled as clean')
plt.grid(True)
plt.legend(loc = 0)
         

    
    
         
 
