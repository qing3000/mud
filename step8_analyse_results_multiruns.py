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

def CopyImagesByRange(runNum, r, vs, imageNums, cribNums, rows, cols):
    for i in np.nonzero(np.logical_and(vs > r, vs < r + 0.01))[0][:3]:
        fn = 'Run%d_Image%05d_Crib%04d_Row%d_Col%d' % (runNum, imageNums[i], cribNums[i], rows[i], cols[i])
        im = imread('Output\\Blocks\\Run_%d\\%s.png' % (runNum, fn))
        imwrite('Diagnostics\\Value%02d_%s.jpg' % (r, fn), im)
        #copy('Output\\Blocks\\Run_%d\\%s.png' % (runNum, fn), 'Diagnostics\\Value%02d_%s.png' % (r, fn))


'''Load in the ranking done by human'''
'''Load in the labelling by the CNN and recalculate the labels based on a new threshold'''
cleanValues = np.loadtxt('output\\Clean_values.csv')
muddyValues = np.loadtxt('output\\Mud_values.csv')

imageNum1s, cribNum1s, row1s, col1s, v1s, temp, labels = np.loadtxt('Output\\Run132_result.csv', delimiter = ',', skiprows = 1, unpack = True)
imageNum2s, cribNum2s, row2s, col2s, v2s, temp, labels = np.loadtxt('Output\\Run354_result.csv', delimiter = ',', skiprows = 1, unpack = True)
imageNum3s, cribNum3s, row3s, col3s, v3s, temp, labels = np.loadtxt('Output\\Run364_batch1_result.csv', delimiter = ',', skiprows = 1, unpack = True)
imageNum4s, cribNum4s, row4s, col4s, v4s, temp, labels = np.loadtxt('Output\\Run364_batch2_result.csv', delimiter = ',', skiprows = 1, unpack = True)
imageNum5s, cribNum5s, row5s, col5s, v5s, temp, labels = np.loadtxt('Output\\Run364_batch3_result.csv', delimiter = ',', skiprows = 1, unpack = True)


# print('Copy problematic images')
# for r in range(-10, 10):
    #CopyImagesByRange(132, r, v1s, imageNum1s, cribNum1s, row1s, col1s)
    #CopyImagesByRange(354, r, v2s, imageNum2s, cribNum2s, row2s, col2s)
    #CopyImagesByRange(364, r, v3s, imageNum3s, cribNum3s, row3s, col3s)
    
binSize = 0.2
hx_clean, hy_clean = fixedHist(cleanValues, binSize)
hx_muddy, hy_muddy = fixedHist(muddyValues, binSize)

hx1, hy1 = fixedHist(v1s, binSize)
hx2, hy2 = fixedHist(v2s, binSize)
hx3, hy3 = fixedHist(v3s, binSize)
hx4, hy4 = fixedHist(v3s, binSize)
hx5, hy5 = fixedHist(v3s, binSize)

plt.subplot(2,1,1)
plt.plot(hx_clean, hy_clean, label = 'Clean training data (%d samples)' % len(cleanValues))
plt.plot(hx_muddy, hy_muddy, label = 'Muddy training data (%d samples)' % len(muddyValues))
plt.plot(hx1, hy1, label = 'Run 132 (%d samples)' % len(imageNum1s))
plt.plot(hx2, hy2, label = 'Run 354 (%d samples)' % len(imageNum2s))
plt.plot(hx3, hy3, label = 'Run 364 batch 1(%d samples)' % len(imageNum3s))
plt.plot(hx4, hy4, label = 'Run 364 batch 2(%d samples)' % len(imageNum4s))
plt.plot(hx5, hy5, label = 'Run 364 batch 3(%d samples)' % len(imageNum5s))
plt.grid(True)
plt.legend(loc = 0)
plt.title('Classifier 1 output distributions')         
plt.subplot(2,1,2)
plt.plot(hx_clean, hy_clean, label = 'Clean training data (%d samples)' % len(cleanValues))
plt.plot(hx_muddy, hy_muddy, label = 'Muddy training data (%d samples)' % len(muddyValues))
plt.plot(hx1, hy1, label = 'Run 132 (%d samples)' % len(imageNum1s))
plt.plot(hx2, hy2, label = 'Run 354 (%d samples)' % len(imageNum2s))
plt.plot(hx3, hy3, label = 'Run 364 batch 1(%d samples)' % len(imageNum3s))
plt.plot(hx4, hy4, label = 'Run 364 batch 2(%d samples)' % len(imageNum4s))
plt.plot(hx5, hy5, label = 'Run 364 batch 3(%d samples)' % len(imageNum5s))
plt.xlim([-10, 10])
plt.ylim([0, 0.04])
plt.grid(True)
plt.title('Determining the threshold')         

    
    
         
 
