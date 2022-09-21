# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:13:03 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from step7_testing import fixedHist
from shutil import copy
from imageio import imread, imwrite
from step5_cutting import enhance_image_quality
import os
from numpy.fft import fft2, fftshift

def calculate_ring_max(im, cuts):
    M, N = np.shape(im)
    cc = [int(M / 2 + 0.5), int(N / 2 + 0.5)]
    im_copy = im.copy()
    im_copy[cc[0] - cuts[0]: cc[0] + cuts[0], cc[1] - cuts[0] : cc[1] + cuts[0]] = 0
    block = im_copy[cc[0] - cuts[1]: cc[0] + cuts[1], cc[1] - cuts[1] : cc[1] + cuts[1]]
    full_block = im[cc[0] - cuts[1]: cc[0] + cuts[1], cc[1] - cuts[1] : cc[1] + cuts[1]]
    vmax = np.max(block)
    return vmax, block


def CopyImagesByRange(runNum, rngs, vs, imageNums, cribNums, rows, cols):
    indices = np.nonzero(np.logical_and(vs > rngs[0], vs < rngs[1]))[0]
    for i in indices[:500]:
        fn = 'Run%d_Image%05d_Crib%04d_Row%d_Col%d' % (runNum, imageNums[i], cribNums[i], rows[i], cols[i])
        im = imread('Output\\Blocks\\Run_%s\\%s.png' % (runNum, fn))
        foutPath = 'Diagnostics\\Run%d\\%03d-%03d' % (runNum, rngs[0], rngs[1])
        if not os.path.exists(foutPath):
            os.mkdir(foutPath)
        imwrite('%s\\%s.jpg' % (foutPath, fn), im)


'''Load in the ranking done by human'''
'''Load in the labelling by the CNN and recalculate the labels based on a new threshold'''
cleanValues = np.loadtxt('output\\Clean_Cribs_values.csv')
muddyValues = np.loadtxt('output\\Mud_Cribs_values.csv')

binSize = 0.5
hx_clean, hy_clean = fixedHist(cleanValues, binSize)
hx_muddy, hy_muddy = fixedHist(muddyValues, binSize)
plt.plot(hx_clean, hy_clean, label = 'Clean training data (%d samples)' % len(cleanValues))
plt.plot(hx_muddy, hy_muddy, label = 'Muddy training data (%d samples)' % len(muddyValues))
plt.grid(True)
plt.legend(loc = 0)
plt.title('Training Data Distributions (Tiles)')        
raise SystemExit

imageNum1s, cribNum1s, row1s, col1s, v1s, temp, labels = np.loadtxt('Output\\Run132_result.csv', delimiter = ',', skiprows = 1, unpack = True)
imageNum2s, cribNum2s, row2s, col2s, v2s, temp, labels = np.loadtxt('Output\\Run354_result.csv', delimiter = ',', skiprows = 1, unpack = True)
imageNum3s, cribNum3s, row3s, col3s, v3s, temp, labels = np.loadtxt('Output\\Run364_batch1_result.csv', delimiter = ',', skiprows = 1, unpack = True)
imageNum4s, cribNum4s, row4s, col4s, v4s, temp, labels = np.loadtxt('Output\\Run364_batch2_result.csv', delimiter = ',', skiprows = 1, unpack = True)
imageNum5s, cribNum5s, row5s, col5s, v5s, temp, labels = np.loadtxt('Output\\Run364_batch3_result.csv', delimiter = ',', skiprows = 1, unpack = True)
fn6s, row6s, col6s, v6s, temp, labels = np.genfromtxt('Output\\RioTinto_All_result.csv', delimiter = ',', skip_header = 1, unpack = True)
fn7s, row6s, col6s, v7s, temp, labels = np.genfromtxt('Output\\RioTinto_Muddy_result.csv', delimiter = ',', skip_header = 1, unpack = True)
fn6s = np.genfromtxt('Output\\RioTinto_All_result.csv', delimiter = ',', skip_header = 1, dtype = str, usecols = 0)
fn7s = np.genfromtxt('Output\\RioTinto_Muddy_result.csv', delimiter = ',', skip_header = 1, dtype = str, usecols = 0)

# print('Copy problematic images')
# cuts = [10, 50, 90]
# clrmap = cm.jet
# clrmap.set_under('white')
# for rangeStart in range(-10, 10):
#     foutPath = 'Diagnostics\\RioTinto\\%d-%d\\' % (rangeStart, rangeStart + 1)
#     if not os.path.exists(foutPath):
#         os.mkdir(foutPath)
#     indices = np.nonzero(np.logical_and(v7s > rangeStart, v7s <= rangeStart + 1))[0]
#     for index in indices[20:]:
#         fn = fn7s[index]

#         im = imread('Output\\Blocks\\RioTinto\\Muddy\\%s.png' % fn)
#         fim = fftshift(fft2(im - np.mean(im)))
#         fim_mag = np.abs(fim)

#         vmax1, block1 = calculate_ring_max(fim_mag, [0, cuts[0]])
#         vmax2, block2 = calculate_ring_max(fim_mag, [cuts[0], cuts[1]])
#         vmax3, block3 = calculate_ring_max(fim_mag, [cuts[1], cuts[2]])
#         vmax4, block4 = calculate_ring_max(fim_mag, [cuts[2], 128])

#         plt.subplot(1,2,1)
#         plt.imshow(im, interpolation = 'none', cmap = 'gray')
#         plt.subplot(2,4,3)
#         plt.imshow(block1, interpolation = 'none', cmap = clrmap, vmin = 1e-9, vmax = vmax1)
#         plt.subplot(2,4,4)
#         plt.imshow(block2, interpolation = 'none', cmap = clrmap, vmin = 1e-9,  vmax = vmax2)
#         plt.subplot(2,4,7)
#         plt.imshow(block3, interpolation = 'none', cmap = clrmap, vmin = 1e-9,  vmax = vmax3)
#         plt.subplot(2,4,8)
#         plt.imshow(block4, interpolation = 'none', cmap = clrmap, vmin = 1e-9,  vmax = vmax4)
#         raise SystemExit
#         imwrite('%s\\%s.jpg' % (foutPath, fn), im)

for r in range(-30, 10, 5):
    CopyImagesByRange(132, (r, r + 5), v1s, imageNum1s, cribNum1s, row1s, col1s)
    #CopyImagesByRange(354, (r, r + 5), v2s, imageNum2s, cribNum2s, row2s, col2s)
    #CopyImagesByRange(364, (r, r + 5), v3s, imageNum3s, cribNum3s, row3s, col3s)
    


hx1, hy1 = fixedHist(v1s, binSize)
hx2, hy2 = fixedHist(v2s, binSize)
hx3, hy3 = fixedHist(v3s, binSize)
hx4, hy4 = fixedHist(v4s, binSize)
hx5, hy5 = fixedHist(v5s, binSize)
hx6, hy6 = fixedHist(v6s, binSize)
hx7, hy7 = fixedHist(v7s, binSize)

plt.subplot(3,1,1)
plt.plot(hx_clean, hy_clean, label = 'Clean training data (%d samples)' % len(cleanValues))
plt.plot(hx_muddy, hy_muddy, label = 'Muddy training data (%d samples)' % len(muddyValues))
plt.xlim([-50, 20])
plt.gca().xaxis.set_ticklabels([])
plt.grid(True)
plt.legend(loc = 0)
plt.title('Training Data Distributions (Tiles)')         

plt.subplot(3,1,2)
plt.plot(hx1, hy1, label = 'Run 132 (%d samples)' % len(imageNum1s))
plt.plot(hx2, hy2, label = 'Run 354 (%d samples)' % len(imageNum2s))
plt.plot(hx3, hy3, label = 'Run 364 batch 1(%d samples)' % len(imageNum3s))
plt.plot(hx4, hy4, label = 'Run 364 batch 2(%d samples)' % len(imageNum4s))
plt.plot(hx5, hy5, label = 'Run 364 batch 3(%d samples)' % len(imageNum5s))
plt.xlim([-50, 20])
plt.gca().xaxis.set_ticklabels([])
plt.grid(True)
plt.legend(loc = 0)
plt.title('Kiwi ARC and Vline Distributions (Tiles)')

plt.subplot(3,1,3)
plt.plot(hx6, hy6, label = 'RioTinto All Cribs (%d samples)' % len(v6s))
plt.plot(hx7, hy7, label = 'RioTinto Muddy (%d samples)' % len(v7s))
plt.xlim([-50, 20])
plt.grid(True)
plt.title('Rio Tinto Distributions (Tiles)')         

    
    
         
 
