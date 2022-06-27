# -*- coding: utf-8 -*-
"""
Created on Sun May  1 16:42:46 2022

@author: zhangq
"""

import numpy as np
import glob
import os
from imageio import imread, imwrite
import matplotlib.pyplot as plt
import scipy.ndimage as ni

def enhance_image_quality(im0):
    contrastNSigma = 3.5
    smoothWinSize = 10
    im = im0.astype('float')
    M, N = np.shape(im)    
    
    p1 = np.mean(im, 0)
    p2 = ni.convolve(p1, np.ones(smoothWinSize) / smoothWinSize, mode = 'mirror')    
    intensityMatrix = np.array([p2] * M)
    im2 = im / intensityMatrix - 1
    rowSigmas = np.std(im2, axis = 1)
    averageSigma = np.mean(rowSigmas)
    contrastRatio = 128 / (averageSigma * contrastNSigma)
    im3 = im2 * contrastRatio + 128;
    im3[im3 < 0] = 0
    im3[im3 > 254] = 254
    # plt.subplot(3,1,1)
    # plt.imshow(im, interpolation = 'none', cmap = 'gray')
    # plt.grid(True)
    # plt.title('Raw crib image')
    # plt.subplot(3,1,2)
    # plt.plot(p2)
    # plt.title('Intensity profile')
    # plt.xlim([0, N - 1])
    # plt.grid(True)
    # plt.subplot(3,1,3)
    # plt.imshow(im3, interpolation = 'none', cmap = 'gray')
    # plt.grid(True)
    # plt.title('Contrast enhanced')
    # raise SystemExit
    return im3

fns = glob.glob('Data\\RioTinto\\mudspot_cribs\\Run_122-20211004@162148\\*.png')

foutPath = 'Output\\Blocks\\RioTinto\\Muddy\\'
if not os.path.exists(foutPath):
    os.mkdir(foutPath)
trimSize = 50

blockWidth = 256
blockHeight = 256
ignoreFraction = 0.7
for k, fn in enumerate(fns):
    if k % 10 == 0:
        print('%d out of %d' % (k, len(fns)))
    im = imread(fn)[trimSize : -trimSize,trimSize : -trimSize]
    im = enhance_image_quality(im)
    M, N = np.shape(im)
    m = int(np.ceil(M / blockHeight))
    n = int(np.ceil(N / blockWidth))
    em = M % blockHeight
    en = N % blockWidth
    
    for i in range(m):
        for j in range(n):
            startRow = i * blockHeight
            endRow = startRow + blockHeight - 1
            if endRow > M - 1:
                endRow = M - 1
                
            startCol = j * blockWidth
            endCol = startCol + blockWidth - 1
            if endCol > N - 1:
                endCol = N - 1
            
            bm = endRow - startRow + 1
            bn = endCol - startCol + 1
            if bm * bn / blockWidth / blockHeight > ignoreFraction:
                block = im[startRow : endRow + 1, startCol : endCol + 1]
                container = (np.ones((blockHeight, blockWidth)) * 255).astype('uint8')
                container[:bm, :bn] = block
                shortfn = fn[fn.rfind('\\') + 1:-4]
                imwrite(foutPath + '%s_Row%d_Col%d.png' % (shortfn, i, j), container)
    