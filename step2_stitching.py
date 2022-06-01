# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:52:02 2022

@author: zhangq
"""

import numpy as np
import scipy.ndimage as ni
from imageio import imread, imwrite
import os
from glob import glob

'''Improve image constrast and brightness by a statistical method'''   
def enhance_image_quality(im0, railCentre, railWidth):
    contrastNSigma = 3.5
    halfRailWidth = int(railWidth / 2)
    smoothWinSize = 10
    im = im0.astype('float')
    M, N = np.shape(im)    
    
    '''Exclude the rail'''
    validColumns = np.array([True] * N)
    validColumns[railCentre - halfRailWidth : railCentre + railWidth - halfRailWidth] = False
    p1 = np.mean(im, 0)
    p2 = ni.convolve(p1, np.ones(smoothWinSize) / smoothWinSize, mode = 'mirror')    
    p3 = np.interp(np.arange(0, N), np.nonzero(validColumns)[0], p2[validColumns])
    intensityMatrix = np.array([p3] * M)
    im2 = im / intensityMatrix - 1
    rowSigmas = np.std(im2, axis = 1)
    averageSigma = np.mean(rowSigmas)
    contrastRatio = 128 / (averageSigma * contrastNSigma)
    im3 = im2 * contrastRatio + 128;
    im3[im3 < 0] = 0
    im3[im3 > 254] = 254
    return im3


overlappingSize = 425
runNum = 354
dataPath = 'C:\\Personal\\Mudspots\\Run_354-20200216@032951_08000-13966\\'

foutPath = 'Output\\Stitched\\Run_%03d\\' % runNum
if not os.path.exists(foutPath):
    os.mkdir(foutPath)

'''Get all the image filenames'''
fpaths = glob('Run_354-20200216@032951_08000-13966\\Run*')
fns = []
for fpath in fpaths:
    fns += glob(fpath + '\\*.jpg')

'''Load in the precalculated rail centre lines'''
railCentres = np.loadtxt('RailCentreLines.csv', delimiter = ',')

'''Image contrast enhancement and stitching'''    
railWidth = 100
for i in range(0, int(len(fns) / 4)):
    if i % 100 == 0:
        print('Stitching image %d out of %d' % (i, len(fns) / 4))
    j = i * 4
    railCentre1 = int(railCentres[i, 0] + 0.5)
    railCentre2 = int(railCentres[i, 1] + 0.5)
    railCentre3 = int(railCentres[i, 2] + 0.5)
    railCentre4 = int(railCentres[i, 3] + 0.5)
    
    '''Enhance the image contrast and brightness'''
    im1 = enhance_image_quality(imread(fns[j]), railCentre1, railWidth)
    im2 = enhance_image_quality(imread(fns[j + 1]), railCentre2, railWidth)
    im3 = enhance_image_quality(imread(fns[j + 2]), railCentre3, railWidth)
    im4 = enhance_image_quality(imread(fns[j + 3]), railCentre4, railWidth)
    
    '''Draw lines to record the rail centres'''
    im1[:, railCentre1 - 1] = 255
    im3[:, railCentre3 - 1] = 255

    '''Combine 4 cameras'''
    im = np.concatenate((im1[:, :railCentre1], \
                         im2[:, railCentre2:-int(overlappingSize / 2)], \
                         im3[:, int(overlappingSize / 2):railCentre3], \
                         im4[:,railCentre4:]), axis = 1)

    '''Save the stitched image'''
    fn = fns[j]
    imageNum = int(fn[fn.rfind('\\') + 1:fn.rfind('_')])
    imwrite('%s%5d.png' % (foutPath, imageNum), im.astype('uint8'))

