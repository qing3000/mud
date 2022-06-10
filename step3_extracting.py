# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:06:15 2022

@author: zhangq
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ni
from imageio import imread, imwrite
from itertools import combinations
from glob import glob
import cv2
import os

'''Duplicate the single plane gray image into a RGB image with 3 identical planes'''
def gray2rgb(im):
    rgbIm = np.array([im.copy(), im.copy(), im.copy()])
    rgbIm = np.swapaxes(rgbIm, 0, 2)
    rgbIm = np.swapaxes(rgbIm, 0, 1)
    return rgbIm

'''Stack two images with slighly different widths.'''
def stack_images(im1, im2):
    '''Stack two images up'''
    M1, N1 = np.shape(im1)
    M2, N2 = np.shape(im2)
    N = min(N1, N2)
    im = np.concatenate((im1[:, :N], im2[:, :N]))
    return im

'''A very simple clustering algorithm by threshold'''
def simple_clustering(x, threshold):
    xx = np.sort(x)
    '''Calculate a binary array to flag up any close samples (gap < threshold)'''
    b = np.array([0] + (np.diff(xx) < threshold).astype('int').tolist() + [0])
    
    '''Locate any continuous sections of the close samples, for being a "cluster"'''
    '''Any such section will contains at least two samples.'''
    diff_b = np.diff(b)
    i0s = np.nonzero(diff_b == 1)[0]
    i1s = np.nonzero(diff_b == -1)[0] + 1
    
    
    '''Sort the clusters by size because we are more confident on a cluster with more samples'''
    index = np.argsort(i1s - i0s)[::-1]
    i0s = i0s[index]
    i1s = i1s[index]
    
    '''Prepare the class labelling'''
    L = len(xx)
    labels = np.zeros(L)
    
    '''Label the clusters and calculate the class means'''
    class_means = []
    label = 1
    for i0, i1 in zip(i0s, i1s):
        labels[i0 : i1] = label
        label += 1
        class_means.append(np.mean(xx[i0 : i1]))
    
    '''Those leftover samples are individual points, each of which is a cluster''' 
    class_counts = (i1s - i0s).tolist()
    for i in range(L):
        if labels[i] == 0:
            labels[i] = label
            label += 1
            class_means.append(xx[i])
            class_counts.append(1)
    
    return labels, class_means, class_counts    

'''Remove outliers to meet the gap requirement (gap >= threshold)'''
def remove_outliers(x, class_counts, threshold):
    '''Sort and make a copy of the samples'''
    xx = np.sort(x).copy().tolist()
    
    '''Calculate the gaps'''
    diff_x = np.diff(xx)
    if any(diff_x < threshold):
        '''Locate all the bad points, which have large gap with neighbours'''
        diff_x_full = np.array([True] + diff_x.tolist() + [True])
        bad_points = np.logical_or(diff_x_full[1:] < threshold, diff_x_full[:-1] < threshold)
        indices = np.nonzero(bad_points)[0]
        
        '''Remove a subset of the bad points and check if the rest meet the gap requirement'''
        '''Starting from removing one bad point. As soon as as solution is found, terminate'''
        for i in range(1, len(indices) - 1):
            
            '''Find all the combinations for the subset'''
            combs = combinations(indices, i)
            solutions = []
            
            '''For each combination, if is a working solution, calculate:'''
            '''1. The class counts. We would like to remove those classes with less sample points (less confidence).'''
            '''2. Range. We would like to favour the larger range'''
            '''3. Standard deviation. We would like to favour those with smaller standard deviation by assuming sleepers are evenly distributed'''
            for comb in combs:
                xcopy = x.copy().tolist()
                
                '''Remove the possible outliers'''
                for i in comb[::-1]:
                    del(xcopy[i])
                    
                '''Check if it is a solution'''
                diff_x = np.diff(xcopy)
                if all(diff_x >= threshold):
                    '''Calculate the values for sorting later on'''
                    class_count = np.sum(class_counts[np.array(comb)])
                    solutions.append((comb, class_count, np.max(xcopy) - np.min(xcopy), np.std(diff_x)))
            if len(solutions) > 0:
                '''If we have multiple solutions, we would like to choose the best one based on the 3 criteria listed above'''
                sorted_solutions = sorted(solutions, key = lambda e : (-e[1], e[2], -e[3]))
                solution = sorted_solutions[-1][0]
                break
        for i in solution[::-1]:
            del(xx[i])
    return xx



'''Average a strip section parallel to the rails'''
'''The section in relation to the rail is defined by the offsets'''
def calculate_tie_strip(im, offsets):
    M, N = np.shape(im)
    rowsum = np.sum(im, 0)
    rails = np.nonzero(rowsum == 255 * M)[0]
    strip = np.mean(im[:, rails[0] + offsets[0] : rails[1] - offsets[0]], axis = 1)
    return strip, rails

'''Locate strong peaks in a curve'''
'''Each peak should have a minimum height of minHeight'''
'''They should be separated by a least minGap'''
'''There should not be more than maxNum of peaks'''
def locate_peaks(x, minHeight, minGap, maxNum):
    '''Locate strong peaks'''
    boolPeaks = np.diff(np.sign(np.diff(x))) <= -1 
    boolPeaks = np.array([False] + list(boolPeaks) + [False])
    boolStrongPeaks = np.logical_and(boolPeaks, x > minHeight)
    
    '''Sort the strong peaks'''
    index = np.argsort(x[boolStrongPeaks])
    peakCandidates = np.nonzero(boolStrongPeaks)[0][index[::-1]]
    
    '''Pick out the far apart peaks'''
    goodPeaks = []
    if len(peakCandidates) > 0:
        goodPeaks = [peakCandidates[0]]
        for peakCandidate in peakCandidates[1:]:
            if all(np.abs(peakCandidate - goodPeaks) > minGap):
                goodPeaks.append(peakCandidate)
            if len(goodPeaks) >= maxNum:
                break
    return goodPeaks

'''Locate the sleeper centre based on the intensity (sleeper has relatively higher reflectivity)'''
def locate_tie_centres_by_intensity(im1, im2):
    '''Calculate the vertical strips for tie calculation'''
    offsets = [220, 250]
    strip1, rails1 = calculate_tie_strip(im1, offsets)
    strip2, rails2 = calculate_tie_strip(im2, offsets)
    strip = np.concatenate((strip1, strip2))
    
    '''Top hat filter'''
    m = 100
    n = 200
    tophat = np.array([-0.5 / m] * m + [1.0 / n] * n + [-0.5 / m] * m)
    filteredStrip = ni.convolve(strip, tophat, mode = 'nearest')
    
    tieCentres = locate_peaks(filteredStrip, 10, 500, 5)

    # MM = np.shape(im1)[0] + np.shape(im2)[0]
    # plt.subplot(1,2,1)
    # plt.imshow(stack_images(im1, im2), interpolation = 'none', cmap = 'gray', aspect = 'auto')
    # for tieCentre in tieCentres:
    #     plt.plot([0, np.shape(im1)[1] - 1], [tieCentre, tieCentre], 'r--')
    # plt.subplot(1,2,2)
    # plt.plot(strip, range(len(strip)), label = 'Averaged Column')
    # plt.plot(filteredStrip, range(len(strip)), label = 'Tophat filtered')
    # plt.plot(filteredStrip[tieCentres], tieCentres, 'g.', label = 'Tie centres')
    # plt.ylim([0, MM])
    # plt.gca().invert_yaxis()
    # plt.grid(True)
    # plt.legend(loc = 0)
    # raise SystemExit

    return np.sort(np.array(tieCentres))

'''Locate the sleeper centre based on identified bolts next to the rails.'''
def locate_tie_centres_by_bolts(im1, im2):
    '''Calculate the vertical strips for tie calculation'''
    M, N = np.shape(im1)
    rails1 = np.nonzero(np.sum(im1, 0) == 255 * M)[0]
    rails2 = np.nonzero(np.sum(im2, 0) == 255 * M)[0]
    rails = ((rails1 + rails2) / 2 + 0.5).astype('int')
    im = stack_images(im1, im2)

    offsets = [60, 150]    
    strip1Im = im[:, rails[0] - offsets[1] : rails[0] - offsets[0]]
    strip2Im = im[:, rails[0] + offsets[0] : rails[0] + offsets[1]]
    strip3Im = im[:, rails[1] - offsets[1] : rails[1] - offsets[0]]
    strip4Im = im[:, rails[1] + offsets[0] : rails[1] + offsets[1]]
    profile1 = np.mean(strip1Im, 0)
    profile2 = np.mean(strip2Im, 0)
    profile3 = np.mean(strip3Im, 0)
    profile4 = np.mean(strip4Im, 0)

    grad1 = np.diff(profile1)
    grad2 = np.diff(profile2)
    grad3 = np.diff(profile3)
    grad4 = np.diff(profile4)
    
    # plt.plot(grad1)
    # plt.plot(grad2)
    # plt.plot(grad3)
    # plt.plot(grad4)
    # plt.grid(True)
    # raise SystemExit
    row1 = np.argmax(np.abs(grad1))
    row2 = np.argmax(np.abs(grad2))
    row3 = np.argmax(np.abs(grad3))
    row4 = np.argmax(np.abs(grad4))
    
    width = 20
    strip1 = np.mean(strip1Im[:, row1 : row1 + width], axis = 1)
    strip2 = np.mean(strip2Im[:, row2 - width : row2], axis = 1)
    strip3 = np.mean(strip3Im[:, row3 : row3 + width], axis = 1)
    strip4 = np.mean(strip4Im[:, row4 - width : row4], axis = 1)
   
    m = 30
    n = 60
    tophat1 = np.array([-0.5 / m] * m + [1.0 / n] * n + [-0.5 / m] * m)
    
    m = 30
    n = 20
    tophat2 = np.array([-1.0 / m] * m + [2.0 / n] * n + [-2.0 / n] * n + [2.0 / n] * n + [-1.0 / m] * m) 
    fStrip1 = ni.convolve(strip1, tophat1, mode = 'mirror')
    fStrip2 = ni.convolve(strip2, tophat1, mode = 'mirror')
    fStrip3 = ni.convolve(strip3, tophat1, mode = 'mirror')
    fStrip4 = ni.convolve(strip4, tophat2, mode = 'mirror')
    peakThreshold = 30
    peakMinGap = 500
    maxNumPeaks = 5
    peaks1 = locate_peaks(-fStrip1, peakThreshold, peakMinGap, maxNumPeaks)
    peaks2 = locate_peaks(-fStrip2, peakThreshold, peakMinGap, maxNumPeaks)
    peaks3 = locate_peaks(-fStrip3, peakThreshold, peakMinGap, maxNumPeaks)
    peaks4 = locate_peaks(-fStrip4, peakThreshold, peakMinGap, maxNumPeaks)
    
    # M, N = np.shape(im)
    # plt.figure(figsize = (12, 8))
    # plt.subplot(4,1,1)
    # plt.imshow(strip1Im.T, interpolation = 'none', cmap = 'gray', aspect = 'auto')
    # plt.plot([0, M - 1], [row1, row1], 'r--')
    # plt.subplot(4,1,2)
    # plt.plot(strip1)
    # plt.plot(fStrip1)
    # plt.plot(peaks1, fStrip1[peaks1], '.')
    # plt.xlim([0, np.shape(im)[0]])
    # plt.grid(True)
    
    # plt.subplot(4,1,3)
    # plt.imshow(strip2Im.T, interpolation = 'none', cmap = 'gray', aspect = 'auto')
    # plt.plot([0, M - 1], [row2, row2], 'r--')    
    # plt.subplot(4,1,4)
    # plt.plot(strip2)
    # plt.plot(fStrip2)    
    # plt.plot(peaks2, fStrip2[peaks2], '.')    
    # plt.xlim([0, np.shape(im)[0]])
    # plt.grid(True)
   
    # plt.figure(figsize = (12, 8))
    # plt.subplot(4,1,1)
    # plt.imshow(strip3Im.T, interpolation = 'none', cmap = 'gray', aspect = 'auto')
    # plt.plot([0, M - 1], [row3, row3], 'r--')
    # plt.subplot(4,1,2)
    # plt.plot(strip3)
    # plt.plot(fStrip3)    
    # plt.plot(peaks3, fStrip3[peaks3], '.')    
    # plt.xlim([0, np.shape(im)[0]])
    # plt.grid(True)
    
    # plt.subplot(4,1,3)
    # plt.imshow(strip4Im.T, interpolation = 'none', cmap = 'gray', aspect = 'auto')
    # plt.plot([0, M - 1], [row4, row4], 'r--')    
    # plt.subplot(4,1,4)
    # plt.plot(strip4)
    # plt.plot(fStrip4)
    # plt.plot(peaks4, fStrip4[peaks4], '.')    
    # plt.xlim([0, np.shape(im)[0]])
    # plt.grid(True)
    # raise SystemExit

    peaks = np.sort(peaks1 + peaks2 + peaks3 + peaks4)
    
    labels, class_means, class_counts = simple_clustering(peaks, 30)

    index = np.argsort(class_means)
    class_means = np.array(class_means)[index]
    class_counts = np.array(class_counts)[index]

    tieCentres = remove_outliers(class_means, class_counts, 400)
    
    # gap_indices = np.nonzero(np.diff(peaks) > 450)[0]
    # indices = [0] + list(gap_indices + 1) + [len(peaks)]
    # tieCentres = []
    # print(np.diff(indices))
    # for i0, i1 in zip(indices[:-1], indices[1:]):
    #     if i1 - i0 > 1:
    #         tieCentres.append(np.mean(peaks[i0 : i1]))
    # tieCentres = np.sort(np.array(tieCentres))

    return tieCentres

'''Match the sleepers identified in the upper image with the ones identified in the previous lower image'''
def matchTies(x, y, logfn):
    matchThreshold = 50
    M = len(x)
    N = len(y)
    if M == 0:
        '''Previous image has no tie to match'''
        f = open(logfn, 'a')
        f.write('Previous image has no tie to match.\n')
        f.close()
        match_index = -1
    elif N == 0:
        f = open(logfn, 'a')
        f.write('Current image has no tie to match.\n')
        f.close()
        '''Current image has no tie to match'''
        match_index = -1
    else:
        xx = np.array([x] * N).T
        yy =np.array([y] * M)
        d = np.abs(xx - yy)
        if np.min(d) < matchThreshold:
            row, col = np.unravel_index(np.argmin(d), np.shape(d))
            match_index = col + (M - row - 1)
        else:
            f = open(logfn, 'a')
            f.write('Failed to match a tie, start from the first.\n')
            f.close()
            match_index = -2
    return match_index

def draw_ties(rgbIm, tieCentres):
    M, N, K = np.shape(rgbIm)
    halfTieWidth = int(210 / 2)
    for tieCentre in tieCentres:
        rgbIm[tieCentre, :, 0] = 255
        rgbIm[tieCentre, :, 1] = 0
        rgbIm[tieCentre, :, 2] = 0
        tieTop = tieCentre + halfTieWidth
        if tieTop < M:
            rgbIm[tieTop, :, 0] = 255
            rgbIm[tieTop, :, 1] = 0
            rgbIm[tieTop, :, 2] = 0
        tieBottom = tieCentre - halfTieWidth
        if tieBottom > 0:
            rgbIm[tieBottom, :, 0] = 255
            rgbIm[tieBottom, :, 1] = 0
            rgbIm[tieBottom, :, 2] = 0
            
'''Processing parameters'''
tieHalfWidth = 120
railHalfWidth = 130
runNum = 364

logfn = 'log.txt'
f = open(logfn, 'w')
f.write('Log file\n')
f.close()

'''File paths'''
finPath = 'Output\\Stitched\\Run_%03d\\' % runNum
foutPath1 = 'Output\\Cribs\\Run_%03d\\' % runNum
foutPath2 = 'Output\\Drawn\\Run_%03d\\' % runNum
if not os.path.exists(foutPath1):
    os.mkdir(foutPath1)
if not os.path.exists(foutPath2):
    os.mkdir(foutPath2)

'''Output drawings for results checking'''
draw = True
fns = glob(finPath + '*.png')
cribNum = 0
preTieCentres = []

'''Combine every pair of images and locate sleepers'''
for fn1, fn2 in zip(fns[:-1], fns[1:]):
    im1 = imread(fn1)
    M1 = np.shape(im1)[0]
    im2 = imread(fn2)

    shortfn1 = fn1[fn1.rfind('\\') + 1:-4]
    shortfn2 = fn2[fn2.rfind('\\') + 1:-4]

    f = open(logfn, 'a')
    f.write('Processing %s\n' % shortfn1)
    f.close()
    print('Processing %s' % shortfn1)
    
    '''Locate the rail centre lines by the unique value 255'''
    rails1 = np.nonzero(np.all(im1 == 255, axis = 0))[0]
    rails2 = np.nonzero(np.all(im2 == 255, axis = 0))[0]
    rails = ((rails1 + rails2) / 2 + 0.5).astype('int')

    '''Stack two images up and extract the section between two rails'''
    im = stack_images(im1, im2)
    M, N = np.shape(im)
    betweenRailsIm = im[:, rails[0] + railHalfWidth : rails[1] - railHalfWidth]
    
    '''Locate the sleepers centres by the intensity method'''
    tieCentres = locate_tie_centres_by_intensity(im1, im2).astype('int')
    
    '''If it fails, use the bolts next the rails to locate sleepers'''
    # if len(tieCentres) < 3:
    #     f = open(logfn, 'a')
    #     f.write('Less than 3 sleepers found, use the bolt algorithm to locate sleepers')
    #     f.close()
    #     tieCentres = np.array(locate_tie_centres_by_bolts(im1, im2)).astype('int')
    
    '''Prepare an image for diagnostic drawing'''
    rgbIm = gray2rgb(im)            
    rgbIm = rgbIm.copy()
    save_drawing = True
    
    if len(tieCentres) > 0:
        '''Check the section before the first tie.'''
        '''If the first tie is partly in the lower image, we can't do tie matching.'''
        '''We have to export the section up to the edge of the first tie identified,'''
        '''starting from the first row of the upper image'''
        if tieCentres[0] + tieHalfWidth >= M1:
            cribNum += 1
            cribIm = betweenRailsIm[:tieCentres[0] - tieHalfWidth, :]
            imwrite(foutPath1 + 'Run%03d_Image%s_Crib%04d.png' % (runNum, shortfn1, cribNum), cribIm)
    
            if draw:
                cribCentre = (tieCentres[0] - tieHalfWidth) / 2
            cv2.putText(rgbIm, str(cribNum), (100, int(cribCentre)), fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = 3, color = (255, 0, 0), thickness = 3)

        '''Export the section between every pair of ties for the upper image.'''
        if len(tieCentres) >= 2:
            '''Match with the previous image'''
            match_index = matchTies(preTieCentres, tieCentres, logfn)
            if match_index < 0:
                match_index = 0
            for i in range(match_index, len(tieCentres) - 1):
                cribNum += 1
                cribCentre = (tieCentres[i] + tieCentres[i + 1]) / 2
                if cribCentre < M1:
                    imageNum = shortfn1
                else:
                    imageNum = shortfn2
                cribIm = betweenRailsIm[tieCentres[i] + tieHalfWidth : tieCentres[i + 1] - tieHalfWidth, :]
                imwrite(foutPath1 + 'Run%03d_Image%s_Crib%04d.png' % (runNum, imageNum, cribNum), cribIm)
                if draw:
                    cv2.putText(rgbIm, str(cribNum), (100, int(cribCentre)), fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = 3, color = (255, 0, 0), thickness = 3)

        '''If no tie is identified in the lower image, we have to export the section starting from the'''
        '''lower edge of the last tie up to the last row of the upper image'''
        '''Also, we can't match the tie to the next pair of images.'''
        if tieCentres[-1] + tieHalfWidth < M1:
            cribNum += 1
            cribIm = betweenRailsIm[tieCentres[-1] + tieHalfWidth : M1]
            imwrite(foutPath1 + 'Run%03d_Image%s_Crib%04d.png' % (runNum, shortfn1, cribNum), cribIm)
            if draw:
                cribCentre = (tieCentres[-1] + tieHalfWidth + M1) / 2
                cv2.putText(rgbIm, str(cribNum), (100, int(cribCentre)), fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = 3, color = (255, 0, 0), thickness = 3)

        if draw:    
            '''Draw ties'''
            draw_ties(rgbIm, tieCentres)   
        preTieCentres = tieCentres - M1
    else:
        cribNum += 1
        cribIm = betweenRailsIm[:M1, :]
        imwrite(foutPath1 + 'Run%03d_Image%s_Crib%04d.png' % (runNum, shortfn1, cribNum), cribIm)
        preTieCentres = []
        if draw:
            cribCentre = M1 / 2
            cv2.putText(rgbIm, str(cribNum), (100, int(cribCentre)), fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = 3, color = (255, 0, 0), thickness = 3)


    if save_drawing:
        plt.imsave(foutPath2 + '%s_%s.png' % (shortfn1,shortfn2), rgbIm)

