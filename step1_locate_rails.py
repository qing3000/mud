# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:52:02 2022

@author: zhangq
"""

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import scipy.ndimage as ni
from imageio import imread

def detect_rail(im, rng):
    M, N = np.shape(im)
    imf = im.astype('float')
    x = np.arange(M)
    p = np.polyfit(x, imf[:, rng[0]:rng[1]], 2)
    im_residual = imf.copy()
    for k in range(rng[0], rng[1]):
        im_residual[:, k] -= np.polyval(p[:, k - rng[0]], x)
    p1 = np.std(im_residual, 0)
    m = 80
    n = 110
    tophat = np.array([0.5 / m] * m + [-1.0 / n] * n + [0.5 / m] * m)
    p2 = ni.convolve(p1, tophat, mode = 'mirror')    
    railCentre = np.argmax(p2[rng[0] : rng[1]]) + rng[0]
    #When the rail is close to the edge of the image, the tophat filter will fail because it misses a wing.
    #We have to use a smooth filter to locate the rail centre.
    if railCentre < n / 2: 
        halfhat = np.array([1.0 / m] * m + [-1.0 / n] * n)
        p3 = ni.convolve(p1, halfhat, mode = 'mirror')    
        railCentre = np.argmax(p3[rng[0] : rng[1]]) + rng[0]

    # plt.subplot(3,1,1)
    # plt.imshow(im, interpolation = 'none', cmap = 'gray', aspect = 'auto', vmax = 30)
    # plt.colorbar()
    # plt.title('Raw image')
    # plt.subplot(3,1,2)
    # plt.imshow(im_residual, interpolation = 'none', cmap = 'gray', aspect = 'auto', vmax = 30)
    # plt.colorbar()
    # plt.title('Polyfit residual image')
    # plt.subplot(3,1,3)
    # plt.plot(p1, label = 'Standard deviation of each row')
    # plt.plot(p2, label = 'Tophat filtered')
    # if 'p3' in locals():
    #     plt.plot(p3, label = 'Halfhat filtered')
    # plt.plot([railCentre, railCentre], [np.min(p1), np.max(p1)],'--r', label = 'Estimated Rail Centre')
    # plt.xlim([0, 1023])
    # plt.colorbar()
    # plt.grid(True)
    # plt.legend(loc = 0)
    # raise SystemExit

    return railCentre

def calculate_rail_centres(fns, rng):
    railCentres = []
    for i, fn in enumerate(fns):
        if i % 100 == 0:
            print('Calculate rail centres: %d out of %d' % (i, len(fns)))
        im = imread(fn)
        railCentres.append(detect_rail(im, rng))
    smoothRailCentres = ni.convolve(railCentres, np.ones(20) / 20)
    dev = railCentres - smoothRailCentres
    valid = np.abs(dev) < np.std(dev) * 2
    x = np.arange(len(fns))
    newRailCentres = np.interp(x, x[valid], np.array(railCentres)[valid])
    smoothNewRailCentres = ni.convolve(newRailCentres, np.ones(20) / 20)
    # plt.subplot(2,1,1)
    # plt.plot(railCentres, label = 'Rail centres (first round)')
    # plt.plot(smoothRailCentres, label = 'Smoothed version')
    # plt.plot(newRailCentres, label = 'interpolated version')
    # plt.plot(smoothNewRailCentres, label = 'Smoothed version')    
    # plt.legend(loc = 0)
    # plt.grid(True)
    # plt.subplot(2,1,2)
    # plt.plot(np.abs(railCentres - smoothRailCentres))
    # plt.grid(True)
    # plt.title(np.std(railCentres - smoothRailCentres))
    # raise SystemExit
        
    return smoothNewRailCentres
        

dataPath = 'C:\\Personal\\Mudspots\\Run_354-20200216@032951_08000-13966\\'

'''Get all the image filenames'''
fpaths = glob(dataPath + 'Run*')
fns = []
for fpath in fpaths:
    fns += glob(fpath + '\\*.jpg')

'''Work out the rail centre lines for each camera by a two-iteration method'''
print('Calculate rail centre lines for camera 1')
railCentres1 = calculate_rail_centres(fns[::4], [500, 1000])
print('Calculate rail centre lines for camera 2')
railCentres2 = calculate_rail_centres(fns[1::4], [0, 400])
print('Calculate rail centre lines for camera 3')
railCentres3 = calculate_rail_centres(fns[2::4], [500, 1000])
print('Calculate rail centre lines for camera 4')
railCentres4 = calculate_rail_centres(fns[3::4], [0, 500])

railCentres = np.array([railCentres1, railCentres2, railCentres3, railCentres4])
np.savetxt('RailCentreLines.csv', railCentres.T, fmt = '%d', delimiter = ',')

plt.plot(railCentres1, label = 'Camera 1')
plt.plot(railCentres2, label = 'Camera 2')
plt.plot(railCentres3, label = 'Camera 3')
plt.plot(railCentres4, label = 'Camera 4')
plt.legend(loc = 0)
plt.grid(True)



