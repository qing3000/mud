# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 21:01:49 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from step7_testing_cribs import fixedHist
import glob 
import imagesize 

def get_hist(fpath):
    fns = glob.glob(fpath + '*.png')
    widths = []
    heights = []
    for i, fn in enumerate(fns):
        if i % 1000 == 0:
            print('%d out of %d' % (i, len(fns)))
        width, height = imagesize.get(fn)
        #im = imread(fn)
        #M, N = np.shape(im)
        widths.append(width)
        heights.append(height)
        
    hx1, hy1 = fixedHist(widths, 2)
    hx2, hy2 = fixedHist(heights, 2)
    return hx1, hy1, hx2, hy2


wx1, wy1, hx1, hy1 = get_hist('Output\\Cribs\\Run_132\\')
wx2, wy2, hx2, hy2 = get_hist('Output\\Cribs\\Run_354\\')
wx3, wy3, hx3, hy3 = get_hist('Output\\Cribs\\Run_364_batch1\\')
wx4, wy4, hx4, hy4 = get_hist('Output\\Cribs\\Run_364_batch2\\')
wx5, wy5, hx5, hy5 = get_hist('Output\\Cribs\\Run_364_batch3\\')
                              
plt.plot(wx1, wy1, label = 'Run132 crib width')
plt.plot(hx1, hy1, label = 'Run132 crib height')
plt.plot(wx2, wy2, label = 'Run354 crib width')
plt.plot(hx2, hy2, label = 'Run354 crib height')
plt.plot(wx3, wy3, label = 'Run364 batch1 crib width')
plt.plot(hx3, hy3, label = 'Run364 batch1 crib height')
plt.plot(wx4, wy4, label = 'Run364 batch2 crib width')
plt.plot(hx4, hy4, label = 'Run364 batch2 crib height')
plt.plot(wx5, wy5, label = 'Run364 batch3 crib width')
plt.plot(hx5, hy5, label = 'Run364 batch3 crib height')
plt.grid(True)
plt.legend(loc = 0)
plt.xlim([0, 1500])
plt.title('Crib image width and height histograms')
