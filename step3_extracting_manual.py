# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:45:07 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite

csvfile = 'Data\\AMTrak\\all.csv'
fns = np.loadtxt(csvfile, dtype = str, delimiter = ',', usecols = 0, skiprows = 1)
labels = np.loadtxt(csvfile, dtype = str, delimiter = ',', usecols = 5, skiprows = 1)
ulabels = np.unique(labels)
x1, x2, y1, y2 = np.loadtxt(csvfile, delimiter = ',', usecols = (1, 2, 3, 4), skiprows = 1, unpack = True)

fpath = '..\\AMTrakImages\\'
foutpath = '..\\AMTrakCribs\\'
pre_fn = fns[0]
crib_counter = 0
for i, fn in enumerate(fns):
    im = imread(fpath + fn[:-5])
    M, N, K = np.shape(im)
    row1 = int(y1[i] * M + 0.5)
    row2 = int(y2[i] * M + 0.5)
    col1 = int(x1[i] * N + 0.5)
    col2 = int(x2[i] * N + 0.5)
    crib = im[row1:row2 ,  col1:col2 , :]
    if fn != pre_fn:
        crib_counter = 0
    if labels[i] == ulabels[3]:
        if np.abs(col1 + col2 - N) < 100:
            if fn == pre_fn:
                crib_counter += 1
            crib_fn = fn[:-9] + '_crib_%d' % crib_counter + '.png'
            # imwrite(foutpath + 'Mud\\' + crib_fn, im[row1:row2, col1:col2, 0])
            plt.imshow(im)
            plt.plot([col1, col2, col2, col1, col1], [row1, row1, row2, row2, row1], 'r')
            plt.ginput()
            plt.close()

        
    pre_fn = fn
    #raise SystemExit