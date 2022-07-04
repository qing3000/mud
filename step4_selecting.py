# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:09:26 2022

@author: zhangq
"""
import numpy as np
import os
import glob
from shutil import copy
from imageio import imread, imwrite
import matplotlib.pyplot as plt


runNum = 364
finPath = 'Output\\Cribs\\Run_%03d_batch1\\' % runNum 
foutPath = 'ForCNN\\MuddyCribs\\Run_%03d\\' % runNum
if not os.path.exists(foutPath):
    os.mkdir(foutPath)

keyword = 'Run_132-20190424@105356_38000-48000'
keyword = 'Run_354-20200216@032951_08000-13966'
keyword = 'Run_364-20200424@011547_52000-62000'
csvfile = 'C:\Personal\Mudspots\%s\\%s_Rankings.csv' % (keyword, keyword)    

imgNums, smi = np.loadtxt(csvfile, delimiter = ',', skiprows = 1, usecols = (0, 3), unpack = True)
selectedImgNums = imgNums[smi == 3].astype('int')


# L = len(selectedImgNums)
# rng = np.linspace(0, L - 1, 150).astype('int')
# selectedImgNums = selectedImgNums[rng]
fns = glob.glob(finPath + '*.png')
for fn in fns:
    shortfn = fn[fn.rfind('\\') + 1:-4]
    ss = shortfn.split('_')
    imageNum = int(ss[1][5:])
    if imageNum in selectedImgNums:
        im = imread(fn)
        copy(fn, foutPath)
