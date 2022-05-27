# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:09:26 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from shutil import copy

rootPath = 'E:\\TestData\\rr\\Result_ALL\\'
runNum = 132
finPath = rootPath + 'Cribs\\Run_%03d\\' % runNum 
foutPath = rootPath + 'CleanCribs\\Run_%03d\\' % runNum
if not os.path.exists(foutPath):
    os.mkdir(foutPath)

csvfile = 'Run_132-20190424@105356_38000-48000\\Run_132-20190424@105356_38000-48000_Rankings.csv'    

imgNums, smi = np.loadtxt(csvfile, delimiter = ',', skiprows = 1, usecols = (0, 3), unpack = True)
muddyImgNums = imgNums[smi == 1]

fns = glob.glob(finPath + '*.png')
for fn in fns:
    shortfn = fn[fn.rfind('\\') + 1:-4]
    ss = shortfn.split('_')
    imageNum = int(ss[1][5:])
    if imageNum in muddyImgNums:
        copy(fn, foutPath)
