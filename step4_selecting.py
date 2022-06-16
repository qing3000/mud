# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:09:26 2022

@author: zhangq
"""
import numpy as np
import os
import glob
from shutil import copy

runNum = 364
finPath = 'Output\\Cribs\\Run_%03d\\' % runNum 
foutPath = 'Output\\MuddyCribs\\Run_%03d\\' % runNum
if not os.path.exists(foutPath):
    os.mkdir(foutPath)

csvfile = 'C:\\Personal\\Mudspots\\Run_364-20200424@011547_52000-62000\\Run_364-20200424@011547_52000-62000_Rankings.csv'    

imgNums, smi = np.loadtxt(csvfile, delimiter = ',', skiprows = 1, usecols = (0, 3), unpack = True)
muddyImgNums = imgNums[smi > 1]

fns = glob.glob(finPath + '*.png')
for fn in fns:
    shortfn = fn[fn.rfind('\\') + 1:-4]
    ss = shortfn.split('_')
    imageNum = int(ss[1][5:])
    if imageNum in muddyImgNums:
        copy(fn, foutPath)
