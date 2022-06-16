# -*- coding: utf-8 -*-
"""
Created on Sun May  1 16:42:46 2022

@author: zhangq
"""

import numpy as np
import glob
import os
from imageio import imread, imwrite

runNum = 364
fns = glob.glob('Output\\MuddyCribs\\Run_%03d\\*.png' % runNum)

foutPath = 'Output\\MuddyBlocks\\Run_%03d\\' % runNum
if not os.path.exists(foutPath):
    os.mkdir(foutPath)

blockWidth = 256
blockHeight = 256
ignoreFraction = 0.7
for j, fn in enumerate(fns):
    if j % 10 == 0:
        print('%d out of %d' % (j, len(fns)))
    im = imread(fn)
    # plt.imshow(im, interpolation = 'none', cmap = 'gray')
    # plt.xlim([0, 1024])
    # plt.ylim([512, 0])
    # plt.gca().set_xticks(range(0, 1024, blockWidth))
    # plt.gca().set_yticks(range(0, 512, blockWidth))
    # plt.grid(True)
    # raise SystemExit
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
    