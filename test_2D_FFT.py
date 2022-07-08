# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:13:03 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from imageio import imread
from numpy.fft import fft2, fftshift
from scipy.ndimage import zoom
import glob
def calculate_ring_max(im, cuts):
    M, N = np.shape(im)
    cc = [int(M / 2 + 0.5), int(N / 2 + 0.5)]
    im_copy = im.copy()
    im_copy[cc[0] - cuts[0]: cc[0] + cuts[0], cc[1] - cuts[0] : cc[1] + cuts[0]] = 0
    block = im_copy[cc[0] - cuts[1]: cc[0] + cuts[1], cc[1] - cuts[1] : cc[1] + cuts[1]]
    vmax = np.max(block)
    return vmax, block

def calculate_fft2(im):
    empty_cols = np.all(im == 255, axis = 0)
    empty_rows = np.all(im == 255, axis = 1)
    if any(empty_cols) or any(empty_rows):
        good_cols = np.logical_not(empty_cols)
        good_rows = np.logical_not(empty_rows)
        im1 = im[good_rows]
        im2 = im1[:, good_cols]
        im3 = im2 - np.mean(im2)
        sfim = fftshift(fft2(im3))
        fim_mag_small = np.abs(sfim)
        M, N = np.shape(im)
        MM, NN = np.shape(im2)
        fim_mag = zoom(fim_mag_small, [M / MM, N / NN])
    else:
        im2 = im - np.mean(im)
        sfim = fftshift(fft2(im2))
        fim_mag = np.abs(sfim)
    return fim_mag, im2

#====================================
if __name__ == '__main__':  
    
    fns = glob.glob('Output\\CleanCribs\\Run_364\\*.png')
    
    cuts = [16, 32, 64]
    clrmap = cm.jet
    clrmap.set_under('white')
    for fn in fns[:50]:
        im = imread(fn)[:, :500]
        fim_mag, im = calculate_fft2(im)
        vmax1, block1 = calculate_ring_max(fim_mag, [0, cuts[0]])
        vmax2, block2 = calculate_ring_max(fim_mag, [cuts[0], cuts[1]])
        vmax3, block3 = calculate_ring_max(fim_mag, [cuts[1], cuts[2]])
        vmax4, block4 = calculate_ring_max(fim_mag, [cuts[2], 128])
    
        plt.subplot(1,2,1)
        plt.imshow(im, interpolation = 'none', cmap = 'gray')
        plt.title('Crib image (left half)')
        plt.subplot(2,4,3)
        plt.imshow(block1, interpolation = 'none', cmap = clrmap, vmin = 1e-9, vmax = 500000)
        #plt.colorbar()
        plt.title('2D FFT block 1')
        plt.subplot(2,4,4)
        plt.imshow(block2, interpolation = 'none', cmap = clrmap, vmin = 1e-9,  vmax = vmax2)
        plt.title('2D FFT ring 1')
        plt.subplot(2,4,7)
        plt.imshow(block3, interpolation = 'none', cmap = clrmap, vmin = 1e-9,  vmax = vmax3)
        plt.title('2D FFT ring 2')        
        plt.subplot(2,4,8)
        plt.imshow(block4, interpolation = 'none', cmap = clrmap, vmin = 1e-9,  vmax = vmax4)
        plt.title('2D FFT ring 3')       
        shortfn = fn[fn.rfind('\\' ) + 1:-4]
        plt.savefig('Diagnostics\\TrueClean\\%s.jpg' % shortfn, dpi = 600, bbox_inches = 'tight')
        plt.close()
