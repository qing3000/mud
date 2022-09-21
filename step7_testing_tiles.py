# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:40:46 2022

@author: zhangq
"""
import numpy as np
from tensorflow.keras import models
import keras.backend as K
import matplotlib.pyplot as plt
from glob import glob
from imageio import imread
from step6_training_tiles import shuffle_filenames_and_labels, read_image
from random import shuffle

def fixedHist(x, binWidth):
    binEdges = np.arange(np.min(x) - binWidth / 2.0, np.max(x) + binWidth / 2.0, binWidth)
    if binEdges[-1] < np.max(x): 
        binEdges = np.append(binEdges, np.max(x) + binWidth / 2.0)
    hy, binEdges = np.histogram(x, binEdges, density = True)
    hx = (binEdges[:-1] + binEdges[1:]) / 2
    return hx, hy

'''Calculate the output values from a particular layer of the CNN'''
def cnn_get_output(fns, get_output_func, blockSize = 50):
    '''Due to memory limitation, we have to call the output function by a block size each time and combine at the end'''
    outputs = []
    for i in range(0, len(fns), blockSize):
        images = np.array([read_image(fn) for fn in fns[i : i + blockSize]])   
        print('Get output %d out of %d' % (i, len(fns)))
        outputs += [get_output_func([images])]
    outputs = np.concatenate(outputs, 0)
    return outputs

#====================================
if __name__ == '__main__':  

    runNum = 364
    
    rootPath = '.\\'
    
    print('Load in the traning file names')
    cleanFns = glob('..\\AMTrakTiles\\Clean\\*.png', recursive = True)
    muddyFns = glob('..\\AMTrakTiles\\Mud\\*.png', recursive = True)

    print('Load in the pretrained CNN model')
    model = models.load_model('CNN_Model_Tiles')
    get_output_func = K.function([model.layers[0].input], model.layers[-2].output)
    
    print('Calculate the output')
    '''Due to memory limitation, we have to call the output function by a block size each time and combine at the end'''
    clean_values = cnn_get_output(cleanFns, get_output_func)[:, 0]
    muddy_values = cnn_get_output(muddyFns, get_output_func)[:, 0]
    x = clean_values[:-800]
    y = muddy_values[:-800]
    lda = (np.mean(x) - np.mean(y))**2 / (np.var(x) + np.var(y))
    
    '''Save the training output values for reference'''
    f = open('Output\\Clean_Tiles_values.csv', 'w')
    f.write('\n'.join(map(str, clean_values)))
    f.close()
    f = open('Output\\Muddy_Tiles_values.csv', 'w')
    f.write('\n'.join(map(str, muddy_values)))
    f.close()


    clean_values = np.loadtxt('Output\\Clean_Tiles_values.csv')
    muddy_values = np.loadtxt('Output\\Muddy_Tiles_values.csv')
    '''Calculate the histogram distributions'''
    print('Calculate the histograms')
    binSize = 1
    hx_clean, hy_clean = fixedHist(clean_values[:-800], binSize)
    hx_muddy, hy_muddy = fixedHist(muddy_values[:-800], binSize)
    hx_test_clean, hy_test_clean = fixedHist(clean_values[:800], binSize)
    hx_test_muddy, hy_test_muddy = fixedHist(muddy_values[:800], binSize)
    
    '''Plot the histogram distributions'''
    plt.plot(hx_clean, hy_clean, label = 'Training (clean)')
    plt.plot(hx_muddy, hy_muddy, label = 'Training (muddy)')
    plt.plot(hx_test_clean, hy_test_clean, label = 'Testing (clean)')
    plt.plot(hx_test_muddy, hy_test_muddy, label = 'Testing (muddy)')
    plt.legend(loc = 0)
    plt.grid(True)
    plt.title('Network output distributions LDA =%g (tile images)' % lda)
