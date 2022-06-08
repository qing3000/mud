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
from step6_training import shuffle_filenames_and_labels, read_image


def fixedHist(x, binWidth):
    binEdges = np.arange(np.min(x) - binWidth / 2.0, np.max(x) + binWidth / 2.0, binWidth)
    if binEdges[-1] < np.max(x): 
        binEdges = np.append(binEdges, np.max(x) + binWidth / 2.0)
    hy, binEdges = np.histogram(x, binEdges, density = True)
    hx = (binEdges[:-1] + binEdges[1:]) / 2
    return hx, hy

'''Calculate the output values from a particular layer of the CNN'''
def cnn_get_output(fns, get_output_func, blockSize = 500):
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

    runNum = 354
    
    rootPath = '.\\'
    
    print('Load in the traning file names')
    cleanFns = glob(rootPath + 'ForCNN\\CleanBlocks\\*.png')
    muddyFns = glob(rootPath + 'ForCNN\\MuddyBlocks\\*.png')
    train_fns, train_labels = shuffle_filenames_and_labels(cleanFns, muddyFns)
    
    print('Load in the test file names')
    test_fns = glob(rootPath + 'Output\\Blocks\\Run_%03d\\*.png' % runNum)
    
    print('Load in the pretrained CNN model')
    model = models.load_model('CNN_Model')
    get_output_func = K.function([model.layers[0].input], model.layers[-2].output)
    
    print('Calculate the output')
    '''Due to memory limitation, we have to call the output function by a block size each time and combine at the end'''
    train_output = cnn_get_output(train_fns, get_output_func)
    test_output = cnn_get_output(test_fns, get_output_func)
    
    '''Calculate the output values of the second last layer'''
    clean_c1_values = train_output[train_labels == 0, 0]
    muddy_c1_values = train_output[train_labels == 1, 0]
    clean_c2_values = train_output[train_labels == 0, 1]
    muddy_c2_values = train_output[train_labels == 1, 1]
    test_c1_values = test_output[:, 0]
    test_c2_values = test_output[:, 1]
    
    '''Calculate the histogram distributions'''
    print('Calculate the histograms')
    binSize = 0.2
    hx_c1_clean, hy_c1_clean = fixedHist(clean_c1_values, binSize)
    hx_c1_muddy, hy_c1_muddy = fixedHist(muddy_c1_values, binSize)
    hx_c2_clean, hy_c2_clean = fixedHist(clean_c2_values, binSize)
    hx_c2_muddy, hy_c2_muddy = fixedHist(muddy_c2_values, binSize)
    hx_c1_test, hy_c1_test = fixedHist(test_c1_values, binSize)
    hx_c2_test, hy_c2_test = fixedHist(test_c2_values, binSize)
    
    '''Plot the histogram distributions'''
    plt.subplot(2,1,1)
    plt.plot(hx_c1_clean, hy_c1_clean, label = 'Classifier 1 output (clean)')
    plt.plot(hx_c1_muddy, hy_c1_muddy, label = 'Classifier 1 output (muddy)')
    plt.plot(hx_c1_test, hy_c1_test, label = 'Classifier 1 output (Clean test)')
    plt.legend(loc = 0)
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(hx_c2_clean, hy_c2_clean, label = 'Classifier 2 output (clean)')
    plt.plot(hx_c2_muddy, hy_c2_muddy, label = 'Classifier 2 output (muddy)')
    plt.plot(hx_c2_test, hy_c2_test, label = 'Classifier 2 output (Clean test)')
    plt.grid(True)
    plt.legend(loc = 0)
    
    
    '''Classification'''
    threshold = -1.0
    labels = test_c1_values < threshold
    
    '''Output classification results to csv'''
    f = open('Output\\Run%03d_result.csv' % runNum, 'w')
    f.write('Image#,Crib#,Row,Column,Class 1 Value,Class 2 Value,Classification\n')
    for i in range(len(test_fns)):
        fn = test_fns[i]
        shortfn = fn[fn.rfind('\\') + 1:fn.rfind('.')]
        ss = shortfn.split('_')
        imageNum = int(ss[1][5:])
        cribNum = int(ss[2][4:])
        rowNum = int(ss[3][3:])
        colNum = int(ss[4][3:])
        f.write('%d,%d,%d,%d,%f,%f,%d\n' % (imageNum, cribNum, rowNum, colNum, test_c1_values[i], test_c2_values[i], labels[i]))
    f.close()
    
