# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:40:46 2022

@author: zhangq
"""
import numpy as np
# import tensorflow as tf
from tensorflow.keras import models
# from tensorflow.keras.utils import plot_model
import keras.backend as K
import matplotlib.pyplot as plt
from glob import glob
from imageio import imread

#from step6_training import read_images_labels

def fixedHist(x, binWidth):
    binEdges = np.arange(np.min(x) - binWidth / 2.0, np.max(x) + binWidth / 2.0, binWidth)
    if binEdges[-1] < np.max(x): 
        binEdges = np.append(binEdges, np.max(x) + binWidth / 2.0)
    hy, binEdges = np.histogram(x, binEdges, density = True)
    hx = (binEdges[:-1] + binEdges[1:]) / 2
    return hx, hy

'''Calculate the output values from a particular layer of the CNN'''
def cnn_get_output(images, get_output_func, blockSize = 100):
    '''Due to memory limitation, we have to call the output function by a block size each time and combine at the end'''
    outputs = []
    for i in range(0, len(images), blockSize):
        print('Get output %d out of %d' % (i, len(images)))
        outputs += [get_output_func([images[i : i + blockSize, :, :]])]
    outputs = np.concatenate(outputs, 0)
    return outputs

def read_images_labels(fns0, fns1):
    n0 = len(fns0)
    n1 = len(fns1)
    print(n0)
    print(n1)
    all_images = np.array([imread(fn) for fn in fns0 + fns1])
    all_labels = np.array([0] * n0 + [1] * n1)
    
    '''Mix up the two proportionally so that the two categories are evenly distributed''' 
    indices0 = range(n0)
    indices1 = -np.ones(n0)
    skip_indices = np.arange(0, n0, n0 / n1).astype('int')
    indices1[skip_indices] = np.array(range(n1)) + n0
    indices = np.reshape(np.array([indices0] + [indices1]).T, (1, -1))
    indices = indices[indices >= 0].astype('int')
    return all_images[indices], all_labels[indices]

runNum = 132
'''Load in the training images as a reference in the distribution plot'''
cleanFns = glob('ForCNN\\CleanBlocks\\Run_%03d\\*.png' % runNum)
muddyFns = glob('ForCNN\\MuddyBlocks\\Run_%03d\\*.png' % runNum)
train_images, train_labels = read_images_labels(cleanFns[:6000], muddyFns)

'''Load in the test images'''
testFns = glob('Output\\Blocks\\Run_%03d\\*.png' % runNum)
test_images = np.array([imread(fn) for fn in testFns])

'''Normalize pixel values to be between 0 and 1'''
train_images, test_images = train_images / 255.0, test_images / 255.0

'''Load in the trained model and prepare the output function'''
model = models.load_model('CNN_Model')
get_output_func = K.function([model.layers[0].input], model.layers[-2].output)

'''Due to memory limitation, we have to call the output function by a block size each time and combine at the end'''
blockSize = 100
train_output = cnn_get_output(train_images, get_output_func)
test_output = cnn_get_output(test_images, get_output_func)

'''Calculate the output values of the second last layer'''
clean_c1_values = train_output[train_labels == 0, 0]
muddy_c1_values = train_output[train_labels == 1, 0]
clean_c2_values = train_output[train_labels == 0, 1]
muddy_c2_values = train_output[train_labels == 1, 1]
test_c1_values = test_output[:, 0]
test_c2_values = test_output[:, 1]

'''Calculate the histogram distributions'''
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
plt.plot(hx_c1_test, hy_c1_test, label = 'Classifier 1 output (test)')
plt.legend(loc = 0)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(hx_c2_clean, hy_c2_clean, label = 'Classifier 2 output (clean)')
plt.plot(hx_c2_muddy, hy_c2_muddy, label = 'Classifier 2 output (muddy)')
plt.plot(hx_c2_test, hy_c2_test, label = 'Classifier 2 output (test)')
plt.grid(True)
plt.legend(loc = 0)

'''Classification'''
threshold = -1.0
labels = test_c1_values < threshold

'''Output classification results to csv'''
f = open('Output\\Run%03d_result.csv' % runNum, 'w')
f.write('Image#,Crib#,Row,Column,Classification\n')
for testFn, label in zip(testFns, labels):
    fn = testFn[testFn.rfind('\\') + 1:testFn.rfind('.')]
    ss = fn.split('_')
    imageNum = int(ss[1][5:])
    cribNum = int(ss[2][4:])
    rowNum = int(ss[3][3:])
    colNum = int(ss[4][3:])
    f.write('%d,%d,%d,%d,%d\n' % (imageNum, cribNum, rowNum, colNum, label))
f.close()
    
