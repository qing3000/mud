# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:40:46 2022

@author: zhangq
"""
import numpy as np
from tensorflow.keras import models, layers, losses
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from glob import glob
from imageio import imread
from random import shuffle
from numpy.fft import fft2, fftshift
import pdb
import multiprocessing as mp
import tqdm

'''Read in the images from two list of files and create the label array'''
def shuffle_filenames_and_labels(fns0, fns1):
    n0 = len(fns0)
    n1 = len(fns1)
    all_fns = np.array(fns0 + fns1)
    all_labels = np.array([0] * n0 + [1] * n1)
    
    '''populate a list of indices for class 1'''
    indices0 = range(n0)
    
    '''Populate a list of indices for class 2'''
    '''First, set all indices to be invalid numbers with the same length as class 1'''
    '''Second, populate the ones by skipping a certain step, which is the ratio n0/n1'''
    indices1 = -np.ones(n0)
    skip_indices = np.linspace(0, n0 - 1, n1).astype('int')
    indices1[skip_indices] = np.array(range(n1)) + n0
    
    '''Combine the two lists of indices'''
    blockIndices = np.array([indices0] + [indices1]).T
    
    '''Reshape it to a flat array''' 
    indices = blockIndices.flat
    
    '''Ignore those invalid indices'''
    indices = indices[indices >= 0].astype('int')
    return all_fns[indices], all_labels[indices]

def read_image(fn, cut = 128):
    '''Read in the image'''
    img = imread(fn)
    
    '''Normalisation'''
    im = (img - np.mean(img))
    sigma = np.std(im)
    if sigma == 0:
        sigma = 1
    im /= sigma
    
    '''2D FFT'''
    NFFT = 1024
    fim = np.abs(fftshift(fft2(im, (NFFT, NFFT))))
    
    '''Extract the ROI'''
    cc = int(NFFT / 2)
    block = fim[cc - cut: cc + cut, cc - cut : cc + cut]    
    # plt.subplot(2, 1, 1)
    # plt.imshow(img, interpolation = 'none', cmap = 'gray')
    # plt.subplot(2,2,3)
    # plt.imshow(fim, interpolation = 'none', )
    # plt.subplot(2,2,4)
    # plt.imshow(block, interpolation = 'none', )
    # raise SystemExit
    return block
  
def main():
    runNum = 132
    print('Get the filenames')
    cleanFns = glob('..\\AMTrakCribs\\Clean\\*.png', recursive = True)[:-500]
    shuffle(cleanFns)
    cleanFns = cleanFns
    muddyFns = glob('..\\AMTrakCribs\\Mud\\\\*.png', recursive = True)[:-500]
    train_fns, train_labels = shuffle_filenames_and_labels(cleanFns, muddyFns)
    print('Load in training images')    
    train_images = [read_image(train_fn) for train_fn in train_fns]
    # mp.freeze_support()
    # pool = mp.Pool(mp.cpu_count())
    # train_images = []
    # for train_image in tqdm.tqdm(pool.imap(read_image, train_fns), total = len(train_fns)):
        # train_images.append(train_image)
    # pool.close()
    
    print('Construct the CNN')
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (256, 256, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten(input_shape = (64, 64, 1)))
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dense(512, activation = 'relu'))    
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    
    print('Compile the CNN')
    model.compile(optimizer = 'adam',
                  loss = losses.SparseCategoricalCrossentropy(),
                  metrics = ['accuracy'])
    
    print('Plot the model as an image')
    #model.summary()
    plot_model(model, 'cnn_model_fft.png', show_shapes = True)
    
    print('Train the CNN')
    #history = model.fit(train_images, train_labels, epochs = 30, validation_data = (test_images, test_labels))
    history = model.fit(np.array(train_images), train_labels, validation_split = 0.3, epochs = 50)
    
    print('Save the model')
    model.save('CNN_Model_FFT')
    
    '''Plot the accuracy curve'''
    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')
    # plt.grid(True)
    
    np.save('TrainingHistory_fft.npy', history.history)
    #ret = model.predict(train_images)

#====================================
if __name__ == '__main__':  
    main()