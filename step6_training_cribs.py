# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:40:46 2022

@author: zhangq
"""
import numpy as np
from tensorflow.keras import models, layers, losses
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from glob import glob
from imageio import imread
from random import shuffle
from scipy.ndimage import zoom
#from test_2D_FFT import calculate_fft2

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

def read_image(fn):
    '''read in the image'''
    im0 = imread(fn)
    
    '''Statistical normalisation to zero mean and unit variance'''
    im1 = im0 - np.mean(im0)
    sigma = np.std(im1)
    if sigma == 0:
        sigma = 1
    im1 /= sigma

    '''Resize image'''
    fixedWidth = 1250 #Standard gauge 1450mm-200mm
    fixedNarrowWidth = 867 #Narrow gauge 1067 - 200mm
    fixedHeight = 450 #Standard crib size
    M, N = np.shape(im1)
    shortfn = fn[fn.rfind('\\') + 1:]
    if shortfn[3:6] == '132': #narrow gauge
        zim = zoom(im1, (1, fixedNarrowWidth / N))
    else:
        zim = zoom(im1, (1, fixedWidth / N))
    im = fix_image_size(zim, (fixedHeight, fixedWidth))
    return im

def fix_image_size(im, sz):
    blockHeight = sz[0]
    blockWidth = sz[1]    
    block = im[:blockHeight, :blockWidth]
    container = np.zeros((blockHeight, blockWidth))
    M, N = np.shape(block)        
    container[:M, :N] = block
    return container
  
def main():
    runNum = 132
    print('Get the filenames')
    cleanFns = glob('..\\AMTrakCribs\\Clean\\*.png', recursive = True)[:-500]
    # shuffle(cleanFns)
    # cleanFns = cleanFns[:5000]
    muddyFns = glob('..\\AMTrakCribs\\Mud\\*.png', recursive = True)[:-500]
    train_fns, train_labels = shuffle_filenames_and_labels(cleanFns, muddyFns)
    print('Load in training images')
    train_images = [read_image(train_fn) for train_fn in train_fns]
    # mp.freeze_support()
    # pool = mp.Pool(mp.cpu_count())
    # train_images = []
    # for train_image in tqdm.tqdm(pool.imap(read_image, train_fns), total = len(train_fns)):
        # train_images.append(train_image)
    # pool.close()

    M, N = np.shape(train_images[0])
    
    print('Load in the test images')
    #test_fns = glob('Output\\Cribs\\Run_%03d\\*.png' % runNum)
    #test_images = np.array([read_image(fn) for fn in test_fns[10000:16000]])
    
    print('Construct the CNN')
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (M, N, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten(input_shape = (32, 32, 1)))
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
    plot_model(model, 'cnn_model.png', show_shapes = True)
    
    # print('Prepare the batch generator')
    # train_data_generator = ImageDataGenerator()
    # batchSize = 10
    # print(len(train_images))
    # print(len(train_labels))
    # train_ll = to_categorical(train_labels, 2)
    # train_generator = train_data_generator.flow(train_images, train_ll, batch_size = batchSize)
    
    print('Train the CNN')
    history = model.fit(np.array(train_images), train_labels, validation_split = 0.3, epochs = 10)
    #history = model.fit(train_images, train_labels, epochs = 30, validation_data = (test_images, test_labels))
    # history = model.fit_generator(train_generator, steps_per_epoch = len(train_images) // batchSize, epochs = 10)
    
    print('Save the model')
    model.save('CNN_Model_Tiles')
    
    '''Plot the accuracy curve'''
    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')
    # plt.grid(True)
    # plt.show()
    
    np.save('TrainingHistory_cribs.npy', history.history)
    #ret = model.predict(train_images)

#====================================
if __name__ == '__main__':  
    main()