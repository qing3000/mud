# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 09:16:14 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt

history_cribs = np.load('TrainingHistory_cribs.npy', allow_pickle = True).item()
history_tiles = np.load('TrainingHistory_tiles.npy', allow_pickle = True).item()
history_fft = np.load('TrainingHistory_fft.npy', allow_pickle = True).item()

plt.plot(history_cribs['accuracy'], 'r', label = 'Accuracy (crib image)') 
plt.plot(history_cribs['val_accuracy'], 'r--', label = 'Validation accuracy (crib image)')
plt.plot(history_tiles['accuracy'], 'g', label = 'Accuracy (tile image)') 
plt.plot(history_tiles['val_accuracy'], 'g--', label = 'Validation accuracy (tile image)')
plt.plot(history_fft['accuracy'], 'b', label = 'Accuracy (fft image)') 
plt.plot(history_fft['val_accuracy'], 'b--', label = 'Validation accuracy (fft image)')
plt.xlabel('Epochs')
plt.legend(loc = 0) 
plt.grid(True)


plt.plot(history_cribs['loss'], 'r', label = 'Loss (crib image)') 
plt.plot(history_cribs['val_loss'], 'r--', label = 'Validation loss (crib image)')
plt.plot(history_tiles['loss'], 'g', label = 'Loss (tile image)') 
plt.plot(history_tiles['val_loss'], 'g--', label = 'Validation loss (tile image)')
plt.plot(history_fft['loss'], 'b', label = 'Loss (fft image)') 
plt.plot(history_fft['val_loss'], 'b--', label = 'Validation loss (fft image)')
plt.xlabel('Epochs')
plt.legend(loc = 0) 
plt.grid(True)