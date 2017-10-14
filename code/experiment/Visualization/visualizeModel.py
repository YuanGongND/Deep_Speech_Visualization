# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 21:03:04 2017

Conduct erxperiment on IEMOCAP, three labels: 
    
    96001: emotion(0-4, 5 = other emotions)
    96002: speaker(0-9)
    96003: gender(male=0, female=1)
    

@author: Kyle
"""

import os
import seaborn as sns
from sys import argv

newFolderName = 'test'
gpuI = 0

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuI)

import sys
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.python.platform  import tf_logging as logging
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import time
import shutil

def visualizeFilter( layerName = 'conv1', modelIndexList = list( range( 10, 110, 10 ) ), folderName = '' ):
    
    figureIndex = 1    
    for modelIndex in modelIndexList:
        modelName = folderName + 'model_' + str( modelIndex ) + '_.ckpt'
        saver = tf.train.import_meta_graph( modelName + '.meta' )
        with tf.Session() as sess:
          saver.restore( sess, modelName )
          print("Model restored.")
          #rGraph = sess.graph
          allFilter =  tf.get_collection( tf.GraphKeys.VARIABLES, scope= layerName )
          kernal = allFilter[ 0 ].eval( )
          filterNum = kernal.shape[ 3 ]
          filterNum = 16
          for filterIndex in range( 0, filterNum ):
              tempFilter = kernal[ 0, :, 0, filterIndex ]
              filterFFT = np.fft.fft( tempFilter )
              print( tempFilter )
              # plot filter
              plt.subplot( len( modelIndexList ), filterNum *2, figureIndex )
              plt.plot( list( range( len( tempFilter ) ) ), tempFilter, linewidth = 0.5 )
              plt.xticks( [ ] )
              plt.yticks( [ ] )
              # plot fft
              figureIndex += 1
              plt.subplot( len( modelIndexList ),  filterNum *2, figureIndex )
              plt.plot( list( range( len( filterFFT ) ) ), filterFFT, 'r', linewidth = 0.5 )
              #plt.xticks( [ ] )
              #plt.yticks( [ ] )
              figureIndex += 1
    fig = plt.gcf()
    fig.set_size_inches( 250, 10 )
    fig.savefig( filename = folderName + layerName +'Visualization.png', dpi = 100 )


#%% calculate the number of elements of an high-dimensional tensor
def countElements( inputM ):
    inputShape = inputM.shape
    dim = 1
    for i in inputShape:
        dim *= i
    return dim

#%%
def plotInputDistribution( inputM ):
    output = np.reshape( inputM, [ countElements( inputM ) ] )
    fig1 = plt.figure(  )
    ax1 = fig1.gca()
    ax1.hist( output )
    
#%% 
def getTensorByLayer( sess, layerName ):
    pass
    
#%%
if __name__ == '__main__':
#    #%% get the name of test 
    folderName = '../GenderSoundNet/ex15/0.0001_32_RandomUniform/models/'
#    visualizeFilter( layerName = 'conv1', modelIndexList = list( range( 1, 45, 5 ) ), folderName = folderName )
#    plt.show(  )

    layerName = 'conv1'
    modelIndexList = [ 100 ]
    figureIndex = 1    
    for modelIndex in modelIndexList:
        modelName = folderName + str( modelIndex ) + '_.ckpt'
        saver = tf.train.import_meta_graph( modelName + '.meta' )
        with tf.Session() as sess:
          saver.restore( sess, modelName )
          print("Model restored.")
          #rGraph = sess.graph
          allFilter =  tf.get_collection( tf.GraphKeys.VARIABLES, scope= layerName )
          print( allFilter )
#          kernal = allFilter[ 3 ].eval( )
#          filterNum = kernal.shape[ 3 ]
#          filterNum = 16
#          for filterIndex in range( 0, filterNum ):
#              tempFilter = kernal[ 0, :, 0, filterIndex ]
#              filterFFT = np.fft.fft( tempFilter )
#              print( tempFilter )
#              # plot filter
#              plt.subplot( len( modelIndexList ), filterNum *2, figureIndex )
#              plt.plot( list( range( len( tempFilter ) ) ), tempFilter, linewidth = 0.5 )
#              plt.xticks( [ ] )
#              plt.yticks( [ ] )
#              # plot fft
#              figureIndex += 1
#              plt.subplot( len( modelIndexList ),  filterNum *2, figureIndex )
#              plt.plot( list( range( len( filterFFT ) ) ), filterFFT, 'r', linewidth = 0.5 )
#              #plt.xticks( [ ] )
#              #plt.yticks( [ ] )
#              figureIndex += 1
#    fig = plt.gcf()
#    fig.set_size_inches( 250, 10 )
#    fig.savefig( filename = folderName + layerName +'Visualization.png', dpi = 100 )