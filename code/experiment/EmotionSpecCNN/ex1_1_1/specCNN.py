# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:46:44 2017

@author: Kyle
"""
import tensorflow as tf
import numpy as np
import keras 
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import Conv2D, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense
import sys
sys.path.append("../experiment")
#import expUtil
import math

#%% test convolutional network
def specCNN( input, inputHeight = 256, inputWidth = 256, numClass = 2, \
            convSize = 3, convStride = 1, convUnit = 'relu', l2_reg = 0.01, convLayerNum = 5, \
            convFilterNum = 32, init = 'lecun_uniform', biasInit = 'Zeros', \
            dropoutRate = 0.5, poolSize = 2, \
            denseUnit = 'relu', denseLayerNum = 1, denseUnitNum = 64 ):
    
    # prepare the tensor    
    
#    showSample = input[ 5, : ]
#    showSample.resize( [ 256, 256 ] )
#    plt.imshow( showSample , cmap='hot', interpolation='nearest')
#    plt.show()
    input = tf.convert_to_tensor( input )
    sampleNum = input.get_shape().as_list()[ 0 ]
    input = tf.reshape( input, [ sampleNum, inputWidth, inputHeight, 1 ] )
    print( 'After preprocess : ' + str( input.shape ) )
    
    # conv layer
    for layers in range( 0, convLayerNum ):
        with tf.name_scope( 'conv' + str( layers ) ): 
            input = Conv2D( filters = convFilterNum, kernel_size = [ convSize, convSize ], strides = convStride, \
                                    padding = 'same', activation= convUnit, kernel_regularizer=regularizers.l2( l2_reg ), \
                                    kernel_initializer = init, bias_initializer = biasInit )( input )
        
        input = tf.layers.batch_normalization( input )
        input = tf.nn.dropout( input, keep_prob = dropoutRate )
        input = MaxPooling2D( pool_size=( poolSize, poolSize ), padding='valid' )( input )
        print( 'Conv_' + str( layers ) +' : ' + str( input.shape ) )
    
    newShape = input.get_shape().as_list()
    newDim = newShape[ 1 ] *newShape[ 2 ] *newShape[ 3 ]
    input = tf.reshape( input, [ sampleNum, newDim ] )
    print( 'Flatten : ' + str( input.shape ) )
    
    # dense layer
    for layers in range( 0, denseLayerNum ):
        with tf.name_scope( 'dense' + str( layers ) ): 
            input = Dense( units = denseUnitNum, activation = denseUnit, kernel_initializer = init, bias_initializer = biasInit )( input )
        input = tf.layers.batch_normalization( input )
        input = tf.nn.dropout( input, keep_prob = dropoutRate )
        print( 'Dense_' + str( layers ) +' : ' + str( input.shape ) )
        
    # output layer
    output = Dense( numClass, activation = 'softmax' )( input )
    print( 'Output : ' + str( output.shape ) )
    
    return output

#%%    
if __name__ == '__main__':
    toyData = expUtil.iter_loadtxt( '../../processedData/toySpectrogram/16000_original/session_1.csv' )[ 0:55, 0:65536 ]
    #toyData = ( toyData - np.mean( toyData ) ) /math.sqrt( np.var( toyData ) )
    testInput =  toyData[ 0:15, 0: 65536 ]
    specCNN( input = testInput )
    
# sns.distplot(toyData, hist=False, rug=True);
#import seaborn as sns