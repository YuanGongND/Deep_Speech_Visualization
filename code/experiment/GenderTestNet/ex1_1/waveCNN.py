# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:25:16 2017

Keras model of 

@author: Kyle
"""

import tensorflow as tf
import numpy as np
import keras 
from keras.models import Model
from keras import regularizers

#%%
def waveCNN( input, timeStep_num = 150, convLayer_num_front = 8, filter_num = 32, numClass = 2, \
            activationUnit = 'relu', conv_filter_size_front = 40, pooling_size = 2, convLayer_num_back = 6, conv_filter_size_back = 40 ):
    # the input shape is [ example_num, whole_audio_length ], e.g., [ 200 samples, 96000 points ]
    
    # convert it to tensor
    input = tf.convert_to_tensor( input )
    
    # parameters of the network
    example_num = input.get_shape().as_list()[ 0 ]
    
    # length of each sub-sequence, e.g., 96000/timeStep(150)
    subSequence_length = int( input.get_shape().as_list()[ 1 ] / timeStep_num )

    # reshape into [ example_num * sequence, subsequence_length ]
    input = tf.reshape( input, [ example_num *timeStep_num, 1, subSequence_length, 1 ] )
    print( input.shape )
    
    # first conduct average pooling
    input = keras.layers.pooling.AveragePooling2D( pool_size=( 1, pooling_size ), strides=None, padding='same' )( input )
    
    print( input.shape )
    
    # convLayer_num *( conv + maxpooling )
    for i in range( convLayer_num_front ):
        input = keras.layers.convolutional.Conv2D( filter_num, ( 1, conv_filter_size_front ), padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( 0.01) )( input )
        print( input.shape )
        input = keras.layers.pooling.MaxPooling2D( ( 1, pooling_size ), padding='same' )( input )
        print( input.shape )
        print( i )
    
    # reshape for preparision of LSTM layers 
    print( input.shape )
    input = tf.transpose( input, [ 3, 0, 1, 2 ] ) # change the column order
    print( input.shape )
    restPoint = input.get_shape().as_list()[ -1 ]
    print( input.shape )
    input = tf.reshape( input, [ filter_num, 1, example_num, timeStep_num*restPoint ] )
    print( input.shape )
    input = tf.transpose( input, [ 2,1,3,0 ] )
    print( input.shape )
    
    for i in range( convLayer_num_back ):
        input = keras.layers.convolutional.Conv2D( filter_num, ( 1, conv_filter_size_back ), padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2(0.01) )( input )
        print( input.shape )
        input = keras.layers.pooling.MaxPooling2D( ( 1, pooling_size ), padding='same' )( input )
        print( input.shape )
        print( i )
        
    newSubSequence_length = np.multiply( *input.get_shape().as_list()[ -2: ] )
    input = tf.reshape( input, [ example_num, newSubSequence_length ] )
    print( input.shape )
    
    # start the LSTM layers 
    input = keras.layers.core.Dense( 64, activation = activationUnit )( input )
    print( input.shape )
    output = keras.layers.core.Dense( numClass, activation = 'softmax' )( input )
    print( output.shape )
    
    return output

#%%
def waveCNNBN( input, timeStep_num = 150, convLayer_num_front = 8, filter_num = 32, numClass = 4, init = 'glorot_uniform',\
            activationUnit = 'relu', conv_filter_size_front = 40, pooling_size = 2, convLayer_num_back = 6, conv_filter_size_back = 40, l2_reg = 0.01,\
            denseUnitNum = 64 ):
    # the input shape is [ example_num, whole_audio_length ], e.g., [ 200 samples, 96000 points ]
    
    # convert it to tensor
    input = tf.convert_to_tensor( input )
    
    # parameters of the network
    example_num = input.get_shape().as_list()[ 0 ]
    
    # length of each sub-sequence, e.g., 96000/timeStep(150)
    subSequence_length = int( input.get_shape().as_list()[ 1 ] / timeStep_num )

    # reshape into [ example_num * sequence, subsequence_length ]
    input = tf.reshape( input, [ example_num *timeStep_num, 1, subSequence_length, 1 ] )
    print( input.shape )
    
    # first conduct average pooling
    #input = tf.layers.batch_normalization( input )
    input = keras.layers.pooling.AveragePooling2D( pool_size=( 1, pooling_size ), strides=None, padding='same' )( input )
    print( input.shape )
    
    # convLayer_num *( conv + maxpooling )
    for i in range( convLayer_num_front ):
        input = tf.layers.batch_normalization( input )
        with tf.name_scope( 'conv' + str( i + 1 ) ):
            input = keras.layers.convolutional.Conv2D( filter_num, ( 1, conv_filter_size_front ), padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init )( input )
        print( input.shape )
        input = keras.layers.pooling.MaxPooling2D( ( 1, pooling_size ), padding='same' )( input )
        print( input.shape )
        print( i )
    
    # reshape for preparision of LSTM layers 
    print( input.shape )
    input = tf.transpose( input, [ 3, 0, 1, 2 ] ) # change the column order
    print( input.shape )
    restPoint = input.get_shape().as_list()[ -1 ]
    print( input.shape )
    input = tf.reshape( input, [ filter_num, 1, example_num, timeStep_num*restPoint ] )
    print( input.shape )
    input = tf.transpose( input, [ 2,1,3,0 ] )
    print( input.shape )
    
    for i in range( convLayer_num_back ):
        input = tf.layers.batch_normalization( input )
        input = keras.layers.convolutional.Conv2D( filter_num, ( 1, conv_filter_size_back ), padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init )( input )
        print( input.shape )
        input = keras.layers.pooling.MaxPooling2D( ( 1, pooling_size ), padding='same' )( input )
        print( input.shape )
        print( i )
        
    newSubSequence_length = np.multiply( *input.get_shape().as_list()[ -2: ] )
    input = tf.reshape( input, [ example_num, newSubSequence_length ] )
    print( input.shape )
    
    # start the LSTM layers 
    input = keras.layers.core.Dense( 64, activation = activationUnit, kernel_initializer = init )( input )
    print( input.shape )
    output = keras.layers.core.Dense( numClass, activation = 'softmax' )( input )
    print( output.shape )
    
    return output

#%%    
if __name__ == '__main__':
    time_seq = list( range( 1, 16 ) ) 
    testInput =  np.zeros( [ 64, 96000 ] )
    waveCNNBN( input = testInput )