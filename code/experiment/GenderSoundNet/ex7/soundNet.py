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

def soundNet( input, numClass = 2, activationUnit = 'relu', l2_reg = 0.01, init = 'glorot_uniform' ):
    
    # conv1 pool1
    input = tf.convert_to_tensor( input )
    example_num = input.get_shape().as_list()[ 0 ]
    input = tf.reshape( input, [ example_num, 1, 96000, 1 ] )
    
    # conv1 pool1
    with tf.name_scope( 'conv1' ):
        input = keras.layers.convolutional.Conv2D( filters = 16, kernel_size = ( 1, 64 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init )( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )      
    with tf.name_scope( 'pool1' ):
        input = tf.layers.batch_normalization( input )
        input = keras.layers.pooling.MaxPooling2D( ( 1, 8 ), padding='valid' )( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        
    # conv2 pool2
    with tf.name_scope( 'conv2' ):
        input = tf.convert_to_tensor( input )
        input = keras.layers.convolutional.Conv2D( filters = 32, kernel_size = ( 1, 32 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init  )( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
    with tf.name_scope( 'pool2' ):   
        input = tf.layers.batch_normalization( input )
        input = keras.layers.pooling.MaxPooling2D( ( 1, 8 ), padding='valid' )( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        
    # conv3
    with tf.name_scope( 'conv3' ):
        input = keras.layers.convolutional.Conv2D( filters = 64, kernel_size = ( 1, 16 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init  )( input )
        input = tf.layers.batch_normalization( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        
    # conv4
    with tf.name_scope( 'conv4' ):
        input = keras.layers.convolutional.Conv2D( filters = 128, kernel_size = ( 1, 8 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init  )( input )
        input = tf.layers.batch_normalization( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        
    # conv5 pool5
    with tf.name_scope( 'conv5' ):
        input = keras.layers.convolutional.Conv2D( filters = 256, kernel_size = ( 1, 4 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init  )( input )
        input = tf.layers.batch_normalization( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
    with tf.name_scope( 'pool5' ):
        input = keras.layers.pooling.MaxPooling2D( ( 1, 4 ), padding='valid' )( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        
    # conv6
    with tf.name_scope( 'conv6' ):
        input = keras.layers.convolutional.Conv2D( filters = 512, kernel_size = ( 1, 4 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init  )( input )
        input = tf.layers.batch_normalization( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        
    # conv7
    with tf.name_scope( 'conv7' ):    
        input = keras.layers.convolutional.Conv2D( filters = 1024, kernel_size = ( 1, 4 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init  )( input )
        input = tf.layers.batch_normalization( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        
    # conv8
    with tf.name_scope( 'conv8' ):
        input = keras.layers.convolutional.Conv2D( filters = 1024, kernel_size = ( 1, 8 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init  )( input )
        input = tf.layers.batch_normalization( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        
        # dense
    with tf.name_scope( 'flatten' ):
        newSubSequence_length = np.multiply( *input.get_shape().as_list()[ -2: ] )
        input = tf.reshape( input, [ example_num, newSubSequence_length ] )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
    with tf.name_scope( 'dense1' ):
        input = keras.layers.core.Dense( 64, activation = activationUnit )( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
    with tf.name_scope( 'dense2' ):
        output = keras.layers.core.Dense( numClass, activation = 'softmax' )( input )
        print( tf.get_default_graph().get_name_scope() + str( output.shape ) )
    
    return output

#%%    
if __name__ == '__main__':
    time_seq = list( range( 1, 16 ) ) 
    testInput =  np.zeros( [ 15, 96000 ] )
    soundNet( input = testInput )