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

def soundNet( input, numClass = 2, activationUnit = 'relu', l2_reg = 0.01, init = 'lecun_uniform', biasInit = 'Zeros', denseUnitNum = 64, denseUnit = 'relu' ):
    
    # conv1 pool1
    input = tf.convert_to_tensor( input )
    example_num = input.get_shape().as_list()[ 0 ]
    input = tf.reshape( input, [ example_num, 1, 96000, 1 ] )
    
    # conv1 pool1
    with tf.name_scope( 'conv1' ):
#        input = tf.layers.conv2d( inputs = input, filters = 16, kernel_size = ( 1, 64 ), strides=( 1, 2 ), padding='same' )
#        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        input = keras.layers.convolutional.Conv2D( filters = 16, kernel_size = ( 1, 64 ), strides= ( 1, 2 ), padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init, bias_initializer = biasInit  )( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        conv1Out = tf.identity( input, name = 'conv1Out' ) 
#    input = conv2d( input, 1, 16, k_w=64, d_w=2, p_h=32, name_scope='conv1')
#    print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        
    with tf.name_scope( 'pool1' ):
        input = tf.layers.batch_normalization( input )
        input = keras.layers.pooling.MaxPooling2D( ( 1, 8 ), padding='valid' )( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        pool1Out = tf.identity( input, name = 'pool1Out' ) 
        
    # conv2 pool2
    with tf.name_scope( 'conv2' ):
        input = tf.convert_to_tensor( input )
        input = keras.layers.convolutional.Conv2D( filters = 32, kernel_size = ( 1, 32 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init  )( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        conv2Out = tf.identity( input, name = 'conv2Out' ) 
    with tf.name_scope( 'pool2' ):   
        input = tf.layers.batch_normalization( input )
        input = keras.layers.pooling.MaxPooling2D( ( 1, 8 ), padding='valid' )( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        pool2Out = tf.identity( input, name = 'conv2Out' ) 
        
    # conv3
    with tf.name_scope( 'conv3' ):
        input = keras.layers.convolutional.Conv2D( filters = 64, kernel_size = ( 1, 16 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init  )( input )
        input = tf.layers.batch_normalization( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        conv3Out = tf.identity( input, name = 'conv3Out' ) 
        
    # conv4
    with tf.name_scope( 'conv4' ):
        input = keras.layers.convolutional.Conv2D( filters = 128, kernel_size = ( 1, 8 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init  )( input )
        input = tf.layers.batch_normalization( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        conv4Out = tf.identity( input, name = 'conv4Out' ) 
        
    # conv5 pool5
    with tf.name_scope( 'conv5' ):
        input = keras.layers.convolutional.Conv2D( filters = 256, kernel_size = ( 1, 4 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init  )( input )
        input = tf.layers.batch_normalization( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        conv4Out = tf.identity( input, name = 'conv4Out' ) 
    with tf.name_scope( 'pool5' ):
        input = keras.layers.pooling.MaxPooling2D( ( 1, 4 ), padding='valid' )( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        pool5Out = tf.identity( input, name = 'pool5Out' ) 
        
    # conv6
    with tf.name_scope( 'conv6' ):
        input = keras.layers.convolutional.Conv2D( filters = 512, kernel_size = ( 1, 4 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init  )( input )
        input = tf.layers.batch_normalization( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        conv6Out = tf.identity( input, name = 'conv6Out' ) 
        
    # conv7
    with tf.name_scope( 'conv7' ):    
        input = keras.layers.convolutional.Conv2D( filters = 1024, kernel_size = ( 1, 4 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init  )( input )
        input = tf.layers.batch_normalization( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        conv7Out = tf.identity( input, name = 'conv7Out' ) 
        
    # conv8
    with tf.name_scope( 'conv8' ):
        input = keras.layers.convolutional.Conv2D( filters = 1024, kernel_size = ( 1, 8 ), strides=2, padding='same', activation= activationUnit, kernel_regularizer=regularizers.l2( l2_reg ), kernel_initializer = init  )( input )
        input = tf.layers.batch_normalization( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        conv8Out = tf.identity( input, name = 'conv8Out' ) 
        
    # flatten
    with tf.name_scope( 'flatten' ):
        newSubSequence_length = np.multiply( *input.get_shape().as_list()[ -2: ] )
        input = tf.reshape( input, [ example_num, newSubSequence_length ] )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
    
    # dense1
    with tf.name_scope( 'dense1' ):
        input = keras.layers.core.Dense( denseUnitNum, activation = denseUnit, kernel_initializer = init, bias_initializer = biasInit )( input )
        print( tf.get_default_graph().get_name_scope() + str( input.shape ) )
        dense1Out = tf.identity( input, name = 'dense1Out' ) 
    
    # dense2
    with tf.name_scope( 'dense2' ):
        output = keras.layers.core.Dense( numClass, activation = 'softmax' )( input )
        print( tf.get_default_graph().get_name_scope() + str( output.shape ) )
        dense2Out = tf.identity( input, name = 'dense2Out' ) 
   
    return output

#%%    
if __name__ == '__main__':
    time_seq = list( range( 1, 16 ) ) 
    testInput =  np.zeros( [ 15, 96000 ] )
    soundNet( input = testInput )