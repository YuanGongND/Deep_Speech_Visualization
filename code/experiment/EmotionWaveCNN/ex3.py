# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 21:03:04 2017

Conduct erxperiment on IEMOCAP, three labels: 
    
    96001: emotion(0-4)
    96002: speaker(0-9)
    96003: gender(male=0, female=1)
    

@author: Kyle
"""

import os
from sys import argv
_, newFolderName, gpuI = argv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuI)

import eteModel
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.python.platform import tf_logging as logging
from keras import backend as K
import matplotlib.pyplot as plt
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras.models import load_model
from cleverhans.model import CallableModelWrapper
import time
import shutil

#%% creat folder to save model 
if ~os.path.exists( newFolderName ):
    #newFolderName = 'tmp_' + str( time.strftime('%Y_%m_%d_%H_%M',time.localtime(time.time())) )
    os.mkdir( newFolderName )
    shutil.copy( 'ex3.py', newFolderName )
    shutil.copy( 'eteModel.py', newFolderName )

#%% fix random seed and session
tf.set_random_seed( 7 )
sess = tf.Session(  )
K.set_session( sess )

#%%
dataFileName = '../../processedData/datasetNormEmotion.csv'

#%% sub-routines
# fast read data function( very useful in large data )
def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data

#%% read data
dataSet = iter_loadtxt( dataFileName )

#%%
dataSet = dataSet.astype( 'float32' )
np.random.seed(seed= 7 )
np.random.shuffle( dataSet )
np.savetxt( 'dataSetAfterShuffle.csv', dataSet, delimiter = ',' )
print( 'DataSet Output!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' )
print( dataSet[ 17, -17 ] )

feature = dataSet[ :, 0: 96000 ]
feature = ( feature + 1 ) / 2

emotionLabel = dataSet[ :, 96000 ]
speakerLabel = dataSet[ :, 96001 ]
genderLabel = dataSet[ :, 96002 ]

emotionLabel = np_utils.to_categorical( emotionLabel )
speakerLabel = np_utils.to_categorical( speakerLabel )
genderLabel = np_utils.to_categorical( genderLabel )

#%% define training parameters
batch_size = 100
train_datasize = 1800 # need cross-validation in next version
whole_datasize = 2000

"""Trains the audio model.

  Args:
     feature: [ sample_size, audio_length ]
     label: one-hot style
"""

def trainGender( feature, label ):
    
    iteration_num = 101
     # learning_rate = 0.0001
    result = np.zeros( [ 2, iteration_num ] ) 
     # prepare to save the model
    
    with tf.Session() as sess:
        
        # changable learning rate 
        global_step = tf.Variable(0)  
        #conv
        learning_rate = tf.train.exponential_decay( 0.00005, global_step, int( iteration_num *(train_datasize/batch_size) ), 0.32, staircase=False)  
        #rnn
        #learning_rate = tf.train.exponential_decay( 0.0001, global_step, int( iteration_num *(train_datasize/batch_size) ), 0.5, staircase=False)  

        # fix random index for reproducing result 
        tf.set_random_seed( 17 )
        input_x = tf.placeholder( tf.float32, shape = ( batch_size, 96000 ), name = 'inputx' )
        input_y = tf.placeholder( tf.float32, shape = ( batch_size, 2 ), name = 'inputy' )
        modelT = eteModel.eteModelConvNew
        
        # define a set of adversarials 
        epsList = [ np.inf, 1, 2 ]
        advList = epsList.copy()
        fgsmList = epsList.copy()
        advOut = epsList.copy()
        for epsIndex in range( 0, len( epsList ) ):
            fgsmList[ epsIndex ] = FastGradientMethod( modelT , sess = sess )
            fgsm_params = { 'eps': 1, 'y': input_y, 'ord': epsList[ epsIndex ] }
            advList[ epsIndex ] = fgsmList[ epsIndex ].generate( input_x, **fgsm_params )
            advOut[ epsIndex ] = tf.multiply( advList[ epsIndex ], 1, name = 'adv'+str( epsIndex ) )
        
        prediction = modelT( input_x )
        loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = prediction, labels= input_y ) )
        train_step = tf.train.AdamOptimizer( learning_rate ).minimize( loss, global_step = global_step )
        correct_prediction = tf.equal( tf.argmax( prediction, 1 ), tf.argmax( input_y, 1 )  )
        accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ), name="acc_restore" )
        
        saver = tf.train.Saver()
        
        # initialize the data 
        init_op = tf.global_variables_initializer(  )
        #init_op = tf.truncated_normal_initializer(  ) 
        sess.run( init_op )
        
        # number of iterations
        for iteration in range( 0, iteration_num ):
            # each batch
            for i in range( 0, 1 *int( train_datasize / batch_size ) ):
                
                start = ( i * batch_size ) % train_datasize
                end = min( start + batch_size, train_datasize )
                
                inputTrainFeature = feature[ start: end ]
                inputTrainLabel = label[ start: end ]
                
                _, lossShow = sess.run( [ train_step, loss ], feed_dict = { input_x: inputTrainFeature, input_y: inputTrainLabel } )
                print( 'loss = ' + str( lossShow ) )
             
            # get overall accuracy on the test data 
            inputTestFeature = feature[ train_datasize: train_datasize + batch_size ]
            inputTestLabel = label[ train_datasize: train_datasize + batch_size ]
            
            testResult, accuracyTest = sess.run( [ prediction, accuracy ], feed_dict = { input_x: inputTestFeature, input_y: inputTestLabel } ) 
            np.savetxt( newFolderName + '/testResult.csv', testResult, delimiter = ',' )
            np.savetxt( newFolderName + '/testLabel.csv', inputTestLabel, delimiter = ',' )
            result[ 0, iteration ] = accuracyTest
            print( 'Epoch:' + str( iteration ) + ' result on test: ' + str( accuracyTest ) )
            
            # get result on the training set
            inputTestTrainFeature = feature[ 0: batch_size, : ]
            inputTestTrainLabel = label[ 0: batch_size, : ]
            testTrainResult, accuracyTrain = sess.run( [ prediction, accuracy ], feed_dict = { input_x: inputTestTrainFeature, input_y: inputTestTrainLabel } ) 
            print( 'Epoch:' + str( iteration ) + ' result on train: ' + str( accuracyTrain ) )
            np.savetxt( newFolderName + '/testTrainResult.csv', testTrainResult, delimiter = ',' )
            np.savetxt( newFolderName + '/testTrainLabel.csv', inputTestTrainLabel, delimiter = ',' )
            result[ 1, iteration ] = accuracyTrain
            print( '-----------------------------' )
            print( sess.run(global_step) ) 
            print( sess.run(learning_rate) ) 
            np.savetxt( newFolderName + '/accuracy.csv', result, delimiter = ',' )
            
            # save model every 10 epoches
            if iteration%10 == 0:
                save_path = saver.save( sess, newFolderName + '/model_' + str( iteration ) + '_.ckpt' )
                print("Model saved in file: %s" % save_path)
            
        resultOnTest = result[ 0, : ]
        resultOnTrain = result[ 1, : ]
        plt.plot( list( range( iteration_num ) ), resultOnTrain )
        plt.plot( list( range( iteration_num ) ), resultOnTest )
        plt.savefig( newFolderName + '/accuracy.png' )
        
        #%% attach
        for epsIndex in range( 0, len( epsList ) ):
            
            data_update = np.copy( feature[ train_datasize: whole_datasize, : ] )
            # mini-batch generation on training data
            for i in range( 0, int( ( whole_datasize  - train_datasize ) / batch_size ) ):
                start = i * batch_size
                end = ( i + 1 ) * batch_size
                data_update[ start:end, : ] = sess.run( advList[ epsIndex ], feed_dict = { input_x: feature[ train_datasize + start: train_datasize + end, : ], input_y: label[ train_datasize + start: train_datasize + end, : ] } )
                
            np.savetxt( newFolderName + '/adv'+str( epsList[ epsIndex ] ) + '.csv', data_update, delimiter = ',' )
        
        return data_update
 #%% 
trainGender( feature, emotionLabel )
np.savetxt( newFolderName + '/testSet.csv', dataSet[ train_datasize: whole_datasize, : ], delimiter = ',' )
np.savetxt( newFolderName + '/testFeature.csv', feature[ train_datasize: train_datasize + batch_size ], delimiter = ',' )
np.savetxt( newFolderName + '/testLabelGender.csv', genderLabel[ train_datasize: train_datasize + batch_size ], delimiter = ',' )


