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
from sys import argv
_, newFolderName, gpuI = argv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuI)

import sys
sys.path.append("../../model/")
import soundNet
sys.path.append("../")
import expUtil
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.python.platform  import tf_logging as logging
from keras import backend as K
import matplotlib.pyplot as plt
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras.models import load_model
from cleverhans.model import CallableModelWrapper
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import time
import shutil

#%% creat folder to save model, the code, and model configuration 
while os.path.isdir( newFolderName ):
    newFolderName = newFolderName + '_1'
    print( 'exist' )

os.mkdir( newFolderName )
shutil.copy( 'emotionSoundNet.py', newFolderName )
shutil.copy( '../../model/soundNet.py', newFolderName )

#%% fix random seed and session
tf.set_random_seed( 7 )
sess = tf.Session(  )
K.set_session( sess )

#%% load data, devide it into training/test set, and seperate out the laebls 
# normalize the feature to [0, 1]
# for emotion tests, filter out value = 4 (other emotions)
# folder list, i.e., IEMCOCAP has 5 sessions, speakers are independent between sessions, always use leave-one-session-out stragegy
folderList = [ 0, 1, 2, 3, 4 ]
testFolder = 4

trainFolderList = folderList.copy( )
del trainFolderList[ testFolder - 1 ]

sampleRate = 16000
precision = 'original'
dataFileFolder = '../../../processedData/waveform/' + str( sampleRate ) + '_' + precision + '/session_'

fold = [ 0, 0, 0, 0, 0 ]
for i in folderList:
    fold[ i ] = eval( 'expUtil.iter_loadtxt( dataFileFolder + str(' + str( i + 1 ) + ') + ".csv" )' )

# seperate training and testing data
trainData = eval( 'np.concatenate( ( fold[ ' + str( trainFolderList[ 0 ] ) + \
                                  ' ], fold[ ' + str( trainFolderList[ 1 ] ) + \
                                  ' ], fold[ ' + str( trainFolderList[ 2 ] ) + \
                                  ' ], fold[ ' + str( trainFolderList[ 3 ] ) + ' ] ), axis=0 )' )
testData = eval( 'fold[ ' + str( testFolder ) + ' ]' )

trainFeature, trainEmotionLabel = expUtil.processData( trainData, task = 'emotion', balance = 'balance' )
testFeature, testEmotionLabel = expUtil.processData( testData, task = 'emotion' ) # note: don't balance the test set

#%% define training parameters
batch_size = 32
learningRate = 0.0001
iterationNum = 100

"""Trains the audio model.

  Args:
     feature: [ sample_size, audio_length ]
     label: one-hot style
"""

def train( testFeature, testLabel, trainFeature, trainLabel, iteration_num = 100, lr_decay = 0.1 ):
    
    result = np.zeros( [ 2, iteration_num ] )
    class_num = testLabel.shape[ 1 ]
    train_datasize = trainFeature.shape[ 0 ]
    
    with tf.Session() as sess:
        
        # changable learning rate 
        global_step = tf.Variable(0)  
        learning_rate = tf.train.exponential_decay( learningRate, global_step, int( iteration_num *(train_datasize/batch_size) ), lr_decay, staircase=False)  

        # fix random index for reproducing result 
        tf.set_random_seed( 17 )
        input_x = tf.placeholder( tf.float32, shape = ( batch_size, 96000 ), name = 'inputx' )
        input_y = tf.placeholder( tf.float32, shape = ( batch_size, class_num ), name = 'inputy' )
        modelT = soundNet.soundNet
        
        # define a set of adversarials (adversarial stuffs)
        orderList = [ np.inf, 1, 2 ]
        advList = orderList.copy( )
        fgsmList = orderList.copy( )
        advOut = orderList.copy( )
        for epsIndex in range( 0, len( orderList ) ):
            fgsmList[ epsIndex ] = FastGradientMethod( modelT , sess = sess )
            fgsm_params = { 'eps': 1, 'y': input_y, 'ord': orderList[ epsIndex ] }
            advList[ epsIndex ] = fgsmList[ epsIndex ].generate( input_x, **fgsm_params )
            advOut[ epsIndex ] = tf.multiply( advList[ epsIndex ], 1, name = 'adv'+str( epsIndex ) )
        
        prediction = modelT( input_x, numClass = class_num )
        loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = prediction, labels= input_y ) )
        train_step = tf.train.AdamOptimizer( learning_rate ).minimize( loss, global_step = global_step )
        correct_prediction = tf.equal( tf.argmax( prediction, 1 ), tf.argmax( input_y, 1 )  )
        accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ), name="acc_restore" )
        
        saver = tf.train.Saver()
        
        # initialize the data 
        init_op = tf.global_variables_initializer(  )
        sess.run( init_op )
        
        # number of iterations
        for iteration in range( 0, iteration_num ):
            # each batch
            for i in range( 0, 1 *int( train_datasize / batch_size ) ):
                
                start = ( i * batch_size ) % train_datasize
                end = min( start + batch_size, train_datasize )
                
                inputTrainFeature = trainFeature[ start: end ]
                inputTrainLabel = trainLabel[ start: end ]
                
                _, lossShow = sess.run( [ train_step, loss ], feed_dict = { input_x: inputTrainFeature, input_y: inputTrainLabel } )
                #print( 'loss = ' + str( lossShow ) )
             
            # get accuracy on a small subset of test data (just several epoch), a very fast approximation of the performance 
            testBatchNum = 3
            testSubsetResult = [ None ] *( batch_size *testBatchNum )
            testSubsetLabel = [ None ] *( batch_size *testBatchNum )
            for testBatch in range( 0, testBatchNum ): # 3*32=96 test samples
                start = testBatch * batch_size 
                end = start + batch_size
                inputTestFeature = testFeature[ start: end, : ]
                inputTestLabel = testLabel[ start: end, : ]     
                tempTestResult, tempAccuracyTest = sess.run( [ prediction, accuracy ], feed_dict = { input_x: inputTestFeature, input_y: inputTestLabel } ) 
                testSubsetLabel[ start :end ] = np.argmax( inputTestLabel, 1 )
                testSubsetResult[ start :end ] = np.argmax( tempTestResult, 1 ) 
            #np.savetxt( newFolderName + '/testResult.csv', testResult, delimiter = ',' )
            #np.savetxt( newFolderName + '/testLabel.csv', inputTestLabel, delimiter = ',' )
            accuracyTest = accuracy_score( testSubsetLabel, testSubsetResult )
            print( confusion_matrix( testSubsetLabel, testSubsetResult ) )
            result[ 0, iteration ] = accuracyTest
            print( 'Epoch:' + str( iteration ) + ' result on test: ' + str( accuracyTest ) )
            
            # get accuracy on a small subset of training data (just one epoch), a very fast approximation of the training loss/ overfitting 
            inputTestTrainFeature = trainFeature[ 0: batch_size, : ]
            inputTestTrainLabel = trainLabel[ 0: batch_size, : ]
            testTrainResult, accuracyTrain = sess.run( [ prediction, accuracy ], feed_dict = { input_x: inputTestTrainFeature, input_y: inputTestTrainLabel } ) 
            print( 'Epoch:' + str( iteration ) + ' result on train: ' + str( accuracyTrain ) )
            #np.savetxt( newFolderName + '/testTrainResult.csv', testTrainResult, delimiter = ',' )
            #np.savetxt( newFolderName + '/testTrainLabel.csv', inputTestTrainLabel, delimiter = ',' )
            result[ 1, iteration ] = accuracyTrain
            print( '-----------------------------' )
            print( sess.run(global_step) ) 
            print( sess.run(learning_rate) )
            # record the accuracy of both test/ training error approximation on the small subset
            np.savetxt( newFolderName + '/accuracy.csv', result, delimiter = ',' )
            
            # save model every 10 epoches
            if ( iteration + 1 )%10 == 0:
                save_path = saver.save( sess, newFolderName + '/model_' + str( iteration + 1 ) + '_.ckpt' )
                print("Model saved in file: %s" % save_path)
            
            resultOnTest = result[ 0, : ]
            resultOnTrain = result[ 1, : ]
            plt.plot( list( range( iteration_num ) ), resultOnTrain )
            plt.plot( list( range( iteration_num ) ), resultOnTest )
            plt.savefig( newFolderName + '/accuracy.png' )
        
        #%% get adversarial samples
#        for epsIndex in range( 0, len( orderList ) ): 
#            data_update = np.copy( feature[ train_datasize: whole_datasize, : ] )
#            # mini-batch generation on training data
#            for i in range( 0, int( ( whole_datasize  - train_datasize ) / batch_size ) ):
#                start = i * batch_size
#                end = ( i + 1 ) * batch_size
#                data_update[ start:end, : ] = sess.run( advList[ epsIndex ], feed_dict = { input_x: feature[ train_datasize + start: train_datasize + end, : ], input_y: label[ train_datasize + start: train_datasize + end, : ] } )      
#            np.savetxt( newFolderName + '/adv'+str( orderList[ epsIndex ] ) + '.csv', data_update, delimiter = ',' )
#        return data_update
#%% start test  
testSamples = testFeature.shape[ 0 ]
trainSamples = testFeature.shape[ 0 ]
log = 'testSample_num = ' + str( testSamples ) + '\n trainSample_num = ' + str( trainSamples )
np.savetxt( newFolderName + '/log.txt', log )
train( testFeature, testEmotionLabel, trainFeature, trainEmotionLabel )

#np.savetxt( newFolderName + '/testSet.csv', dataSet[ train_datasize: whole_datasize, : ], delimiter = ',' )
#np.savetxt( newFolderName + '/testFeature.csv', feature[ train_datasize: train_datasize + batch_size ], delimiter = ',' )
#np.savetxt( newFolderName + '/testLabelGender.csv', genderLabel[ train_datasize: train_datasize + batch_size ], delimiter = ',' )


