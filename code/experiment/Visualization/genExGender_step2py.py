# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 01:47:15 2017

@author: Kyle
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

#%%
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

#%%
def findTheFirstZero( inputSignal ):
    for i in range( 0, len( inputSignal ) ):
        if inputSignal[ i ] != 0:
            break
    return i 

#def getAccuracy( true, pred ):
#     true = np.argmax( true, 1 )
#     pred = np.argmax( pred, 1 )
#     acc = accuracy_score( true, pred )
#     return acc

def setThreshold( inputM, threshold ):
    outputM = np.copy( inputM )
    lenM = inputM.shape[0]
    colM = inputM.shape[1]
    for i in range( 0, lenM ):
        for j in range( 0, colM ):
            if inputM[i,j] < threshold:
                outputM[i,j] = 1
            else:
                outputM[i,j] = 0
    return outputM
    

def testAccuracy( sess, accuracy, input_x, input_y, advTestFeature, inputTestLabel ):
    batch_size = 100
    data_size= len( inputTestLabel )
    result = []
    for i in range( 0, int( data_size/ batch_size ) ):
        start = ( i * batch_size ) % data_size
        end = min( start + batch_size, data_size )
        tempInput_x = advTestFeature[ start:end, : ]
        tempInput_y = inputTestLabel[ start:end ]
        tempAcc = sess.run( accuracy, feed_dict = { input_x: tempInput_x, input_y: tempInput_y } )
        result.append( tempAcc )
    overallAcc = np.mean( result )
    #print(result)
    return overallAcc

def getAdv( sess, adv, input_x, input_y, inputFeature, inputTestLabel ):
    batch_size = 100
    data_size= len( inputTestLabel )
    data_update = np.copy( inputFeature )
    # mini-batch generation on training data
    for i in range( 0, int( data_size / batch_size ) ):
        start = i * batch_size
        end = ( i + 1 ) * batch_size
        data_update[ start: end, : ] = sess.run( adv, feed_dict = { input_x: inputFeature[ start:end, : ], input_y: inputTestLabel[ start:end, : ] } )
    return data_update
  
def directAdv( modelName, inputTestFeature, inputTestLabel, eps, iteration_num, norm ):
    saver = tf.train.import_meta_graph( modelName + '.meta' )
    with tf.Session() as sess:
      # Restore variables from disk.
      saver.restore( sess, modelName )
      print("Model restored.")
      rGraph = sess.graph
      accuracy = rGraph.get_tensor_by_name( "acc_restore: 0" )
      input_x = rGraph.get_tensor_by_name( "inputx:0" )
      input_y = rGraph.get_tensor_by_name( "inputy:0" )
      adv = rGraph.get_tensor_by_name( "adv" + str(norm) + ":0" )
      testLength = len( inputTestFeature )
      
      resultAdv = []
      resultNoise = []
      for iteration in range( 0, iteration_num ):
      
          tempAdvTestFeature = iter_loadtxt( 'genExGender/' + str(eps) + '_' + str(iteration) + 'adv.csv' )
          tempAdvAccuracy = testAccuracy( sess, accuracy, input_x, input_y, tempAdvTestFeature, inputTestLabel )
          print( str( iteration ) + ' / ' +str( eps ) + ' = ' + str( tempAdvAccuracy ) )
          #np.savetxt(  str( iteration ) + '_' +str( eps ) + '_' + str( tempAdvAccuracy ) + '.csv', tempAdvTestFeature[ 50, : ], delimiter = ',' )
          resultAdv.append( tempAdvAccuracy )
          

          tempNoiseTestFeature = iter_loadtxt( 'genExGender/' + str(eps) + '_' + str(iteration) + 'noise.csv' )
          tempNoiseAccuracy = testAccuracy( sess, accuracy, input_x, input_y, tempNoiseTestFeature, inputTestLabel )
          print( str( iteration ) + ' / ' +str( eps ) + ' = ' + str( tempNoiseAccuracy ) )
          #np.savetxt(  str( iteration ) + '_' +str( eps ) + '_' + str( tempNoiseAccuracy ) + '.csv', tempNoiseTestFeature[ 50, : ], delimiter = ',' )
          resultNoise.append( tempNoiseAccuracy )
          
      return resultAdv, resultNoise

iteration_num = 2
for norm in [ 0 ]:
    folderName = 'gender_rnn/fm12'
    #
    testSet = iter_loadtxt( folderName + '/testSet.csv' )
    testFeature = testSet[ :, 0: 96000 ]
    testFeature = ( testFeature + 1 ) / 2
    testLabel = testSet[ :, 96002 ] ################### Change for other tasks  #############################
    testLength = testSet[ :, 96003 ]
    testLabel = np_utils.to_categorical( testLabel )
    #
    if norm == 0:
        epsList = np.multiply( list( range( 0, 20, 1 ) ), 0.004 )
    elif norm == 1:
        epsList = np.multiply( list( range( 1, 25, 1 ) ), 40 )
    elif norm == 2:
        epsList = np.multiply( list( range( 1, 101, 1 ) ), 0.10 )
    
    overallResultAdv = np.zeros( [ len(epsList), iteration_num ] )
    overallResultNoise = np.zeros( [ len(epsList), iteration_num ] )
    for epsIndex in range( 0, len( epsList ) ):
        resultList1, resultList2 = directAdv( folderName + '/model_100_.ckpt', testFeature, testLabel, epsList[ epsIndex ], iteration_num, norm )
#        plt.plot( list( range( 0, iteration_num ) ), resultList1,  label='Adv' )
#        plt.plot( list( range( 0, iteration_num ) ), resultList2,  label='NoiseNew' )
#        plt.show()
        overallResultAdv[ epsIndex, : ] = resultList1
        overallResultNoise[ epsIndex, : ] = resultList2
        np.savetxt( 'genExGender/' + str(norm) + 'RNNoverallAdv.csv', overallResultAdv, delimiter = ','  )
        np.savetxt( 'genExGender/' + str(norm) + 'RNNoverallNoise.csv', overallResultNoise, delimiter = ','  )
        

