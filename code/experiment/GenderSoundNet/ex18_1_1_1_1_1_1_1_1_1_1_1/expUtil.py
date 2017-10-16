# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 00:40:02 2017

@author: Kyle
"""

import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from imblearn.under_sampling import NearMiss, AllKNN, RandomUnderSampler
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
if __name__ == '__main__':
    sys.path.append("../model/")
    import soundNet
    import waveCNN
else:
    sys.path.append("../../model/")
    import soundNet
    import waveCNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import math
import seaborn as sns

#%% slice the matrix using discontinuous row index 
def discontSliceRow( matrix, index ):
    outputMatrix = np.zeros( [ len( index ), matrix.shape[ 1 ] ] )
    outputIndex = 0
    for processLine in range( 0, len( matrix ) ):
        if processLine in index:
            outputMatrix[ outputIndex, : ] = matrix[ processLine, : ]
            outputIndex += 1
    return outputMatrix

#%% slice the matrix using discontinuous column index
def discontSliceCol( matrix, index ):
    outputMatrix = np.zeros( [ matrix.shape[ 0 ], len( index ) ] )
    outputIndex = 0
    for processCol in range( 0, matrix.shape[1] ):
        if processCol in index:
            outputMatrix[ :, outputIndex ] = matrix[ :, processCol ]
            outputIndex += 1
    return outputMatrix

#%%
def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype= float):
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
def processData( dataSet, task = 'nonEmotion', balance = 'imbalance', dataType = 'waveform' ):
    
    if dataType == 'waveform' or dataType == 'toyWaveform':
        dataSize = 96000
    else:
        dataSize = 256 *256
    
    # for speaker and gender task, use all database
    if task != 'emotion':
        dataSet = dataSet.astype( 'float32' )
        np.random.seed(seed= 7 )
        np.random.shuffle( dataSet )
        #np.savetxt( 'dataSetAfterShuffle.csv', dataSet, delimiter = ',' )
        print( dataSet[ 17, -17 ] )
        
        feature = dataSet[ :, 0: dataSize ]
        # normalize the data
        feature = ( feature - np.mean( feature ) ) /math.sqrt( np.var( feature ) )
         
        emotionLabel = dataSet[ :, dataSize + 0 ]
        speakerLabel = dataSet[ :, dataSize + 1 ]
        genderLabel = dataSet[ :, dataSize + 2 ]
        
        emotionLabel = np_utils.to_categorical( emotionLabel )
        speakerLabel = np_utils.to_categorical( speakerLabel )
        genderLabel = np_utils.to_categorical( genderLabel )
        
        if task == 'speaker':
            return feature, speakerLabel
        elif task == 'gender':
            return feature, genderLabel
    
    # for emotion task, only select 4 classes ( 0, 1, 2, 3 ), label 4 means other emotion, should abandon
    if task == 'emotion':
        dataSet = dataSet.astype( 'float32' )
        np.random.seed(seed= 7 )
        np.random.shuffle( dataSet )
        #np.savetxt( 'dataSetAfterShuffle.csv', dataSet, delimiter = ',' )
        print( dataSet[ 17, -17 ] )
        
        # select only label with 0,1,2,3
        emotionLabel = dataSet[ :, dataSize ]
        emotionIndices = [ i for i, x in enumerate( emotionLabel ) if x != 4]
        dataSet = discontSliceRow( dataSet, emotionIndices )
        
        feature = dataSet[ :, 0: dataSize ]
        # normalize the data
        feature = ( feature - np.mean( feature ) ) /math.sqrt( np.var( feature ) )
        
        emotionLabel = dataSet[ :, dataSize ]
        emotionLabel = np_utils.to_categorical( emotionLabel )
        
        # random oversampling
        if balance == 'balance':
            ros = RandomOverSampler( random_state= 7 )
            feature, emotionLabel = ros.fit_sample( feature, np.argmax( emotionLabel, 1 ) )
            numSamples = len( emotionLabel )
            emotionLabel = np.array( emotionLabel )
            emotionLabel.resize( [ numSamples, 1 ] )
            dataSet = np.concatenate( ( feature, emotionLabel ), axis = 1 ) 
            np.random.shuffle( dataSet )
            feature = dataSet[ :, 0: dataSize ]
            emotionLabel = dataSet[ :, dataSize ]
            emotionLabel = np_utils.to_categorical( emotionLabel )
        
        assert emotionLabel.shape[ 1 ] == 4
        
        return feature, emotionLabel

#%% 
def train( testFeature, testLabel, trainFeature, trainLabel, newFolderName, iteration_num = 100, \
           lr_decay = 0.1, batch_size = 32, learningRate = 0.0001, iterationNum = 100, \
           modelT = soundNet.soundNet, init = 'lecun_uniform', saveSign = False, denseUnitNum = 64,\
           dataType = 'waveform' ):
    
    if dataType == 'waveform' or dataType == 'toyWaveform':
        dataSize = 96000
    else:
        dataSize = 256 *256
    
    os.mkdir( newFolderName + '/weight' )
    os.mkdir( newFolderName + '/models' )
    result = np.zeros( [ 2, iteration_num ] )
    class_num = testLabel.shape[ 1 ]
    train_datasize = trainFeature.shape[ 0 ]
    
    tf.set_random_seed( 7 )
    with tf.Session() as sess:
        
        # changable learning rate 
        global_step = tf.Variable(0)  
        learning_rate = tf.train.exponential_decay( learningRate, global_step, int( iteration_num *(train_datasize/batch_size) ), lr_decay, staircase=False)  

        # fix random index for reproducing result 
        tf.set_random_seed( 17 )
        input_x = tf.placeholder( tf.float32, shape = ( batch_size, dataSize ), name = 'inputx' )
        input_y = tf.placeholder( tf.float32, shape = ( batch_size, class_num ), name = 'inputy' )
        
        prediction = modelT( input_x, numClass = class_num, l2_reg = 0.5, init = init, denseUnitNum  = denseUnitNum )
        loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = prediction, labels= input_y ) )
        train_step = tf.train.AdamOptimizer( learning_rate ).minimize( loss, global_step = global_step )
        #train_step = tf.train.GradientDescentOptimizer( learning_rate ).minimize( loss, global_step = global_step )
        correct_prediction = tf.equal( tf.argmax( prediction, 1 ), tf.argmax( input_y, 1 )  )
        accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ), name="acc_restore" )
        
        # initialize the data 
        init_op = tf.global_variables_initializer(  )
        sess.run( init_op )
        saver = tf.train.Saver( max_to_keep= 100 )
        print( tf.trainable_variables() )
        
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
            np.savetxt( newFolderName + '/testTrainResult.csv', testTrainResult, delimiter = ',' )
            np.savetxt( newFolderName + '/testTrainLabel.csv', inputTestTrainLabel, delimiter = ',' )
            result[ 1, iteration ] = accuracyTrain
            print( '-----------------------------' )
            #print( sess.run(global_step) ) 
            #print( sess.run(learning_rate) )
            # record the accuracy of both test/ training error approximation on the small subset
            np.savetxt( newFolderName + '/accuracy.csv', result, delimiter = ',' )
            
            # print variable
            if iteration == 0:
                lastState = printVariable( sess, newFolderName = newFolderName )
            else:
                lastState = printVariable( sess, lastState, iteration + 1, newFolderName = newFolderName )
            #np.savetxt( newFolderName + '/weightConv1' + str( iteration + 1 ) + '.csv', lastState, delimiter = ',' )
            
            # save model every 10 epoches
            #if ( iteration + 1 )%10 == 0 and saveSign == True:
            if ( iteration + 1 )<10 == 0 and saveSign == True:
                save_path = saver.save( sess, newFolderName + '/models/' + str( iteration + 1 ) + '_.ckpt' )
                print("Model saved in file: %s" % save_path)
            
            resultOnTest = result[ 0, : ]
            resultOnTrain = result[ 1, : ]
            plt.plot( list( range( iteration_num ) ), resultOnTrain )
            plt.plot( list( range( iteration_num ) ), resultOnTest )
            plt.savefig( newFolderName + '/accuracy.png' )
            
    return resultOnTrain, resultOnTest

#%%
def printVariable( sess, lastState = -1, iteration = 1, newFolderName = -1 ):     
      #layerList = [ 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'dense1', 'dense2' ]
      layerList = [  'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8' ]
      currentState = [ 0 ] *len( layerList )
      for layerIndex in range( len( layerList ) ):
          
          # get all parameters of this layer
          allFilter =  tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope= layerList[ layerIndex ] )
          #print( allFilter )
          
          # for conv layers
          if layerIndex <= 7:
              
              # get the weight of this layer
              kernal = allFilter[ 0 ].eval( )
              if layerIndex == 0:
                  plotConvFilters( kernal, newFolderName + '/models/' +str( iteration ) + '.png' )
              
              # save the weight of current layer
              currentState[ layerIndex ] = kernal
              
              # track the difference with last state
              if lastState != -1:
                  diff = trackChange( currentState[ layerIndex ], lastState[ layerIndex ] )
                  print( layerList[ layerIndex ] + ' : ' + str( diff )  )
          
      return currentState

#%% track change ( in percentile ) of different layers according to epochs
def trackChange( currentState, lastState ):
    difference = lastState - currentState 
    diffInPercentile = difference / currentState
    absDiffInPercentile = np.abs( diffInPercentile )
    meanChange = np.mean( absDiffInPercentile )
    return meanChange *100
#%% load data, devide it into training/test set, and seperate out the laebls 
# normalize the feature to [0, 1]
# for emotion tests, filter out value = 4 (other emotions)
# folder list, i.e., IEMCOCAP has 5 sessions, speakers are independent between sessions, always use leave-one-session-out stragegy
def loadData( testTask, testFolder = 4, precision = 'original', sampleRate = 16000, dataType = 'toyWaveform' ):
    
    folderList = [ 0, 1, 2, 3, 4 ]    
    trainFolderList = folderList.copy( )
    del trainFolderList[ testFolder ]
    
    if dataType == 'toyWaveform':
        dataFileFolder = '../../../processedData/toyWaveform/' + str( sampleRate ) + '_' + precision + '/session_'
    elif dataType == 'waveform':
        dataFileFolder = '../../../processedData/waveform/' + str( sampleRate ) + '_' + precision + '/session_'
    elif dataType == 'toySpectrogram':
        dataFileFolder = '../../../processedData/toySpectrogram/' + str( sampleRate ) + '_' + precision + '/session_'
    elif dataType == 'spectrogram':
        dataFileFolder = '../../../processedData/spectrogram/' + str( sampleRate ) + '_' + precision + '/session_'
    
    fold = [ 0, 0, 0, 0, 0 ]
    for i in folderList:
        fold[ i ] = eval( 'iter_loadtxt( dataFileFolder + str(' + str( i + 1 ) + ') + ".csv" )' )
    
    # seperate training and testing data
    trainData = eval( 'np.concatenate( ( fold[ ' + str( trainFolderList[ 0 ] ) + \
                                      ' ], fold[ ' + str( trainFolderList[ 1 ] ) + \
                                      ' ], fold[ ' + str( trainFolderList[ 2 ] ) + \
                                      ' ], fold[ ' + str( trainFolderList[ 3 ] ) + ' ] ), axis=0 )' )
    testData = eval( 'fold[ ' + str( testFolder ) + ' ]' )
    
    if testTask == 'emotion':
        trainFeature, trainLabel = processData( trainData, task = testTask, balance = 'balance', dataType = dataType ) # emotion is not
    else:
        trainFeature, trainLabel = processData( trainData, task = testTask, dataType = dataType )
    testFeature, testLabel = processData( testData, task = testTask, dataType = dataType ) # note: don't balance the test set
    
    plotInputDistribution( trainFeature )
    #plotInputDistribution( testFeature[ 0, : ] )
    
    return trainFeature, trainLabel , testFeature, testLabel

#%% calculate the number of elements of an high-dimensional tensor
def countElements( inputM ):
    inputShape = inputM.shape
    dim = 1
    for i in inputShape:
        dim *= i
    return dim

#%% plot the distribution of data, compatatbale with high-dimensional np arrays
def plotInputDistribution( inputM, saveFolder = '' ):
    output = np.reshape( inputM, [ countElements( inputM ) ] )
    fig1 = plt.figure(  )
    ax1 = fig1.gca()
    binwidth = ( max( output ) - min( output ) )/1000
    ax1.hist( output, bins=np.arange( min( output ), max( output ) + binwidth, binwidth ) )
    if saveFolder != '':
        fig1.savefig( saveFolder + '/hist.png' )

#%% plot the filter for both waveform (1-D), and spectrogram (2-D)
def plotConvFilters( inputM, saveFolder = '' ):
    # inputM in the shape of [ filter_height, filter_width, input_channel_num, output_channel_num ]
    # , in which the total number of 1-D filter is input_channel_num *output_channle_num
    input_channel_num = np.shape( inputM )[ 2 ]
    output_channel_num = np.shape( inputM )[ 3 ]
    fig, ax = plt.subplots( nrows= input_channel_num, ncols= output_channel_num )
    for input_channel in range( 0, input_channel_num ):
        for output_channel in range( 0, output_channel_num ):
            tempFilter = inputM[ :, :, input_channel, output_channel ]
            # if 1-D filter, waveform
            if np.shape( tempFilter )[ 0 ] == 1:
                tempFilter = tempFilter.reshape( np.shape( tempFilter )[ 1 ] )
                if input_channel_num == 1:
                    ax[ output_channel ].set_ylim( [ -0.1, 0.1 ] )
                    ax[ output_channel ].plot( list( range( len( tempFilter ) ) ), tempFilter, linewidth = 0.5 )
                else:
                    ax[ input_channel ][ output_channel ].set_ylim( [ -0.1, 0.1 ] )
                    ax[ input_channel ][ output_channel ].plot( list( range( len( tempFilter ) ) ), tempFilter, linewidth = 0.5 )
            # if 2-D filter, spectrogram
            elif np.shape( tempFilter )[ 0 ] != 1:
                if input_channel_num == 1:
                    ax[ output_channel ].imshow( tempFilter )
                else:
                    ax[ input_channel ][ output_channel ].imshow( tempFilter )
    if input_channel_num != 1:
        fig.set_size_inches( input_channel_num *2, output_channel_num *2 )
    else: 
        fig.set_size_inches( output_channel_num *2, 2 )
    fig.savefig( filename = saveFolder, dpi = 200 )
    
#%% plot TSNE (mainly for dense layer, but can also be used for (flattened) convulutional layers )
def plotTSNE( inputM, label ):
    inputShape = np.shape( inputM )
    
    # if already dense layer, in shape [ n_samples, n_features ]
    if len( inputShape ) == 2:
        tsneResult = calculateTSNE( inputM )
    
    # if conv layers, need first flatten to [ n_samples, n_features ]
    elif len( inputShape ) == 4:
        # flatten inputM 
        num_samples = inputShape[ 0 ]
        num_elements = countElements( inputM )
        outputM = np.reshape( inputM, [ num_samples, int( num_elements /num_samples ) ] )
        tsneResult = calculateTSNE( outputM )
    label = [ mapLabelToColor( elem ) for elem in label ]    
    plt.scatter( x = tsneResult[ :,0 ], y = tsneResult[ :, 1 ], c = label )
    return tsneResult

#%% calculate t-SNE
def calculateTSNE( inputM ):
    randomState = 7
    # if many dimensions, first use PCA than t-sne
    if np.shape( inputM )[ 1 ] >= 128:
        pca_50 = PCA( n_components = 50,  random_state= randomState )
        pca_50_result = pca_50.fit_transform( inputM )
        tsne = TSNE( random_state= randomState )
        tsneResult = tsne.fit_transform( pca_50_result )
    
    # if only a few dimensions, directly use t-sne
    else:
        tsne = TSNE( random_state= randomState )
        tsneResult = tsne.fit_transform( inputM )
    return tsneResult

#%% map label to color
def mapLabelToColor( label ):
    if label == 0:
        color = 'r'
    elif label == 1:
        color = 'b'
    return color