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
import math
import random

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
        feature = ( ( feature - np.mean( feature ) ) /math.sqrt( np.var( feature ) ) )
         
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
        feature = ( ( feature - np.mean( feature ) ) /math.sqrt( np.var( feature ) ) )
        
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
    
    # define data size, different size for waveform and spectrogram
    if dataType == 'waveform' or dataType == 'toyWaveform':
        dataSize = 96000
    else:
        dataSize = 256 *256
    
    # make folders 
    os.mkdir( newFolderName + '/weight' )
    os.mkdir( newFolderName + '/models' )
    os.mkdir( newFolderName + '/figure' )
    os.mkdir( newFolderName + '/figure/TSNE' )
    os.mkdir( newFolderName + '/figure/convFilter' )
    
    # initialize 
    result = np.zeros( [ 2, iteration_num ] )
    diffResult = np.zeros( [ 8, iteration_num ] )
    class_num = testLabel.shape[ 1 ]
    train_datasize = trainFeature.shape[ 0 ]
    
    tf.set_random_seed( 7 )
    with tf.Session() as sess:
        
        # changable learning rate 
        global_step = tf.Variable(0)  
        learning_rate = tf.train.exponential_decay( learningRate, global_step, int( iteration_num *(train_datasize/batch_size) ), lr_decay, staircase=False)  

        # fix random index for reproducing result 
        tf.set_random_seed( 17 )
        
        # define place holders 
        input_x = tf.placeholder( tf.float32, shape = ( batch_size, dataSize ), name = 'inputx' )
        input_y = tf.placeholder( tf.float32, shape = ( batch_size, class_num ), name = 'inputy' )
        
        # define a set of tensors for training
        prediction = modelT( input_x, numClass = class_num, l2_reg = 0.5, init = init, denseUnitNum  = denseUnitNum )
        loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = prediction, labels= input_y ) )
        train_step = tf.train.AdamOptimizer( learning_rate ).minimize( loss, global_step = global_step )
        #train_step = tf.train.GradientDescentOptimizer( learning_rate ).minimize( loss, global_step = global_step )
        correct_prediction = tf.equal( tf.argmax( prediction, 1 ), tf.argmax( input_y, 1 )  )
        accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ), name="acc_restore" )
        
        # initialize the variables
        init_op = tf.global_variables_initializer(  )
        sess.run( init_op )
        
        # initialize the model saver
        saver = tf.train.Saver( max_to_keep= 100 )
        
        # print the list of variables
        print( tf.trainable_variables() )
        
        # number of iterations
        for iteration in range( 0, iteration_num ):
            
            # each batch
            for i in range( 0, 1 *int( train_datasize / batch_size ) ):
                
                # prepare data for each train batch
                start = ( i * batch_size ) % train_datasize
                end = min( start + batch_size, train_datasize )
                inputTrainFeature = trainFeature[ start: end ]
                inputTrainLabel = trainLabel[ start: end ]
                
                # train the model
                _, lossShow = sess.run( [ train_step, loss ], feed_dict = { input_x: inputTrainFeature, input_y: inputTrainLabel } )
                #print( 'loss = ' + str( lossShow ) )
             
            # get accuracy on a small subset of test data (just several epoch), a very fast approximation of the performance 
            # number of batches to test
            testBatchNum = 3
            # initialize result recorder
            testSubsetResult = [ None ] *( batch_size *testBatchNum )
            testSubsetLabel = [ None ] *( batch_size *testBatchNum )
            outputBeforeDense = [ None ] *( batch_size *testBatchNum )
            outputDense1 = [ None ] *( batch_size *testBatchNum )
            # get intermediate tensor 
            flattenOut = sess.graph.get_tensor_by_name( 'flatten/flattenOut:0' )   
            dense1Out = sess.graph.get_tensor_by_name( 'dense1/dense1Out:0' )        
            # start test
            for testBatch in range( 0, testBatchNum ): # 3*32=96 test samples
                # prepare input data    
                start = testBatch * batch_size 
                end = start + batch_size
                inputTestFeature = testFeature[ start: end, : ]
                inputTestLabel = testLabel[ start: end, : ]     
                # run test
                tempTestResult, tempAccuracyTest, tempoutputBeforeDense, tempoutputDense1 = sess.run( [ prediction, accuracy, flattenOut, dense1Out ], feed_dict = { input_x: inputTestFeature, input_y: inputTestLabel } ) 
                # record result
                testSubsetLabel[ start :end ] = np.argmax( inputTestLabel, 1 )
                testSubsetResult[ start :end ] = np.argmax( tempTestResult, 1 ) 
                outputBeforeDense[ start :end ] = tempoutputBeforeDense
                outputDense1[ start :end ] = tempoutputDense1
                
            # plot the t-SNE before the dense layer
            plotTSNE( outputBeforeDense, testSubsetLabel, newFolderName + '/figure/TSNE/tSNE_1_' + str( iteration ) + '.png' )
            plotTSNE( outputDense1, testSubsetLabel, newFolderName + '/figure/TSNE/tSNE_2_' + str( iteration ) + '.png' )
            
            if iteration == 119:
                plotALotTSNE( outputBeforeDense, testSubsetLabel, newFolderName + '/figure/TSNE/tSNE_3_' )
                plotALotTSNE( outputDense1, testSubsetLabel, newFolderName + '/figure/TSNE/tSNE_4_' )
                
            #np.savetxt( newFolderName + '/testResult.csv', testResult, delimiter = ',' )
            #np.savetxt( newFolderName + '/testLabel.csv', inputTestLabel, delimiter = ',' )
            accuracyTest = accuracy_score( testSubsetLabel, testSubsetResult )
            print( confusion_matrix( testSubsetLabel, testSubsetResult ) )
            result[ 0, iteration ] = accuracyTest
            print( 'Epoch:' + str( iteration + 1 ) + ' result on test: ' + str( accuracyTest ) )
            
            # get accuracy on a small subset of training data (just one epoch), a very fast approximation of the training loss/ overfitting 
            inputTestTrainFeature = trainFeature[ 0: batch_size, : ]
            inputTestTrainLabel = trainLabel[ 0: batch_size, : ]
            testTrainResult, accuracyTrain = sess.run( [ prediction, accuracy ], feed_dict = { input_x: inputTestTrainFeature, input_y: inputTestTrainLabel } ) 
            print( 'Epoch:' + str( iteration + 1 ) + ' result on train: ' + str( accuracyTrain ) )
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
                lastState, _ = printVariable( sess, newFolderName = newFolderName )
            else:
                lastState, diffThisIter = printVariable( sess, lastState, iteration + 1, newFolderName = newFolderName )
                diffResult[ :, iteration - 1 ] = diffThisIter
            #np.savetxt( newFolderName + '/weightConv1' + str( iteration + 1 ) + '.csv', lastState, delimiter = ',' )
            
            # save model every 10 epoches
            #if ( iteration + 1 )%10 == 0 and saveSign == True:
            if ( iteration + 1 ) < 10 and saveSign == True:
                save_path = saver.save( sess, newFolderName + '/models/' + str( iteration + 1 ) + '_.ckpt' )
                print("Model saved in file: %s" % save_path)
            
            # plot the result
            resultOnTest = result[ 0, : ]
            resultOnTrain = result[ 1, : ]
            plt.plot( list( range( iteration_num ) ), resultOnTrain )
            plt.plot( list( range( iteration_num ) ), resultOnTest )
            plt.savefig( newFolderName + '/accuracy.png' )
            plt.close('all')
            for diffLayerIndex in range( 0, 8 ):
                plt.plot(  list( range( iteration_num ) ), diffResult[ diffLayerIndex, : ], label = 'conv_' + str( diffLayerIndex ) )
            plt.legend( 'upper right' )
            plt.savefig( newFolderName + '/diff.png' )
            plt.close( 'all' )
            
    return resultOnTrain, resultOnTest

#%%
def printVariable( sess, lastState = -1, iteration = 1, newFolderName = -1 ):     
      #layerList = [ 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'dense1', 'dense2' ]
      layerList = [  'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8' ]
      diffList = layerList.copy(  )
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
                  plotConvFilters( kernal, newFolderName + '/figure/convFilter/' + layerList[ layerIndex ] + '_' + str( iteration ) + '.png' )
              else:
                  plotRandomConvFilters( kernal, newFolderName + '/figure/convFilter/' + layerList[ layerIndex ] + '_' + str( iteration ) + '.png' )
              
              # save the weight of current layer
              currentState[ layerIndex ] = kernal
              
              # track the difference with last state
              if lastState != -1:
                  diff = trackChange( currentState[ layerIndex ], lastState[ layerIndex ] )
                  diffList[ layerIndex ] = diff
                  print( layerList[ layerIndex ] + ' : ' + str( diff )  )
                  #trackChangeEachChannel( currentState[ layerIndex ], lastState[ layerIndex ] )
                  
#                  if layerIndex <= 1:
#                      print( trackChangeEachChannel( currentState[ layerIndex ], lastState[ layerIndex ] ) )
          
      return currentState, diffList

#%% track change ( in percentile ) of different layers according to epochs
def trackChange( currentState, lastState ):
    difference = lastState - currentState 
    diffInPercentile = difference / currentState
    absDiffInPercentile = np.abs( diffInPercentile )
    meanChange = np.mean( absDiffInPercentile )
    return meanChange *100

#%% track change ( in percentile ) of different layers according to epochs for each channel
def trackChangeEachChannel( currentState, lastState ):
    difference = lastState - currentState 
    diffInPercentile = difference / currentState
    absDiffInPercentile = np.abs( diffInPercentile )
    absDiff = np.abs( difference )
    
    input_channel_num = np.shape( difference )[ 2 ]
    output_channel_num = np.shape( difference )[ 3 ]
    #initialize an array to track the change of each channel
    diffPerChannel = np.zeros( [ input_channel_num, output_channel_num ] )
    # for each channel 
    for input_channel in range( 0, input_channel_num ):
        for output_channel in range( 0, output_channel_num ):
            tempChannelDiff = absDiff[ 0, :, input_channel, output_channel ]
            #tempChannelDiff = absDiffInPercentile[ 0, :, input_channel, output_channel ]
            diffPerChannel[ input_channel, output_channel ] = np.mean( tempChannelDiff )
    
    print( 'mean: ' + str( np.mean( diffPerChannel ) *100 ))
    print( 'max: ' + str( np.max( diffPerChannel ) *100 ) )
    print( 'min: ' + str( np.min( diffPerChannel ) *100 ) )
    print( '-------------' )
    
    return diffPerChannel *100
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
        plt.close('all')

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
    plt.close('all')
    
#%% plot the filter for both waveform (1-D), and spectrogram (2-D), since the number is large, this function only select a few for each layer
def plotRandomConvFilters( inputM, saveFolder = '', plot_num = 16 ):
    # inputM in the shape of [ filter_height, filter_width, input_channel_num, output_channel_num ]
    # , in which the total number of 1-D filter is input_channel_num *output_channle_num
    input_channel_num = np.shape( inputM )[ 2 ]
    output_channel_num = np.shape( inputM )[ 3 ]
    random.seed( 7 )
    selectedFilter = random.sample( range( 0, min( input_channel_num, output_channel_num ) ), plot_num )
    fig, ax = plt.subplots( nrows= 1, ncols= plot_num )
    for randomIndex in range( 0, plot_num ):
            tempFilter = inputM[ :, :, selectedFilter[ randomIndex ], selectedFilter[ randomIndex ] ]
            # if 1-D filter, waveform
            if np.shape( tempFilter )[ 0 ] == 1:
                tempFilter = tempFilter.reshape( np.shape( tempFilter )[ 1 ] )
                ax[ randomIndex ].set_ylim( [ -0.1, 0.1 ] )
                ax[ randomIndex ].plot( list( range( len( tempFilter ) ) ), tempFilter, linewidth = 0.5 )
            # if 2-D filter, spectrogram
            elif np.shape( tempFilter )[ 0 ] != 1:
                ax[ randomIndex ].imshow( tempFilter )

    fig.set_size_inches( plot_num *2, 2 )
    fig.savefig( filename = saveFolder, dpi = 200 )
    plt.close('all')
    
#%%
def plotALotTSNE( inputM, label, saveFolder ):
    perplexityList = [ 5, 10, 15, 20 ]
    lrList = [ 10 ]
    n_iterList = list( range( 500, 5000, 100 ) )
    for perplexity in perplexityList:
        for lr in lrList:
            for n_iter in n_iterList:
                fileName = saveFolder + str( perplexity ) + '_' + str( lr ) + '_' + str( n_iter ) + '.png'
                plotTSNE( inputM, label, fileName, perplexity = perplexity, n_iter = n_iter, learning_rate = lr )
                

#%% plot TSNE (mainly for dense layer, but can also be used for (flattened) convulutional layers )
def plotTSNE( inputM, label, saveFolder, perplexity= 16.0, n_iter = 5000, learning_rate = 10 ):
    inputShape = np.shape( inputM )
    
    # if already dense layer, in shape [ n_samples, n_features ]
    if len( inputShape ) == 2:
        tsneResult = calculateTSNE( inputM,  perplexity= perplexity, n_iter = n_iter, learning_rate = learning_rate )
    
    # if conv layers, need first flatten to [ n_samples, n_features ]
    elif len( inputShape ) == 4:
        # flatten inputM 
        num_samples = inputShape[ 0 ]
        num_elements = countElements( inputM )
        outputM = np.reshape( inputM, [ num_samples, int( num_elements /num_samples ) ] )
        tsneResult = calculateTSNE( outputM, perplexity= perplexity, n_iter = n_iter, learning_rate = learning_rate )
    label = [ mapLabelToColor( elem ) for elem in label ]    
    plt.scatter( x = tsneResult[ :,0 ], y = tsneResult[ :, 1 ], c = label )
    plt.savefig( filename = saveFolder, dpi = 100 )
    plt.close('all')
    return tsneResult

#%% calculate t-SNE
def calculateTSNE( inputM, perplexity= 16.0, n_iter = 5000, learning_rate = 10 ):
    randomState = 7
    tsne = TSNE( random_state= randomState, perplexity= perplexity, n_iter_without_progress = n_iter, learning_rate = learning_rate )
    # if many dimensions, first use PCA than t-sne
    if np.shape( inputM )[ 1 ] >= 128:
        pca_50 = PCA( n_components = 128,  random_state= randomState )
        pca_50_result = pca_50.fit_transform( inputM )
        tsneResult = tsne.fit_transform( pca_50_result )
    
    # if only a few dimensions, directly use t-sne
    else:
        tsneResult = tsne.fit_transform( inputM )
#        pca_50 = PCA( n_components = 2,  random_state= randomState )
#        tsneResult = pca_50.fit_transform( inputM )
    return tsneResult

#%% map label to color
def mapLabelToColor( label ):
    if label == 0:
        color = 'r'
    elif label == 1:
        color = 'b'
    elif label == 2:
        color = 'm'
    elif label == 3:
        color = 'k'
    return color

def sineInit( shape, dtype=None ):
    InitKernal = np.zeros( shape )
    for filterIndex in range( 0, shape[ 3 ] ):
        InitKernal[ 0, :, 0, filterIndex ] = genSineFilter( 200 *( filterIndex + 1 ) )
        InitKernal = InitKernal /64
    return InitKernal

#%%
def genSineFilter( frequency, points = 64, sampleRate = 16000 ):
    Ts = 1 /sampleRate
    t = list( np.linspace( -points/2*Ts, points/2*Ts, num= points ) )
    #t = list( xrange( -points/2*Ts, points/2*Ts-Ts, Ts ) )
    sinFilter = [ math.sin( 2 * math.pi * frequency *elem) for elem in t ]
    #plt.plot( sinFilter )
    return sinFilter