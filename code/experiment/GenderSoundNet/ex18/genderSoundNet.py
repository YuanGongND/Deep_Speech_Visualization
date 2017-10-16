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
import waveCNN
sys.path.append("../")
import expUtil
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import shutil

#%% creat folder to save model, the code, and model configuration 
while os.path.isdir( newFolderName ):
    newFolderName = newFolderName + '_1'
    print( 'exist' )

os.mkdir( newFolderName )
shutil.copy( os.path.basename(__file__), newFolderName ) # copy this file to the new folder
shutil.copy( '../../model/soundNet.py', newFolderName )
shutil.copy( '../../model/waveCNN.py', newFolderName )
shutil.copy( '../expUtil.py', newFolderName )

# put all configuratation here
thisTask = 'gender'
dataType = 'waveform'

# define the model
model = soundNet.soundNet  # define the model
#model = waveCNN.waveCNN

# according to the configuaration, change the coresponding setting 
#if thisTask == 'emotion':
#    trainNewFolderName = newFolderName 

# load data
trainFeature, trainLabel, testFeature, testLabel = expUtil.loadData( testFolder = 4, testTask = thisTask, precision = 'original', sampleRate = 16000, dataType = dataType )

#%% grid search

#batch_sizeList = [ 32, 24, 16 ]
#learningRateList = [ 1e-3, 5e-4, 1e-4, 5e-5, 1e-5 ]
#initList = [ 'RandomUniform', 'lecun_normal', 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform' ]
batch_sizeList = [ 32 ]
learningRateList = [ 1e-4 ]
initList = [ 'glorot_normal' ]
for batch_size in batch_sizeList:
    resultList = [  ]
    for learningRate in learningRateList:    
        for init in initList:
            tempFolderName = newFolderName + '/' + str( learningRate ) + '_' + str( batch_size ) + '_' + init
            os.mkdir( tempFolderName )
            # train the model
            resultOnTrain, resultOnTest = expUtil.train( testFeature, testLabel, trainFeature, trainLabel, iteration_num = 100, \
                                                        lr_decay = 0.1, batch_size = batch_size, learningRate = learningRate, iterationNum = 100, \
                                                        modelT = model, newFolderName = tempFolderName, init = init, saveSign = True, denseUnitNum = 64, \
                                                        dataType = dataType )
            resultList.append( resultOnTest[ -1 ] )
            np.savetxt( newFolderName + '\_' + str( batch_size ) +'_gridSearch.csv', resultList, delimiter = ',' )
    resultList = np.array( resultList )
    resultList.resize( [ len( learningRateList ), len( initList ) ] )
    np.savetxt( newFolderName + '\_' + str( batch_size ) +'_gridSearch.csv', resultList, delimiter = ',' )

#%% start test  
testSamples = testFeature.shape[ 0 ]
trainSamples = trainFeature.shape[ 0 ]
log = 'testSample_num = ' + str( testSamples ) + '\n trainSample_num = ' + str( trainSamples )
with open( newFolderName + '/log.txt' , "w") as text_file:
    text_file.write( log )


