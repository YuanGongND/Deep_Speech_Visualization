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
import testNet
import testNetSimple
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
shutil.copy( '../../model/testNet.py', newFolderName )
shutil.copy( '../../model/testNetSimple.py', newFolderName )
shutil.copy( '../expUtil.py', newFolderName )

# put all configuratation here
thisTask = 'gender'
dataType = 'toyWaveform'
#model = soundNet.soundNet  # define the model
model = testNetSimple.testNet
#model = waveCNN.waveCNNBN

# according to the configuaration, change the coresponding setting 
#if thisTask == 'emotion':
#    trainNewFolderName = newFolderName 

# load data
for testFolder in [ 0, 1, 2, 3, 4 ]:
    trainFeature, trainLabel, testFeature, testLabel = expUtil.loadData( testFolder = testFolder, testTask = thisTask, precision = 'original', sampleRate = 16000, dataType = dataType )
    
    newFolderNameForThisFolder = newFolderName + '/folder' + str( testFolder )
    os.mkdir( newFolderNameForThisFolder )
    # train the model
    resultOnTrain, resultOnTest = expUtil.train( testFeature, testLabel, trainFeature, trainLabel, iteration_num = 100, \
                                                lr_decay = 0.1, batch_size = 32, learningRate = 0.0001, iterationNum = 100, \
                                                modelT = model, newFolderName = newFolderNameForThisFolder, dataType = dataType, visualSign = 1  )


#%% start test  
testSamples = testFeature.shape[ 0 ]
trainSamples = trainFeature.shape[ 0 ]
log = 'testSample_num = ' + str( testSamples ) + '\n trainSample_num = ' + str( trainSamples )
with open( newFolderName + '/log.txt' , "w") as text_file:
    text_file.write( log )


