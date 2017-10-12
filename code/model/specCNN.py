# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:46:44 2017

@author: Kyle
"""
import tensorflow as tf
import numpy as np
import keras 
import matplotlib.pyplot as plt
from keras.models import Model
from keras import regularizers

defaultDataSetting = {
    'height' : 256,
    'width' : 252,
    'height_use' : 128,
    'trainFile' : 'trainSpecNoNoise.csv',
    'testFile' : 'testSpecNoNoise.csv',
    # 1 = categorical, 2 = valence, 3 = arousal , 4 = dominance
    'labelType' : 99
    }

defaultNetSetting = {
    'input_height' : 256,
    'input_width' : 252,
    'numClasses' : 0,
    'featureNum' : 32, # 32
    'conSize' : 3, # 3
    'dropConv' : 0.5, #0.2
    'dropDense' : 0.5,
    'poolSize' : 2, 
    'denseSize' : 32,
    'denseLayerNum' : 2, # 2     
    'convLayerNum' : 5, # 5
    'epochs' : 200,
    'lrate' : 0.001, # 0.001    
}

#%% test convolutional network
def specCNN( inputHeight = 256, inputWidth = 256, numClass = 2
            netSetting, X_train, y_train, X_test, y_test ):
    
    input = input.resize( [ inputWidth, inputHeight ] )
    plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()
    input = tf.convert_to_tensor( input )
    
    # load setting
    height = netSetting[ 'input_height' ]
    width = netSetting[ 'input_width' ]
    featureNum = netSetting[ 'featureNum' ]
    conSize = netSetting[ 'conSize' ]
    dropConv = netSetting[ 'dropConv' ]
    dropDense = netSetting[ 'dropDense' ]
    poolSize = netSetting[ 'poolSize' ]
    denseSize = netSetting[ 'denseSize' ]
    convLayerNum = netSetting[ 'convLayerNum' ]
    denseLayerNum = netSetting[ 'denseLayerNum' ]
    epochs = netSetting[ 'epochs' ]   
    lrate = netSetting[ 'lrate' ]
    
    # Create the model
    model = Sequential()
    model.add(Convolution2D(featureNum, conSize, conSize, input_shape=(1, height, width), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    #model.add(BatchNormalization())    
    model.add( Dropout( dropConv ) )
    
    
    for layers in range( 1, convLayerNum ):
        model.add(Convolution2D(featureNum, conSize, conSize, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
        #model.add(BatchNormalization())        
        model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))

    model.add(Flatten())
    
    for layers in range( 1, denseLayerNum ):
        model.add(Dense(denseSize, activation='relu', W_constraint=maxnorm(3)))
        model.add( Dropout( dropDense ) )
    
    model.add( Dense( 1, init = 'normal' ) )
    
    # Compile model
    decay = lrate/epochs
    #sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    adam = Adam( lr=lrate, decay = decay )
    model.compile(loss='mean_squared_error', optimizer = adam )    
    print(model.summary())
    
    # Fit the model

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=32 )
     
    # save model    
    time = datetime.datetime.now()
    timeName = str( time.day ) + '_' + str( time.hour ) + '_' +str( time.minute )
    modelName =  timeName + '_' + str( height ) + 'iemoReg.model'
    model.save( modelName )    
    
    # save training history
    historyName =  timeName + '_' + str( height ) + 'iemoReg.history'
    with open( historyName ,'wb') as handle:
        pickle.dump( history.history, handle, protocol=pickle.HIGHEST_PROTOCOL )


    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores *100))
    
    return model