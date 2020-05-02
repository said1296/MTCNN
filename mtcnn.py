import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, PReLU, MaxPooling2D, Reshape, Dense, GlobalMaxPool2D, Softmax, Flatten, Activation
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
import os
from functools import partial
import numpy as np
from math import ceil

def rnetModel(inputShape=(24, 24, 3)):
    inputLayer = Input(inputShape)

    rnetLayer = Conv2D(28, kernel_size=(3,3), strides=1, padding="valid")(inputLayer)
    rnetLayer = PReLU(shared_axes=[1,2])(rnetLayer)
    rnetLayer = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(rnetLayer)

    rnetLayer = Conv2D(48, kernel_size=(3, 3), strides=1, padding="valid")(rnetLayer)
    rnetLayer = PReLU(shared_axes=[1,2])(rnetLayer)
    rnetLayer = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(rnetLayer)

    rnetLayer = Conv2D(64, kernel_size=(2,2), strides=1, padding="valid")(rnetLayer)
    rnetLayer = PReLU(shared_axes=[1,2])(rnetLayer)

    rnetLayer = Flatten()(rnetLayer)
    rnetLayer = Dense(128)(rnetLayer)

    rnetClassification = Dense(2)(rnetLayer)
    rnetClassification = Softmax(name='classification')(rnetClassification)

    rnetBoxRegression = Dense(4)(rnetLayer)
    rnetBoxRegression = Flatten(name='boxRegression')(rnetBoxRegression)

    rnet = Model(inputLayer, [rnetClassification, rnetBoxRegression])

    return rnet

def pnetModel(inputShape=(None, None, 3)):
    inputLayer = Input(inputShape)

    pnetLayer = Conv2D(10, kernel_size=(3,3), strides=(1,1), padding="valid")(inputLayer)
    pnetLayer = PReLU(shared_axes=[1,2])(pnetLayer)
    pnetLayer = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(pnetLayer)

    pnetLayer = Conv2D(16, kernel_size=(3,3), strides=(1,1), padding="valid")(pnetLayer)
    pnetLayer = PReLU(shared_axes=[1,2])(pnetLayer)

    pnetLayer = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="valid")(pnetLayer)
    pnetLayer = PReLU(shared_axes=[1,2])(pnetLayer)

    pnetClassification = Conv2D(2, kernel_size=(1,1), strides=(1,1))(pnetLayer)
    pnetClassification = Softmax(axis=3)(pnetClassification)
    pnetClassificationOutput = Flatten(name='classification')(pnetClassification)

    pnetBoxRegression = Conv2D(4, kernel_size=(1,1), strides=(1,1))(pnetLayer)
    pnetBoxRegressionOutput = Flatten(name='boxRegression')(pnetBoxRegression)

    pnet = Model(inputLayer, [pnetClassificationOutput, pnetBoxRegressionOutput])

    return pnet

def trainRnet(type='rnetClassification', loadModelFlag=True, tfrecordsDir=os.getcwd()+'/dataset/tfrecords/', size=24, inputShape=(None, None, 3), batchSize=500, validationSplit=0.1, epochs=10):
    
    modelPath=os.getcwd()+'/models/rnetModel.h5'
    if(loadModelFlag==True):
        rnet=load_model(modelPath)
    else:
        rnet=rnetModel()

    if(type=='rnetClassification'):
        filename=tfrecordsDir+'rnetClassification'+str(size)+'.tfrecords'
        lossWeights={'classification': 1, 'boxRegression': 0.5}
    if(type=='rnetBoxRegression'):
        filename=tfrecordsDir+'rnetBoxRegression'+str(size)+'.tfrecords'
        lossWeights={'classification': 0, 'boxRegression': 1}

    recordCount = 0
    for record in tf.data.TFRecordDataset(filename):
        recordCount += 1
    numberOfRecords=recordCount
    numberOfValidationRecords=ceil(numberOfRecords*validationSplit)
    validationBatches=(numberOfValidationRecords/batchSize)
    trainBatches=ceil((numberOfRecords-numberOfValidationRecords)/batchSize)

    featureDescription = {
        'image': tf.io.FixedLenFeature([], tf.string,default_value=''),
        'classification': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'boxOffset': tf.io.FixedLenFeature([], tf.string, default_value='')
    }

    validationDataset=tf.data.TFRecordDataset(filename).take(numberOfValidationRecords)
    parsedValidationDataset=validationDataset.map(partial(_parse_function, featureDescription=featureDescription)).batch(batchSize, drop_remainder=False).repeat()

    trainDataset = tf.data.TFRecordDataset(filename).skip(numberOfValidationRecords)
    parsedTrainDataset = trainDataset.map(partial(_parse_function, featureDescription=featureDescription)).batch(batchSize, drop_remainder=False).repeat()
    
    rnet.compile(optimizer='adam', loss={'classification': 'binary_crossentropy', 'boxRegression': 'mse'}, metrics=['accuracy'], loss_weights=lossWeights)
    rnet.fit(trainIterator(parsedTrainDataset, size), validation_data=validationIterator(parsedValidationDataset, size), epochs=epochs, steps_per_epoch=trainBatches, validation_steps=validationBatches, verbose=1, shuffle=True)
    rnet.save(modelPath)

def trainPnet(type='pnetClassification', loadModelFlag=True, tfrecordsDir=os.getcwd()+'/dataset/tfrecords/', size=24, inputShape=(None, None, 3), batchSize=500, validationSplit=0.1, epochs=10):
    modelPath=os.getcwd()+'/models/pnetModel.h5'

    if(loadModelFlag==True):
        pnet=load_model(modelPath)
    else:
        pnet=pnetModel(inputShape=inputShape)

    if(type=='pnetClassification'):
        filename=tfrecordsDir+'pnetClassification'+str(size)+'.tfrecords'
        lossWeights={'classification': 1, 'boxRegression': 0.5}
    if(type=='pnetBoxRegression'):
        filename=tfrecordsDir+'pnetBoxRegression'+str(size)+'.tfrecords'
        lossWeights={'classification': 0, 'boxRegression': 1}

    recordCount = 0
    for record in tf.data.TFRecordDataset(filename):
        recordCount += 1
    numberOfRecords=recordCount
    numberOfValidationRecords=ceil(numberOfRecords*validationSplit)
    validationBatches=(numberOfValidationRecords/batchSize)
    trainBatches=ceil((numberOfRecords-numberOfValidationRecords)/batchSize)

    featureDescription = {
        'image': tf.io.FixedLenFeature([], tf.string,default_value=''),
        'classification': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'boxOffset': tf.io.FixedLenFeature([], tf.string, default_value='')
    }

    validationDataset=tf.data.TFRecordDataset(filename).take(numberOfValidationRecords)
    parsedValidationDataset=validationDataset.map(partial(_parse_function, featureDescription=featureDescription)).batch(batchSize, drop_remainder=False).repeat()

    trainDataset = tf.data.TFRecordDataset(filename).skip(numberOfValidationRecords)
    parsedTrainDataset = trainDataset.map(partial(_parse_function, featureDescription=featureDescription)).batch(batchSize, drop_remainder=False).repeat()
    
    pnet.compile(optimizer='adam', loss={'classification': 'binary_crossentropy', 'boxRegression': 'mse'}, metrics=['accuracy'], loss_weights=lossWeights)
    pnet.fit(trainIterator(parsedTrainDataset, size), validation_data=validationIterator(parsedValidationDataset, size), epochs=epochs, steps_per_epoch=trainBatches, validation_steps=validationBatches, verbose=1, shuffle=True)
    pnet.save(modelPath)


def trainIterator(dataset, size):
    for batch in dataset:
        imagesList=[]
        classificationList=[]
        boxOffsetsList=[]

        imagesBatch=batch['image']
        classificationBatch=batch['classification']
        boxOffsetsBatch=batch['boxOffset']

        for index, image in enumerate(imagesBatch):
            imageAsArray=np.asarray(bytearray(image.numpy()), dtype='uint8').reshape(size,size,3)
            imageAsArray=(imageAsArray-127.5)*(1/128)
            imagesList.append(imageAsArray)

            classification=tf.io.decode_raw(classificationBatch[index], 'int64').numpy()
            classificationList.append(classification)


            boxOffsets=tf.io.decode_raw(boxOffsetsBatch[index], 'float64').numpy()
            boxOffsetsList.append(boxOffsets)
        imagesAsNp=np.array(imagesList)
        classificationAsNp=np.array(classificationList).astype(np.float32)
        boxOffsetsAsNp=np.array(boxOffsetsList)
        yield imagesAsNp, {'classification': classificationAsNp, 'boxRegression': boxOffsetsAsNp}

def validationIterator(dataset, size):
    for batch in dataset:
        imagesList=[]
        classificationList=[]
        boxOffsetsList=[]

        imagesBatch=batch['image']
        classificationBatch=batch['classification']
        boxOffsetsBatch=batch['boxOffset']

        for index, image in enumerate(imagesBatch):
            imageAsArray=np.asarray(bytearray(image.numpy()), dtype='uint8').reshape(size,size,3)
            imageAsArray=(imageAsArray-127.5)*(1/128)
            imagesList.append(imageAsArray)

            classification=tf.io.decode_raw(classificationBatch[index], 'int64').numpy()
            classificationList.append(classification)


            boxOffsets=tf.io.decode_raw(boxOffsetsBatch[index], 'float64').numpy()
            boxOffsetsList.append(boxOffsets)
        imagesAsNp=np.array(imagesList)
        classificationAsNp=np.array(classificationList).astype(np.float32)
        boxOffsetsAsNp=np.array(boxOffsetsList)
        yield imagesAsNp, {'classification': classificationAsNp, 'boxRegression': boxOffsetsAsNp}

def _parse_function(example_proto, featureDescription):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, featureDescription)