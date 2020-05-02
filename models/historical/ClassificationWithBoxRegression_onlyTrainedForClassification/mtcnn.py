import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, PReLU, MaxPooling2D, Reshape, Dense, GlobalMaxPool2D, Softmax
from tensorflow.keras.models import Model, load_model
import os
from functools import partial
import numpy as np
from math import ceil

def pnetModel(inputShape):
    inputLayer = Input(inputShape)

    pnetLayer = Conv2D(10, kernel_size=(3,3), strides=(1,1), padding="valid")(inputLayer)
    pnetLayer = PReLU(shared_axes=[1,2])(pnetLayer)
    pnetLayer = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(pnetLayer)

    pnetLayer = Conv2D(16, kernel_size=(3,3), strides=(1,1), padding="valid")(pnetLayer)
    pnetLayer = PReLU(shared_axes=[1,2])(pnetLayer)

    pnetLayer = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="valid")(pnetLayer)
    pnetLayer = PReLU(shared_axes=[1,2])(pnetLayer)

    pnetClassification = Conv2D(2, kernel_size=(1,1), strides=(1,1))(pnetLayer)
    pnetClassification = GlobalMaxPool2D()(pnetClassification)
    pnetClassificationOutput = Softmax(axis=1, name='classification')(pnetClassification)

    pnetBoxRegression = Conv2D(4, kernel_size=(1,1), strides=(1,1))(pnetLayer)
    pnetBoxRegressionOutput = GlobalMaxPool2D(name='boxRegression')(pnetBoxRegression)

    pnet = Model(inputLayer, [pnetClassificationOutput, pnetBoxRegressionOutput])

    return pnet

def trainPnet(type='pnetBoxRegression', loadModelFlag=True, tfrecordsDir=os.getcwd()+'/dataset/tfrecords/', size=24, inputShape=(None, None, 3), batchSize=500, validationSplit=0.1):
    modelPath=os.getcwd()+'/models/pnetModel.h5'

    if(loadModelFlag==True):
        pnet=load_model(modelPath)
    else:
        pnet=pnetModel(inputShape=inputShape)

    if(type=='pnetClassification'):
        filename=tfrecordsDir+'pnetClassification.tfrecords'
        lossWeights=[1, 0]
    if(type=='pnetBoxRegression'):
        filename=tfrecordsDir+'pnetBoxRegression.tfrecords'
        lossWeights=[0, 1]

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
    
    pnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], loss_weights=lossWeights)
    pnet.fit(trainIterator(parsedTrainDataset, size), validation_data=validationIterator(parsedValidationDataset, size), epochs=10, steps_per_epoch=trainBatches, validation_steps=validationBatches, verbose=1, shuffle=True)
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

trainPnet(inputShape=(24,24,3))