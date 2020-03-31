import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, PReLU, MaxPooling2D, Reshape, Dense, Flatten
from tensorflow.keras.models import Model
import os
from functools import partial
import numpy as np
from math import ceil

def pnetModel(inputShape):
    inputLayer = Input(inputShape)

    pNetLayer = Conv2D(10, kernel_size=(3,3), strides=(1,1), padding="valid")(inputLayer)
    pNetLayer = PReLU(shared_axes=[1,2])(pNetLayer)
    pNetLayer = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(pNetLayer)

    pNetLayer = Conv2D(16, kernel_size=(3,3), strides=(1,1), padding="valid")(pNetLayer)
    pNetLayer = PReLU(shared_axes=[1,2])(pNetLayer)

    pNetLayer = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="valid")(pNetLayer)
    pNetLayer = PReLU(shared_axes=[1,2])(pNetLayer)

    pNetLayer = Flatten()(pNetLayer)
    pNetLayer = Dense(32)(pNetLayer)
    pNetLayer = PReLU()(pNetLayer)

    pNetOutputClassification = Dense(1, activation='sigmoid')(pNetLayer)

    pNet = Model(inputLayer, pNetOutputClassification)

    return pNet

def trainPnet(filename=os.getcwd()+'/dataset/tfrecords/pnetData.tfrecords', size=24, inputShape=(None, None, 3), batchSize=500, validationSplit=0.1):

    recordCount = 0
    for record in tf.data.TFRecordDataset(filename):
        recordCount += 1
    numberOfRecords=recordCount
    numberOfValidationRecords=ceil(numberOfRecords*validationSplit)
    validationBatches=(numberOfValidationRecords/batchSize)
    trainBatches=ceil((numberOfRecords-numberOfValidationRecords)/batchSize)

    featureDescription = {
        'image': tf.io.FixedLenFeature([], tf.string,default_value=''),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }
    validationDataset=tf.data.TFRecordDataset(filename).take(numberOfValidationRecords)
    parsedValidationDataset=validationDataset.map(partial(_parse_function, featureDescription=featureDescription)).batch(batchSize, drop_remainder=False).repeat()

    trainDataset = tf.data.TFRecordDataset(filename).skip(numberOfValidationRecords)
    parsedTrainDataset = trainDataset.map(partial(_parse_function, featureDescription=featureDescription)).batch(batchSize, drop_remainder=False).repeat()


    pNet=pnetModel(inputShape=inputShape)

    pNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    pNet.fit(trainIterator(parsedTrainDataset, size), validation_data=validationIterator(parsedValidationDataset, size), epochs=10, steps_per_epoch=trainBatches, validation_steps=validationBatches,verbose=1, shuffle=True)
    
    pNet.save(os.getcwd()+'/models/pnetModel.h5')

def trainIterator(dataset, size):
    for batch in dataset:
        imagesList=[]
        labelsList=[]
        images=batch['image']
        labels=batch['label']
        for index, image in enumerate(images):
            imageAsArray=np.asarray(bytearray(image.numpy()), dtype='uint8').reshape(size,size,3)
            imageAsArray=(imageAsArray-127.5)*(1/128)
            label=labels[index].numpy()
            label=[label]
            imagesList.append(imageAsArray)
            labelsList.append(label)
        imagesAsNp=np.array(imagesList)
        labelsAsNp=np.array(labelsList)
        yield imagesAsNp, labelsAsNp

def validationIterator(dataset, size):
    for batch in dataset:
        imagesList=[]
        labelsList=[]
        images=batch['image']
        labels=batch['label']
        for index, image in enumerate(images):
            imageAsArray=np.asarray(bytearray(image.numpy()), dtype='uint8').reshape(size,size,3)
            imageAsArray=(imageAsArray-127.5)*(1/128)
            label=labels[index].numpy()
            label=[label]
            imagesList.append(imageAsArray)
            labelsList.append(label)
        imagesAsNp=np.array(imagesList)
        labelsAsNp=np.array(labelsList)
        yield imagesAsNp, labelsAsNp

def _parse_function(example_proto, featureDescription):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, featureDescription)

trainPnet(inputShape=(24,24,3))