from dataset.utils import imagesLister, imagesResizer, imShowRectangles, iou
import re
import os
import numpy as np
from numpy.random import randint as random
import cv2
from math import sqrt, floor, ceil
import tensorflow as tf
from random import shuffle

def generateTfrecord(size=24, numpyDir=os.getcwd()+"/dataset/numpy/", limit=False, pnetFlag=False, rnetFlag=False):
    tfrecordsFolder=os.getcwd()+'/dataset/tfrecords/'
    if not os.path.exists(tfrecordsFolder):
        os.mkdir(tfrecordsFolder)
    
    negativeExamples=[]
    positiveExamples=[]
    partialExamples=[]
    percentage=0
    dataset=np.load(numpyDir+'negativesDataset'+str(size)+'.npy', allow_pickle=True)
    totalNegatives=len(dataset)
    for index,record in enumerate(dataset):
        if(floor((index*100)/totalNegatives)>percentage):
            percentage=floor((index*100)/totalNegatives)
            print("Negatives processed: %d%%" % percentage)
        if(limit and index==limit):
            break
        features = tf.train.Features(feature={
            'image': _bytes_feature(record[0].tostring()),
            'classification': _bytes_feature(record[1].tostring()),
            'boxOffset': _bytes_feature(record[2].tostring())
        })
        negativeExamples.append(tf.train.Example(features=features))
    
    dataset=np.load(numpyDir+'positivesDataset'+str(size)+'.npy', allow_pickle=True)
    totalPositives=len(dataset)
    percentage=0
    for index,record in enumerate(dataset):
        if(floor((index*100)/totalPositives)>percentage):
            percentage=floor((index*100)/totalPositives)
            print("Positives processed: %d%%" % percentage)
        if(limit and index==limit):
            break
        features = tf.train.Features(feature={
            'image': _bytes_feature(record[0].tostring()),
            'classification': _bytes_feature(record[1].tostring()),
            'boxOffset': _bytes_feature(record[2].tostring())
        })
        positiveExamples.append(tf.train.Example(features=features))

    dataset=np.load(numpyDir+'partialsDataset'+str(size)+'.npy', allow_pickle=True)
    totalPartials=len(dataset)
    percentage=0
    for index,record in enumerate(dataset):
        if(floor((index*100)/totalPartials)>percentage):
            percentage=floor((index*100)/totalPartials)
            print("Partials processed: %d%%" % percentage)
        if(limit and index==limit):
            break
        features = tf.train.Features(feature={
            'image': _bytes_feature(record[0].tostring()),
            'classification': _bytes_feature(record[1].tostring()),
            'boxOffset': _bytes_feature(record[2].tostring())
        })
        partialExamples.append(tf.train.Example(features=features))

    if(pnetFlag):
        print("Generating classification TFRecord")
        examples=positiveExamples + negativeExamples
        numberOfRecords=len(examples)
        randomIndexes=list(range(0,numberOfRecords))
        shuffle(randomIndexes)
        percentage=0
        writer = tf.io.TFRecordWriter(tfrecordsFolder+'pnetClassification'+str(size)+'.tfrecords')
        for index, randomIndex in enumerate(randomIndexes):
            if(floor((index*100)/numberOfRecords)>percentage):
                percentage=floor((index*100)/numberOfRecords)
                print("Writting PNet's classification TFRecord: %d%%" % percentage)
            writer.write(examples[randomIndex].SerializeToString())
        writer.close()
        classificationTotal=index

        print("Generating box-regression TFRecord")
        examples=positiveExamples + partialExamples
        numberOfRecords=len(examples)
        randomIndexes=list(range(0,numberOfRecords))
        shuffle(randomIndexes)
        percentage=0
        writer = tf.io.TFRecordWriter(tfrecordsFolder+'pnetBoxRegression'+str(size)+'.tfrecords')
        for index, randomIndex in enumerate(randomIndexes):
            if(floor((index*100)/numberOfRecords)>percentage):
                percentage=floor((index*100)/numberOfRecords)
                print("Writting PNet's box-regression TFRecord: %d%%" % percentage)
            writer.write(examples[randomIndex].SerializeToString())
        writer.close()
        
    if(rnetFlag):
        print("Generating classification TFRecord")
        examples=positiveExamples + negativeExamples + partialExamples
        numberOfRecords=len(examples)
        randomIndexes=list(range(0,numberOfRecords))
        shuffle(randomIndexes)
        percentage=0
        writer = tf.io.TFRecordWriter(tfrecordsFolder+'rnetClassification'+str(size)+'.tfrecords')
        for index, randomIndex in enumerate(randomIndexes):
            if(floor((index*100)/numberOfRecords)>percentage):
                percentage=floor((index*100)/numberOfRecords)
                print("Writting PNet's classification TFRecord: %d%%" % percentage)
            writer.write(examples[randomIndex].SerializeToString())
        writer.close()
        classificationTotal=index

        print("Generating box-regression TFRecord")
        examples=positiveExamples + partialExamples
        numberOfRecords=len(examples)
        randomIndexes=list(range(0,numberOfRecords))
        shuffle(randomIndexes)
        percentage=0
        writer = tf.io.TFRecordWriter(tfrecordsFolder+'rnetBoxRegression'+str(size)+'.tfrecords')
        for index, randomIndex in enumerate(randomIndexes):
            if(floor((index*100)/numberOfRecords)>percentage):
                percentage=floor((index*100)/numberOfRecords)
                print("Writting PNet's box-regression TFRecord: %d%%" % percentage)
            writer.write(examples[randomIndex].SerializeToString())
        writer.close()

        print("RNet's tfrecords generated successfully\nClassification records: %d\nBox-regression records: %d" % (classificationTotal, index))



def _intList_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floatList_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))