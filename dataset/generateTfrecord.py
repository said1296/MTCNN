from utils import imagesLister, imagesResizer, imShowRectangles, iou
import re
import os
import numpy as np
from numpy.random import randint as random
import cv2
from math import sqrt, floor, ceil
import tensorflow as tf
from random import shuffle

def generateTfrecord(size=24, trainDir=os.getcwd()+"/dataset/train/", limit=False, pnetFlag=False, ronetFlag=False):
    positiveImagesFolder    =   trainDir+"positiveImages"
    negativeImagesFolder    =   trainDir+"negativeImages"
    partialImagesFolder     =   trainDir+"partialImages"
    tfrecordsFolder         =   os.getcwd()+'/dataset/tfrecords/'

    if not os.path.exists(positiveImagesFolder):
        os.mkdir(positiveImagesFolder)
    if not os.path.exists(negativeImagesFolder):
        os.mkdir(negativeImagesFolder)
    if not os.path.exists(partialImagesFolder):
        os.mkdir(partialImagesFolder)

    negativeImagesFile  =   open(os.path.join(negativeImagesFolder, "negativeImages.txt"),'r')
    positiveImagesFile  =   open(os.path.join(positiveImagesFolder, "positiveImages.txt"), 'r')
    partialImagesFile   =   open(os.path.join(partialImagesFolder, "partialImages.txt"),'r')

    with open(os.path.join(negativeImagesFolder, "negativeImages.txt")) as f:
        if(limit):
            i=limit
        else:
            for i, l in enumerate(f):
                pass
            
    totalNegatives=i+1

    with open(os.path.join(positiveImagesFolder, "positiveImages.txt")) as f:
        if(limit):
            i=limit
        else:
            for i, l in enumerate(f):
                pass
    totalPositives=i+1

    with open(os.path.join(partialImagesFolder, "partialImages.txt")) as f:
        if(limit):
            i=limit
        else:
            for i, l in enumerate(f):
                pass
    totalPartials=i+1
    
    if not os.path.exists(tfrecordsFolder):
        os.mkdir(tfrecordsFolder)
    
    if(pnetFlag):
        writer = tf.io.TFRecordWriter(tfrecordsFolder+'pnetData.tfrecords')
        examples=[]
        percentage=0
        for index,record in enumerate(negativeImagesFile):
            if(floor((index*100)/totalNegatives)>percentage):
                percentage=floor((index*100)/totalNegatives)
                print("PNet's negatives processed: %d%%" % percentage)
            if(limit and index==limit):
                break
            record=record.strip().split(' ')
            if(len(record)<2):
                continue
            imageRecord = cv2.imread(record[0]).tostring()

            features = tf.train.Features(feature={
                'image': _bytes_feature(imageRecord),
                'label': _int64_feature(0)
            })
            examples.append(tf.train.Example(features=features))
        
        percentage=0
        for index,record in enumerate(positiveImagesFile):
            if(floor((index*100)/totalPositives)>percentage):
                percentage=floor((index*100)/totalPositives)
                print("PNet's positives processed: %d%%" % percentage)
            if(limit and index==limit):
                break
            record=record.strip().split(' ')
            if(len(record)<2):
                continue
            imageRecord = cv2.imread(record[0]).tostring()

            features=tf.train.Features(feature={
                'image': _bytes_feature(imageRecord),
                'label': _int64_feature(1)
            })
            examples.append(tf.train.Example(features=features))

        numberOfRecords=len(examples)
        randomIndexes=list(range(0,numberOfRecords))
        shuffle(randomIndexes)
        percentage=0
        for index, randomIndex in enumerate(randomIndexes):
            if(floor((index*100)/numberOfRecords)>percentage):
                percentage=floor((index*100)/numberOfRecords)
                print("Writting PNet's TFRecord: %d%%" % percentage)
            writer.write(examples[randomIndex].SerializeToString())
        writer.close()

    if(ronetFlag):
        writer = tf.io.TFRecordWriter(tfrecordsFolder+'ronetData.tfrecords')
        examples=[]
        percentage=0
        for index,record in enumerate(positiveImagesFile):
            if(floor((index*100)/totalPositives)>percentage):
                percentage=floor((index*100)/totalPositives)
                print("R/ONet's positives processed: %d%%" % percentage)
            if(limit and index==limit):
                break
            record=record.strip().split(' ')
            if(len(record)<2):
                continue
            imageRecord = cv2.imread(record[0]).tostring()
            groundTruthBoxes= record[1:]
            groundTruthBoxes=np.reshape(np.array(groundTruthBoxes), (-1,4))
            groundTruthBoxes=np.transpose(groundTruthBoxes)
            groundTruthBoxes=groundTruthBoxes.astype(float)

            features = tf.train.Features(feature={
                'image': _bytes_feature(imageRecord),
                'xBox': _floatList_feature(groundTruthBoxes[0]),
                'yBox': _floatList_feature(groundTruthBoxes[1]),
                'widthBox': _floatList_feature(groundTruthBoxes[2]),
                'heightBox': _floatList_feature(groundTruthBoxes[3]),
            })
            examples.append(tf.train.Example(features=features))
        
        percentage=0
        for index,record in enumerate(partialImagesFile):
            if(floor((index*100)/totalPartials)>percentage):
                percentage=floor((index*100)/totalPartials)
                print("R/ONet's partials processed: %d%%" % percentage)
            if(limit and index==limit):
                break
            record=record.strip().split(' ')
            if(len(record)<2):
                continue
            imageRecord = cv2.imread(record[0]).tostring()
            groundTruthBoxes= record[1:]
            groundTruthBoxes=np.reshape(np.array(groundTruthBoxes), (-1,4))
            groundTruthBoxes=np.transpose(groundTruthBoxes)
            groundTruthBoxes=groundTruthBoxes.astype(float)

            features = tf.train.Features(feature={
                'image': _bytes_feature(imageRecord),
                'xStartBox': _floatList_feature(groundTruthBoxes[0]),
                'yStartBox': _floatList_feature(groundTruthBoxes[1]),
                'xEndBox': _floatList_feature(groundTruthBoxes[2]),
                'yEndBox': _floatList_feature(groundTruthBoxes[3])
            })
            examples.append(tf.train.Example(features=features))

        numberOfRecords=len(examples)
        randomIndexes=list(range(0,numberOfRecords))
        shuffle(randomIndexes)
        percentage=0
        for index, randomIndex in enumerate(randomIndexes):
            if(floor((index*100)/numberOfRecords)>percentage):
                percentage=floor((index*100)/numberOfRecords)
                print("Writting R/ONet's TFRecord: %d%%" % percentage)
            writer.write(examples[randomIndex].SerializeToString())
        writer.close()



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