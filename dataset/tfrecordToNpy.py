import tensorflow as tf
from functools import partial
from utils import imShowRectangles
import os, cv2
import numpy as np

def main(tfrecordsDir=os.getcwd()+"/dataset/tfrecords/", numpyDir=os.getcwd()+"/dataset/numpy/", ronetFlag=False, pnetFlag=False, size=24, showEvery=10):
    if(pnetFlag):
        tfrecordPath=tfrecordsDir+'pnetData.tfrecords'
        rawDataset = tf.data.TFRecordDataset(tfrecordPath)

        featureDescription = {
            'image': tf.io.FixedLenFeature([], tf.string,default_value=''),
            'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
        }
        parsedDataset = rawDataset.map(partial(_parse_function, featureDescription=featureDescription))

        imagesList=[]
        labelsList=[]
        for parsedRecord in parsedDataset:
            imageAsArray=np.asarray(bytearray(parsedRecord['image'].numpy()), dtype='uint8').reshape(size,size,3)
            imageAsArray=(imageAsArray-127.5)*(1/128)
            label=parsedRecord['label'].numpy()
            label=[label]
            imagesList.append(imageAsArray)
            labelsList.append(label)
        imagesAsNp=np.array(imagesList)
        labelsAsNp=np.array(labelsList)
        np.save(numpyDir+'pnetImages', imagesAsNp)
        np.save(numpyDir+'pnetLabels', imagesAsNp)
        
    print(labelsAsNp)
    print(labelsAsNp.shape)


    if(ronetFlag):
        tfrecordPath=tfrecordsDir+'ronetData.tfrecords'
        featureDescription = {
            'image': tf.io.FixedLenFeature([], tf.string, default_value='',),
            'xStartBox': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0, allow_missing=True),
            'yStartBox': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0, allow_missing=True),
            'xEndBox': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0, allow_missing=True, ),
            'yEndBox': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0, allow_missing=True, )
        }
        rawDataset = tf.data.TFRecordDataset(tfrecordPath)
        parsedDataset = rawDataset.map(partial(_parse_function, featureDescription=featureDescription))
        showCount=0
        for parsedRecord in parsedDataset.take(numTests):
            if(showCount>showEvery):
                imageAsArray=np.asarray(bytearray(parsedRecord['image'].numpy()), dtype='uint8').reshape(size,size,3)
                labels=np.asarray([parsedRecord['xStartBox'].numpy(), parsedRecord['yStartBox'].numpy(), parsedRecord['xEndBox'].numpy(), parsedRecord['yEndBox'].numpy()])
                labels=np.transpose(labels)
                labels=np.floor(labels*size)
                imShowRectangles(imageAsArray, [labels], thickness=1, coords=True)
            showCount+=1

def _parse_function(example_proto, featureDescription):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, featureDescription)

main(pnetFlag=True)
