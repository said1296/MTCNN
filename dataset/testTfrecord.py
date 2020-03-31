import tensorflow as tf
from functools import partial
from utils import imShowRectangles
import os, cv2
import numpy as np

def main(tfrecordsDir=os.getcwd()+"/dataset/tfrecords/", ronetFlag=False, pnetFlag=False, size=24, showEvery=10, numTests=2000):
    if(pnetFlag):
        tfrecordPath=tfrecordsDir+'pnetDataClassification.tfrecords'
        featureDescription = {
            'image': tf.io.FixedLenFeature([], tf.string,default_value=''),
            'label': tf.io.FixedLenFeature([], tf.string, default_value='')
        }
        rawDataset = tf.data.TFRecordDataset(tfrecordPath)
        parsedDataset = rawDataset.map(partial(_parse_function, featureDescription=featureDescription))
        showCount=0
        for parsedRecord in parsedDataset.take(numTests):
            if(showCount>showEvery):
                imageAsArray=np.asarray(bytearray(parsedRecord['image'].numpy()), dtype='uint8').reshape(size,size,3)
                label=tf.io.decode_raw(parsedRecord['label'], 'int64').numpy()
                cv2.imshow(str(label),imageAsArray)
                cv2.waitKey(0)
                cv2.waitKey(5000)
            showCount+=1

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