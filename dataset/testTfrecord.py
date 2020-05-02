import tensorflow as tf
from functools import partial
from utils import drawRectangle
import os, cv2
import numpy as np

def main(tfrecordsDir=os.getcwd()+"/dataset/tfrecords/", ronetFlag=False, pnetFlag=False, boxRegressionFlag=True, size=24, showEvery=10):
    if(pnetFlag):
        if(boxRegressionFlag):
            tfrecordPath=tfrecordsDir+'pnetBoxRegression'+str(size)+'.tfrecords'
        else:
            tfrecordPath=tfrecordsDir+'pnetClassification'+str(size)+'.tfrecords'
        featureDescription = {
            'image': tf.io.FixedLenFeature([], tf.string,default_value=''),
            'classification': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'boxOffset': tf.io.FixedLenFeature([], tf.string, default_value='')
        }
        rawDataset = tf.data.TFRecordDataset(tfrecordPath)
        parsedDataset = rawDataset.map(partial(_parse_function, featureDescription=featureDescription))
        showCount=0

        recordCount = 0
        for record in tf.data.TFRecordDataset(tfrecordPath):
            recordCount += 1
        numberOfRecords=recordCount

        for parsedRecord in parsedDataset:
            if(showCount>showEvery):
                imageAsArray=np.asarray(bytearray(parsedRecord['image'].numpy()), dtype='uint8').reshape(size,size,3)
                classification=tf.io.decode_raw(parsedRecord['classification'], 'int64').numpy()
                if(classification[0]==0):
                    label="Non-face"
                else:
                    label="Face"
                imageAsArray=cv2.resize(imageAsArray, (500, 500))
                if(boxRegressionFlag):
                    boxOffsets=tf.io.decode_raw(parsedRecord['boxOffset'], 'float64').numpy()
                    print(boxOffsets)
                    boxOffsets=np.floor(boxOffsets*500).astype(int)
                    print(boxOffsets)
                    drawRectangle(imageAsArray, boxOffsets, coords=True)
                cv2.putText(imageAsArray,label, (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (20,40,200), 2)
                cv2.imshow("Test",imageAsArray)
                keyPressed=cv2.waitKey(0)
                if(keyPressed==27):
                    break
                showCount=0
            showCount+=1

def _parse_function(example_proto, featureDescription):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, featureDescription)

main(pnetFlag=True, showEvery=10000)