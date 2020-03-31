from utils import imagesLister, imagesResizer, imShowRectangles, iou
import re
import os
import numpy as np
from numpy.random import randint as random
import cv2
from math import sqrt, floor, ceil
import tensorflow as tf
from random import shuffle

size=24
trainDir=os.getcwd()+"/dataset/train/"

def prepareDataset(parseGroundTruthsFlag=False, pnetFlag=False, limit=False):
    positiveImagesFolder    =   trainDir+"positiveImages"
    negativeImagesFolder    =   trainDir+"negativeImages"
    partialImagesFolder     =   trainDir+"partialImages"

    if not os.path.exists(negativeImagesFolder):
        os.mkdir(negativeImagesFolder)
    negativeImagesFile  =   open(os.path.join(negativeImagesFolder, "negativeImages.txt"),'w')
    negativeId=0

    if not os.path.exists(positiveImagesFolder):
        os.mkdir(positiveImagesFolder)
    if not os.path.exists(partialImagesFolder):
        os.mkdir(partialImagesFolder)
    positiveImagesFile  =   open(os.path.join(positiveImagesFolder, "positiveImages.txt"), 'w')
    partialImagesFile   =   open(os.path.join(partialImagesFolder, "partialImages.txt"),'w')
    positiveId=0
    partialId=0

    if(parseGroundTruthsFlag):
        parseGroundTruths()

    groundTruthsParsed = open(os.path.join(trainDir,"groundTruthsParsed.txt"), 'r')
    negativeExamples=[]
    positiveExamples=[]
    partialExamples=[]
    #Parse ground truths to np arrays
    for index, groundTruth in enumerate(groundTruthsParsed):
        if(limit and limit==index):
            break
        groundTruthArray = groundTruth.split(' ')
        if(len(groundTruthArray)<2):
            continue
        imageTest=cv2.imread(trainDir+'images/'+groundTruthArray[0])
        heightImage, widthImage, _ = imageTest.shape
        groundTruthBoxes = np.reshape(np.array(groundTruthArray[2:], dtype='f8'),(-1,4))
        groundTruthBoxesPurged = [box for box in groundTruthBoxes if (box[2]>40 and box[3]>40 and box[2]/box[3]>0.70 and box[3]/box[2]>0.70)] #Select boxes bigger than 40px and with an aspect ratio bigger than 0.7
        if(len(groundTruthBoxesPurged)==0):
            continue

        #Prcess non-faces (negatives), they have less than 20% iou with the groundtruthboxes
        negatives=0
        negativesPerImage=50
        notFoundIteration=0
        if(negativeId%100==0):
            print("Negatives: %d" % negativeId) #Print count of negatives
        while negatives<negativesPerImage:
            notFoundIteration+=1
            sizeCrop = random(40, min(widthImage,heightImage))
            xCrop = random(0,widthImage-sizeCrop)
            yCrop = random(0, heightImage-sizeCrop)
            cropImage = [xCrop, yCrop, sizeCrop]
            #imShowRectangles(imageTest, [cropImage,groundTruthBoxes], color=(100,50,255), labels=['IoU: %f' % iou(cropImage,groundTruthBoxes,True), ['Ground']])
            iouValue=iou(cropImage,groundTruthBoxes)
            if(notFoundIteration>10000):
                #imShowRectangles(imageCropped, croppedBoxes, color=(100,50,255), windowName=trainDir+'images/'+groundTruthArray[0])
                imShowRectangles(imageTest, [cropImage,groundTruthBoxes], color=(100,50,255), labels=['IoU: %f' % iou(cropImage,groundTruthBoxes,True), 'Ground'])
                notFoundIteration=0
            if(iouValue<20 and negatives<negativesPerImage):
                imageCropped=imageTest[yCrop:yCrop+sizeCrop, xCrop:xCrop+sizeCrop, :]
                #imShowRectangles(imageCropped, [croppedBoxes], color=(100,50,255))
                imageCropped=cv2.resize(imageCropped, (size,size))
                features = tf.train.Features(feature={
                    'image': _bytes_feature(imageCropped),
                    'label': _bytes_feature(np.asarray([0, 1]))
                })
                negativeExamples.append(tf.train.Example(features=features))
                negatives+=1
                negativeId+=1

        #Process positives and partial faces
        positivesPerBox = 10
        partialsPerBox = 10
        for groundTruthBox in groundTruthBoxesPurged:
            positives=0
            partials=0
            xBox, yBox, widthBox, heightBox = groundTruthBox
            notFoundIteration=0
            while positives<positivesPerBox or partials<partialsPerBox:
                sizeCrop = random(max(size,ceil(sqrt(0.4*widthBox*heightBox))), floor(sqrt(2.5*widthBox*heightBox)))
                xCrop = random(max(0,ceil(xBox+0.4*widthBox-sizeCrop)),min(widthImage,floor(xBox+0.6*widthBox)))
                yCrop = random(max(0,ceil(yBox+0.4*heightBox-sizeCrop)), min(heightImage,floor(yBox+0.6*heightBox)))
                cropImage = [xCrop, yCrop, sizeCrop]
                if(notFoundIteration>10000):
                    imShowRectangles(imageTest, [cropImage,groundTruthBoxes], color=(100,50,255), labels=['IoU: %f' % iou(cropImage,groundTruthBoxes,True), 'Ground'])
                    imShowRectangles(imageCropped, groundTruthBox, color=(100,50,255), windowName=trainDir+'images/'+groundTruthArray[0])
                    notFoundIteration=0
                notFoundIteration+=1
                iouValue=iou(cropImage,groundTruthBoxes)
                imageCropped=imageTest[yCrop:yCrop+sizeCrop, xCrop:xCrop+sizeCrop, :]
                if(iouValue>=65 and positives<positivesPerBox):
                    #imShowRectangles(imageCropped, croppedBoxes, color=(100,50,255))
                    positiveImagesFile.write('\n%s/%d.jpg' % (positiveImagesFolder, positiveId))
                    offsets=np.asarray(getOffsets(cropImage, groundTruthBox))/sizeCrop #get the offset of the box with respect to the crop and normalize coordinates to percentage of sizeCrop
                    positiveImagesFile.write(' %f %f %f %f' % (offsets[0], offsets[1], offsets[2], offsets[3]))
                    imageCropped=cv2.resize(imageCropped, (size,size))
                    features = tf.train.Features(feature={
                        'image': _bytes_feature(imageCropped),
                        'label': _bytes_feature(np.asarray([1,0])),
                        'offsets': _bytes_feature(np.asarray(offsets))
                    })
                    positiveExamples.append(tf.train.Example(features=features))
                    positives+=1
                    positiveId+=1
                    if(positiveId%100==0):
                        print("Positives: %d" % positiveId)
                elif(iouValue>=40 and partials<partialsPerBox):
                    #imShowRectangles(imageCropped, croppedBoxes, color=(100,50,255))
                    offsets=np.asarray(getOffsets(cropImage, groundTruthBox))/sizeCrop #get the offset of the box with respect to the crop and normalize coordinates to percentage of sizeCrop
                    imageCropped=cv2.resize(imageCropped, (size,size))
                    features = tf.train.Features(feature={
                        'image': _bytes_feature(imageCropped),
                        'offsets': _bytes_feature(np.asarray(offsets))
                    })
                    partialExamples.append(tf.train.Example(features=features))
                    partials+=1
                    partialId+=1
                    if(partialId%100==0):
                        print("Partials: %d" % partialId)

    print("Generating tfrecords...")
    if(pnetFlag==True):
        tfrecordsFolder=os.getcwd()+'/dataset/tfrecords/'

        writer = tf.io.TFRecordWriter(tfrecordsFolder+'pnetDataClassification.tfrecords')
        examples=negativeExamples + positiveExamples
        randomIndexes=list(range(0,len(examples)))
        shuffle(randomIndexes)
        for index in randomIndexes:
            writer.write(examples[index].SerializeToString())
        writer.close()

        writer = tf.io.TFRecordWriter(tfrecordsFolder+'pnetDataBoxRegression.tfrecords')
        examples=positiveExamples + partialExamples
        randomIndexes=list(range(0,len(examples)))
        shuffle(randomIndexes)
        for index in randomIndexes:
            writer.write(examples[index].SerializeToString())
        writer.close()
            

def getOffsets(cropImage, box):
    xCrop, yCrop, sizeCrop = cropImage
    xBox, yBox, widthBox, heightBox = box
    widthBoxCropped=min(xCrop+sizeCrop,xBox+widthBox) - max(xCrop, xBox)
    heightBoxCropped=min(yCrop+sizeCrop, yBox+heightBox) - max(yCrop, yBox)
    xOffsetStart=max(0, xBox-xCrop)
    yOffsetStart=max(0,yBox-yCrop)
    xOffsetEnd=xOffsetStart+widthBoxCropped
    yOffsetEnd=yOffsetStart+heightBoxCropped
    offsets=[xOffsetStart, yOffsetStart, xOffsetStart, xOffsetEnd]
    return offsets

def parseGroundTruths():
    #Read ground truths
    groundTruths = open(os.path.join(trainDir,"groundTruths.txt"), 'r')
    groundTruthsParsed = open(os.path.join(trainDir,"groundTruthsParsed.txt"), 'w')
    #Parse ground truths, a record per line, save in groundTruthsParsed.txt
    for index, groundTruth in enumerate(groundTruths):
        groundTruth=groundTruth.strip()
        if(re.match(r'\d{1,2}--',groundTruth)):
            groundTruthsParsed.write('\n%s' % groundTruth)
        else:
            groundTruthArray=groundTruth.split()
            if(len(groundTruthArray)>1):
                groundTruthsParsed.write(' %s' % ' '.join(groundTruthArray[:4]))
            else:
                groundTruthsParsed.write(' %s' % groundTruth)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    value=value.tostring()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

prepareDataset(False, True, 10000)