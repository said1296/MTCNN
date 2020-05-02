from dataset.utils import imagesLister, imagesResizer, imShowRectangles, iou
import re
import os
import numpy as np
from numpy.random import randint as random
import cv2
from math import sqrt, floor, ceil
import tensorflow as tf
from random import shuffle

# Negatives: 417800
# Positives: 202160
# Partials: 202160

trainDir=os.getcwd()+"/dataset/train/"

def prepareDataset(parseGroundTruthsFlag=False, size=24, limit=False):
    numpyFolder=os.getcwd()+"/dataset/numpy/"

    if(parseGroundTruthsFlag):
        parseGroundTruths()

    c=0
    for line in open(trainDir+'groundTruthsParsed.txt'):
        c+=1
    totalRecords=c

    if(limit):
        totalRecords=limit

    groundTruthsParsed = open(os.path.join(trainDir,"groundTruthsParsed.txt"), 'r')
    negativesList=[]
    positivesList=[]
    partialsList=[]
    #Parse ground truths to np arrays
    percentage=0
    for index, groundTruth in enumerate(groundTruthsParsed):
        if(floor((index*100)/totalRecords)>percentage):
            percentage=floor((index*100)/totalRecords)
            print("Records processed: %d%%" % percentage)
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
        negativesPerImage=40
        notFoundIteration=0
        while negatives<negativesPerImage:
            notFoundIteration+=1
            sizeCrop = random(40, min(widthImage,heightImage))
            xCrop = random(0,widthImage-sizeCrop)
            yCrop = random(0, heightImage-sizeCrop)
            cropImage = [xCrop, yCrop, sizeCrop]
            #imShowRectangles(imageTest, [cropImage,groundTruthBoxes], color=(100,50,255), labels=['IoU: %f' % iou(cropImage,groundTruthBoxes,True), ['Ground']])
            iouValue=iou(cropImage,groundTruthBoxes)
            if(notFoundIteration>10000):
                imageCopped=imageTest[yCrop: yCrop+sizeCrop, xCrop:xCrop+sizeCrop, :]
                #imShowRectangles(imageCropped, croppedBoxes, color=(100,50,255), windowName=trainDir+'images/'+groundTruthArray[0])
                imShowRectangles(imageTest, [cropImage,groundTruthBoxes], windowName='NOT FOUND NEGATIVES', color=(100,50,255), labels=['IoU: %f' % iou(cropImage,groundTruthBoxes,True), 'Ground'])
                imShowRectangles(imageCropped, [[0,0,0,0]],  color=(100,50,255), windowName=trainDir+'images/'+groundTruthArray[0])
                notFoundIteration=0
            if(iouValue<20 and negatives<negativesPerImage):
                imageCropped=imageTest[yCrop:yCrop+sizeCrop, xCrop:xCrop+sizeCrop, :]
                #imShowRectangles(imageCropped, [croppedBoxes], color=(100,50,255))
                imageCropped=cv2.resize(imageCropped, (size,size))
                data=np.array([imageCropped, np.array([0, 1]), np.array([0, 0, 0, 0])])
                negativesList.append(data)
                negatives+=1

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
                    imageCopped=imageTest[yCrop: yCrop+sizeCrop, xCrop:xCrop+sizeCrop, :]
                    imShowRectangles(imageTest, [cropImage,groundTruthBoxes], windowName='NOT FOUND POSITIVES PARTIALS', color=(100,50,255), labels=['IoU: %f' % iou(cropImage,groundTruthBoxes,True), 'Ground'])
                    imShowRectangles(imageCropped, [groundTruthBox], color=(100,50,255), windowName=trainDir+'images/'+groundTruthArray[0])
                    notFoundIteration=0
                notFoundIteration+=1
                iouValue=iou(cropImage,groundTruthBoxes)
                imageCropped=imageTest[yCrop:yCrop+sizeCrop, xCrop:xCrop+sizeCrop, :]
                if(iouValue>=65 and positives<positivesPerBox):
                    #imShowRectangles(imageCropped, croppedBoxes, color=(100,50,255))
                    offsets=np.array(getOffsets(cropImage, groundTruthBox))/sizeCrop #get the offset of the box with respect to the crop and normalize coordinates to percentage of sizeCrop
                    imageCropped=cv2.resize(imageCropped, (size,size))
                    data=np.array([imageCropped, np.array([1, 0]), offsets])
                    positivesList.append(data)
                    positives+=1
                elif(iouValue>=40 and partials<partialsPerBox):
                    #imShowRectangles(imageCropped, croppedBoxes, color=(100,50,255))
                    offsets=np.asarray(getOffsets(cropImage, groundTruthBox))/sizeCrop #get the offset of the box with respect to the crop and normalize coordinates to percentage of sizeCrop
                    imageCropped=cv2.resize(imageCropped, (size,size))
                    data=np.array([imageCropped, np.array([1, 0]), offsets])
                    partialsList.append(data)
                    partials+=1

    print("Generating numpy files...")
    negativesAsNpy=np.array(negativesList)
    np.save(numpyFolder+'negativesDataset'+str(size)+'.npy', negativesAsNpy)
    positivesAsNpy=np.array(positivesList)
    np.save(numpyFolder+'positivesDataset'+str(size)+'.npy', positivesAsNpy)
    partialsAsNpy=np.array(partialsList)
    np.save(numpyFolder+'partialsDataset'+str(size)+'.npy', partialsAsNpy)
    print("Successfully processsed\nNegatives: %d\nPositives: %d\nPartials: %d" % (len(negativesAsNpy), len(positivesAsNpy), len(partialsAsNpy)))
            

def getOffsets(cropImage, box):
    xCrop, yCrop, sizeCrop = cropImage
    xBox, yBox, widthBox, heightBox = box
    widthBoxCropped=min(xCrop+sizeCrop,xBox+widthBox) - max(xCrop, xBox)
    heightBoxCropped=min(yCrop+sizeCrop, yBox+heightBox) - max(yCrop, yBox)
    xOffsetStart=max(0, xBox-xCrop)
    yOffsetStart=max(0,yBox-yCrop)
    xOffsetEnd=xOffsetStart+widthBoxCropped
    yOffsetEnd=yOffsetStart+heightBoxCropped
    offsets=[xOffsetStart, yOffsetStart, xOffsetEnd, yOffsetEnd]
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