import tensorflow as tf
import cv2
from numpy.random import randint as random
import os, sys
import numpy as np
from dataset.utils import drawRectangle, imShowRectangles, iou
from math import floor, ceil
import time

def predict(threshold=0.9, nmsThreshold=0.1, modelDir=os.getcwd()+"/models/", kernelSize=24, stride=4, imagePath=os.getcwd()+'/predict.jpg', minFace=25, scaleFactor=0.709):
    startTime=time.perf_counter()
    modelPath=modelDir+"pnetModel.h5"
    pnetModel=tf.keras.models.load_model(modelPath)

    image=cv2.imread(imagePath)
    imageNormalized=(image-127.5)*(1/128)
    imageWidth, imageHeight, imageChannels = image.shape
    if(imageChannels!=3):
        print("IMAGE FORMAT NOT SUPPORTED")
        return

    scales=getScales(scaleFactor, kernelSize, minFace, imageWidth, imageHeight)
    print(scales)

    toPredict=[]
    layers=[]
    for scale in scales:
        print(scale)
        imageScaled=cv2.resize(imageNormalized, (round(imageHeight*scale), round(imageWidth*scale)))
        imageScaledHeight, imageScaledWidth, _ = imageScaled.shape
        paddingRight=(kernelSize*ceil(imageScaledWidth/kernelSize))-imageScaledWidth
        paddingBottom=(kernelSize*ceil(imageScaledHeight/kernelSize))-imageScaledHeight
        imageScaled = np.pad(imageScaled, ((0, paddingRight), (0, paddingBottom), (0, 0)), 'constant', constant_values=(0, 0))
        imageScaledHeight, imageScaledWidth, _ = imageScaled.shape
        xCrops=list(range(0, imageScaledWidth-kernelSize+1, stride))
        yCrops=list(range(0, imageScaledHeight-kernelSize+1, stride))
        for yCrop in yCrops:    #feed crops of kernelSize*kernelSize to the network
            for xCrop in xCrops:
                imageCropped=imageScaled[yCrop:yCrop+kernelSize, xCrop:xCrop+kernelSize, :]
                toPredict.append(imageCropped)
        layers.append([xCrops, yCrops])

    toPredict=np.array(toPredict)
    print("AAAAAAA", sys.getsizeof(toPredict))
    prediction=pnetModel.predict(toPredict)
    print("prediction finished")

    boxes=[]
    classifications=[]
    predictionCount=0
    for layerIndex, layer in enumerate(layers):
        for yCrop in layer[1]:    #feed crops of kernelSize*kernelSize to the network
            for xCrop in layer[0]:
                classification=prediction[0][predictionCount][0]
                if(classification>threshold):
                    boxOffsets=prediction[1][predictionCount]*kernelSize
                    xStart=floor((xCrop+(boxOffsets[0]))/scales[layerIndex])
                    yStart=floor((yCrop+(boxOffsets[1]))/scales[layerIndex])
                    xEnd=ceil((xCrop+(boxOffsets[2]))/scales[layerIndex])
                    yEnd=ceil((yCrop+(boxOffsets[3]))/scales[layerIndex])
                    offsets=[xStart, yStart, xEnd, yEnd]
                    boxes.append(offsets)
                    classifications.append(classification)
                predictionCount+=1
    print("FINISHED BOXES")
    boxes=nonMaximumSupression(boxes, classifications, nmsThreshold)
    endTime=time.perf_counter()
    totalTime=endTime-startTime
    print("TOTAL PREDICTION TIME: ", totalTime)
    imShowRectangles(image, [boxes], coords=True, thickness=1)  
    return

def nonMaximumSupression(boxes, classifications, threshold):
    purgedBoxes=[]
    boxes=np.array(boxes)
    while(len(boxes)>0):
        maxIndex=np.argmax(classifications)
        purgedBoxes.append(boxes[maxIndex])
        boxes=np.delete(boxes, maxIndex, axis=0)
        classifications=np.delete(classifications, maxIndex, axis=0)
        indexesSupression=iou(purgedBoxes[-1], boxes, threshold=threshold, coords=True)
        boxes=np.delete(boxes, indexesSupression, axis=0)
        classifications=np.delete(classifications, indexesSupression, axis=0)
    print(len(purgedBoxes))
    return purgedBoxes
        

def getScales(scaleFactor, kernelSize, minFace, imageWidth, imageHeight):
    startScale=kernelSize/minFace #at this scale the network is able to detect faces of size minFace, next layers will be smaller and detect bigger faces
    smallerSide=min(imageWidth, imageHeight)

    scale=startScale
    scale=(smallerSide*scale-((smallerSide*scale)%1))/smallerSide
    scaledSide=smallerSide*scale
    scales=[]
    factorCount=1
    while scaledSide>kernelSize:  #the smallest side can't be smaller than the kernelSize
        scales.append(scale)
        scale=startScale*(scaleFactor**factorCount)
        scale=(smallerSide*scale-((smallerSide*scale)%1))/smallerSide #find nearest multiple in order to make box regression easier
        scaledSide=smallerSide*scale
        factorCount+=1

    return scales