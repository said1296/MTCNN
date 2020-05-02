import tensorflow as tf
import cv2
from numpy.random import randint as random
import os, sys
import numpy as np
from dataset.utils import drawRectangle, imShowRectangles, iou
from math import floor, ceil
import time

modelDir=os.getcwd()+"/models/"

def predictWebcam(threshold=0.95, nmsThreshold=0.8, minNmsBoxes=2, kernelSize=12, stride=4, imagePath=os.getcwd()+'/predict.jpg', minFace=50, scaleFactor=0.6):
    modelPath=modelDir+"pnetModel.h5"
    pnetModel=tf.keras.models.load_model(modelPath)

    cam=cv2.VideoCapture(0)
    firstPassFlag=0
    while True:
        startTime=time.perf_counter()
        image=cam.read()[1]
        imageNormalized=(image-127.5)*(1/128)

        if(firstPassFlag==0):
            imageWidth, imageHeight, imageChannels = image.shape
            if(imageChannels!=3):
                print("IMAGE FORMAT NOT SUPPORTED")
                return
            scales=getScales(scaleFactor, kernelSize, minFace, imageWidth, imageHeight)

        toPredict=[]
        layers=[]
        for scale in scales:
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
        prediction=pnetModel.predict(toPredict)

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
        if(len(boxes)>0):
            imShowRectangles(image, [boxes], windowName='RAW', coords=True, thickness=1, wait=False)
        boxes, probabilities=nonMaximumSupression(boxes, classifications, nmsThreshold, minNmsBoxes)
        if(len(boxes)>0):
            imShowRectangles(image, [boxes], windowName='NMS', coords=True, thickness=1, wait=False)
        boxes=joinSupression(boxes)
        if(len(boxes)>0):
            imShowRectangles(image, [boxes], windowName='Join Suppresion', coords=True, thickness=1, wait=False)
        boxes, _=densitySupression(boxes, probabilities)
        if(len(boxes)>0):
            imShowRectangles(image, [boxes], windowName='Density', coords=True, thickness=1, wait=False)
        # if(len(boxes)>0):
        #     imShowRectangles(image, [boxes], windowName='Join supression', coords=True, thickness=1, wait=False)
        # else:
        #     cv2.imshow('Join supression', image)
        #     cv2.waitKey(5)

        #boxes, probabilities=rnetStage(image, boxes)

        endTime=time.perf_counter()
        totalTime=endTime-startTime
        pressedKey=cv2.waitKey(5)
        print("TOTAL PREDICTION TIME: ", totalTime)
        # if(len(boxes)>0):
        #     imShowRectangles(image, [boxes], windowName='MTCNN', coords=True, thickness=1, wait=False)
        #     pressedKey=cv2.waitKey(5)
        # else:
        #     cv2.imshow('MTCNN', image)
        #     pressedKey=cv2.waitKey(5)
        if(pressedKey==32):
            pressedKey=cv2.waitKey(0)
        if(pressedKey==27):
            return

        if(firstPassFlag==0):
            firstPassFlag=1

def rnetStage(image, boxes, rnetThreshold=0.8):
    modelPath=modelDir+"rnetModel.h5"
    rnetModel=tf.keras.models.load_model(modelPath)
    resizedBoxes=[]
    boxSizes=[]
    refinedBoxes=[]
    refinedProbabilities=[]
    if(len(boxes)>0):
        for box in boxes:
            xStart, yStart, xEnd, yEnd = box
            widthBox=xEnd-xStart
            heightBox=yEnd-yStart
            boxSize=max(widthBox, heightBox)
            boxSizes.append(boxSize)
            xEndNew=xStart+boxSize
            yEndNew=yStart+boxSize
            croppedImage=image[yStart:yEndNew, xStart:xEndNew, :]
            resizedBox=cv2.resize(croppedImage, (24, 24))
            resizedBox=(resizedBox-127.5)*(1/128)
            resizedBoxes.append(resizedBox)
        resizedBoxes=np.asarray(resizedBoxes)
        predictions=rnetModel.predict(resizedBoxes)
        for index, box in enumerate(boxes):
            croppedImage=image[box[1]:box[1]+boxSizes[index], box[0]:box[0]+boxSizes[index], :]
            cv2.imwrite('cara.jpg', croppedImage)
            xStartBoxRefined, yStartBoxRefined, xEndBoxRefined, yEndBoxRefined=(predictions[1][index]/24)*boxSizes[index]
            print(box)
            print(predictions[1][index]/24)
            print(xStartBoxRefined, yStartBoxRefined, xEndBoxRefined, yEndBoxRefined)
            refinedBox=[xStartBoxRefined, yStartBoxRefined, xEndBoxRefined, yEndBoxRefined]
            if(predictions[0][index][0]>rnetThreshold):
                xStartBoxRefined, yStartBoxRefined, xEndBoxRefined, yEndBoxRefined=(predictions[1][index]/24)*boxSizes[index]
                refinedBox=[box[0]+xStartBoxRefined, box[1]+yStartBoxRefined, box[2]+xEndBoxRefined, box[3]+yEndBoxRefined]
                refinedBoxes.append(refinedBox)
                refinedProbabilities.append(predictions[0][index][0])
    return refinedBoxes, refinedProbabilities

def joinSupression(boxes, threshold=0.000001):
    joinedBoxes=[]
    areas=[]
    for box in boxes:
        area=(box[2]-box[0])*(box[3]-box[1])
        areas.append(area)
    while len(boxes)>0:
        maxAreaIndex=areas.index(max(areas))
        maxArea=areas[maxAreaIndex]
        overlapIndexes=iou(boxes[maxAreaIndex], boxes, threshold=threshold, coords=True)
        xStarts=[]
        yStarts=[]
        xEnds=[]
        yEnds=[]
        for overlapIndex in overlapIndexes:
            xStart, yStart, xEnd, yEnd=boxes[overlapIndex]
            xStarts.append(xStart), yStarts.append(yStart), xEnds.append(xEnd), yEnds.append(yEnd)
        joinedBox=[min(xStarts), min(yStarts), max(xEnds), max(yEnds)]
        joinedBoxes.append(joinedBox)
        for x in range(len(overlapIndexes)):
            maxIndexIndex=overlapIndexes.index(max(overlapIndexes))
            maxIndex=max(overlapIndexes)
            del boxes[maxIndex]
            del areas[maxIndex]
            del overlapIndexes[maxIndexIndex]
    return joinedBoxes


def densitySupression(boxes, probabilities):
    lenBoxes=len(boxes)
    mostDenseBoxes=[]
    mostDenseProbabilities=[]
    areas=[]
    for box in boxes:
        area=(box[2]-box[0])*(box[3]-box[1])
        areas.append(area)
    while len(boxes)>0:
        maxAreaIndex=areas.index(max(areas))
        maxArea=areas[maxAreaIndex]
        overlapIndexes=iou(boxes[maxAreaIndex], boxes, threshold=0.1, coords=True)
        if(len(overlapIndexes)==0):
            mostDenseBoxes.append(boxes[maxAreaIndex])
            mostDenseProbabilities.append(probabilities[maxAreaIndex])
            del boxes[maxAreaIndex]
            del areas[maxAreaIndex]
            del probabilities[maxAreaIndex]
        elif(len(overlapIndexes)==2):
            mostDenseBoxes.append(boxes[maxAreaIndex])
            mostDenseProbabilities.append(probabilities[maxAreaIndex])
            for x in range(len(overlapIndexes)):
                maxIndexIndex=overlapIndexes.index(max(overlapIndexes))
                maxIndex=max(overlapIndexes)
                del boxes[maxIndex]
                del probabilities[maxIndex]
                del areas[maxIndex]
                del overlapIndexes[maxIndexIndex]
        else:
            densities=[]
            for index, overlapIndex in enumerate(overlapIndexes):
                densitySum=0
                bigBox=boxes[overlapIndexes[index]]
                for x in range(len(overlapIndexes)-index-1):
                    testIndex=overlapIndexes[index+1+x]
                    testBox=boxes[testIndex]
                    xOverlap=max(0, min(bigBox[2], testBox[2]) - max(bigBox[0], testBox[0]))
                    yOverlap=max(0, min(bigBox[3], testBox[3]) - max(bigBox[1], testBox[1]))
                    areaOverlap=xOverlap*yOverlap
                    testBoxArea=(testBox[2]-testBox[0])*(testBox[3]-testBox[1])
                    bigBoxArea=(bigBox[2]-bigBox[0])*(bigBox[3]-bigBox[1])
                    testBoxPenalization=2*(testBoxArea-areaOverlap)/testBoxArea
                    density=(areaOverlap)/bigBoxArea
                    densitySum+=density-testBoxPenalization
                densities.append(densitySum)
            mostDenseProbabilities.append(probabilities[overlapIndexes[densities.index(max(densities))]])
            mostDenseBoxes.append(boxes[overlapIndexes[densities.index(max(densities))]])

            for x in range(len(overlapIndexes)):
                maxIndexIndex=overlapIndexes.index(max(overlapIndexes))
                maxIndex=max(overlapIndexes)
                del boxes[maxIndex]
                del areas[maxIndex]
                del probabilities[maxIndex]
                del overlapIndexes[maxIndexIndex]
    return mostDenseBoxes, mostDenseProbabilities

def nonMaximumSupression(boxes, classifications, threshold, minNmsBoxes):
    purgedBoxes=[]
    purgedClassifications=[]
    boxes=np.array(boxes)
    while(len(boxes)>0):
        maxIndex=np.argmax(classifications)
        purgedBoxes.append(boxes[maxIndex])
        purgedClassifications.append(classifications[maxIndex])
        boxes=np.delete(boxes, maxIndex, axis=0)
        classifications=np.delete(classifications, maxIndex, axis=0)
        indexesSupression=iou(purgedBoxes[-1], boxes, threshold=threshold, coords=True)
        if(len(indexesSupression)<minNmsBoxes):
            del purgedBoxes[-1]
            del purgedClassifications[-1]
        boxes=np.delete(boxes, indexesSupression, axis=0)
        classifications=np.delete(classifications, indexesSupression, axis=0)
    return purgedBoxes, purgedClassifications
        

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