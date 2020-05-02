import tensorflow as tf
import cv2
from numpy.random import randint as random
import os
import numpy as np
from dataset.utils import drawRectangle
from math import floor, ceil
from dataset.utils import imShowRectangles
import time

def predict(threshold=0.8, modelDir=os.getcwd()+"/models/", kernelSize=24, stride=4, imagePath=os.getcwd()+'/predict.jpg', minFace=24, scaleFactor=0.709):
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

    boxes=[]
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
        percentage=0
        for yIndex, yCrop in enumerate(yCrops):    #feed crops of kernelSize*kernelSize to the network
            if(ceil(yIndex*100/len(yCrops))>percentage):
                percentage=ceil(yIndex*100/len(yCrops))
                print("Processed: %d%%" % percentage)
            for xIndex, xCrop in enumerate(xCrops):
                imageCropped=imageScaled[yCrop:yCrop+kernelSize, xCrop:xCrop+kernelSize, :]
                imagePredict=np.expand_dims(imageCropped, 0) #normalize and add dimension as dummy to fit network (dimension is batch dimension but we predict only one image)
                prediction=pnetModel.predict(imagePredict)
                classification=prediction[0][0][0][0][0]
                if(classification>threshold):
                    boxOffsets=prediction[1][0][0][0]*kernelSize
                    xStart=floor((xCrop+(boxOffsets[0]))/scale)
                    yStart=floor((yCrop+(boxOffsets[1]))/scale)
                    xEnd=ceil((xCrop+(boxOffsets[2]))/scale)
                    yEnd=ceil((yCrop+(boxOffsets[3]))/scale)
                    offsets=[xStart, yStart, xEnd, yEnd]
                    boxes.append(offsets)
                    # imShowRectangles(imageCropped, [boxOffsets], coords=True, thickness=1)
                    # imShowRectangles(imageScaled, [[xCrop+boxOffsets[0], yCrop+boxOffsets[1], xCrop+boxOffsets[2], yCrop+boxOffsets[3]]], coords=True, thickness=1)  
                    # imShowRectangles(image, [boxes], coords=True, thickness=1)  
        endTime=time.perf_counter()
        totalTime=endTime-startTime
        print("TOTAL PREDICTION TIME: ", totalTime)
        imShowRectangles(image, [boxes], coords=True, thickness=1)  
        # cv2.waitKey(0)
        return
            
    # index = random(0, len(negativeDataset))
    # image = negativeDataset[index][0]
    # imagePredict = np.expand_dims((image-127.5)*(1/128),0) #Normalize and expand 1D

    # modelPath=modelDir+"pnetModel.h5"
    # pnetModel=tf.keras.models.load_model(modelPath)
    # prediction=pnetModel.predict(imagePredict)
    # classification=prediction[0]
    # image=cv2.resize(image, (500,500))

    # boxOffsets=np.floor(prediction[1][0]*500).astype(int)
    # drawRectangle(image, boxOffsets, label= 'Predicted', color=(200,40,29),coords=True)

    # cv2.putText(image,"Expected: 0", (20,460), cv2.FONT_HERSHEY_COMPLEX, 0.8, (20,150,100), 1)
    # cv2.putText(image,"Predicted: "+str(classification[0][0]), (20,420), cv2.FONT_HERSHEY_COMPLEX, 0.8, (20,40,200), 1)
    # cv2.imshow("Prediction", image)

def getScales(scaleFactor, kernelSize, minFace, imageWidth, imageHeight):
    startScale=kernelSize/minFace #at this scale the network is able to detect faces of size minFace, next layers will be smaller and detect bigger faces
    smallerSide=min(imageWidth, imageHeight)

    scales=[]
    scale=startScale
    scaledSide=smallerSide
    factorCount=0
    while scaledSide>kernelSize:  #the smallest side can't be smaller than the kernelSize
        scales.append(scale)
        scale=startScale*(scaleFactor**factorCount)
        scale=(smallerSide*scale-((smallerSide*scale)%1))/smallerSide #find nearest multiple in order to make box regression easier
        scaledSide=smallerSide*scale
        factorCount+=1

    return scales