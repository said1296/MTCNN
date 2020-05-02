import tensorflow as tf
import cv2
from numpy.random import randint as random
import os
import numpy as np
from dataset.utils import drawRectangle, imShowRectangles
from predict import getScales
from math import ceil, floor

def predictClassification(positiveFlag=False, threshold=0.6, negativeFlag=False, partialFlag=False, boxRegressionFlag=True, numpyDir=os.getcwd()+"/dataset/numpy/", modelsDir=os.getcwd()+"/models/",network='rnet' , size=24):

    negativeDataset=np.load(numpyDir+'negativesDataset'+str(size)+'.npy', allow_pickle=True)
    positiveDataset=np.load(numpyDir+'positivesDataset'+str(size)+'.npy', allow_pickle=True)
    partialDataset=np.load(numpyDir+'partialsDataset'+str(size)+'.npy', allow_pickle=True)

    while(True and 1==0):
        if(negativeFlag):
            index = random(0, len(negativeDataset))
            image = negativeDataset[index][0]
            imagePredict = np.expand_dims((image-127.5)*(1/128),0) #Normalize and expand 1D

            if(network=='pnet'):
                modelPath=modelsDir+"pnetModel.h5"
            elif(network=='rnet'):
                modelPath=modelsDir+"rnetModel.h5"
            pnetModel=tf.keras.models.load_model(modelPath)
            prediction=pnetModel.predict(imagePredict)
            print(prediction)
            classification=prediction[0]
            image=cv2.resize(image, (500,500))
            if(boxRegressionFlag and classification[0][0]>threshold or 1==1):
                boxOffsets=np.floor(prediction[1][0]*500).astype(int)
                drawRectangle(image, boxOffsets, label= 'Predicted', color=(200,40,29),coords=True)
            cv2.putText(image,"Expected: 0", (20,460), cv2.FONT_HERSHEY_COMPLEX, 0.8, (20,150,100), 1)
            cv2.putText(image,"Predicted: "+str(classification[0][0]), (20,420), cv2.FONT_HERSHEY_COMPLEX, 0.8, (20,40,200), 1)
            cv2.imshow("Prediction", image)
            keyPressed=cv2.waitKey(0)
            if(keyPressed==27):
                break

        if(positiveFlag):
            index = random(0, len(positiveDataset))
            # image = positiveDataset[index][0]
            # imagePredict = np.expand_dims((image-127.5)*(1/128),0) #Normalize and expand 1D
            image = cv2.imread('cara.jpg')
            image = cv2.resize(image, (24,24))
            imagePredict=np.expand_dims((image-127.5)*(1/128), 0)
            if(network=='pnet'):
                modelPath=modelsDir+"pnetModel.h5"
            elif(network=='rnet'):
                modelPath=modelsDir+'rnetModel.h5'
            pnetModel=tf.keras.models.load_model(modelPath)
            prediction=pnetModel.predict(imagePredict)
            classification=prediction[0]
            image=cv2.resize(image, (500,500))
            if(boxRegressionFlag and classification[0][0]>threshold or 1==1):
                print("Predicted: ", prediction[1][0][0])
                print("Expected: ", positiveDataset[index][2])
                boxOffsets=np.floor(prediction[1][0]*500).astype(int)
                drawRectangle(image, (positiveDataset[index][2]*500).astype(int), label= 'Expected', color=(200,150,20), coords=True)
                drawRectangle(image, boxOffsets, label= 'Predicted', color=(100,40,200),coords=True)
            cv2.putText(image,"Expected: 1", (20,460), cv2.FONT_HERSHEY_COMPLEX, 0.8, (20,150,100), 1)
            cv2.putText(image,"Predicted: "+str(classification[0][0]), (20,420), cv2.FONT_HERSHEY_COMPLEX, 0.8, (20,40,200), 1)
            cv2.imshow("Prediction", image)
            keyPressed=cv2.waitKey(0)
            if(keyPressed==27):
                break

    if(1==1):
        kernelSize=12
        threshold=0.7
        image = cv2.imread('cara.jpg')
        imageNormalized=(image-127.5)*(1/128)
        modelPath=modelsDir+"pnetModel.h5"
        pnetModel=tf.keras.models.load_model(modelPath)
        imageWidth, imageHeight, _ = image.shape
        scales=getScales(0.7, kernelSize, 50, imageWidth, imageHeight)
        toPredict=[]
        layers=[]
        boxesPass=[]
        for scale in scales:
            imageScaled=cv2.resize(imageNormalized, (round(imageHeight*scale), round(imageWidth*scale)))
            imagePredict=np.expand_dims(imageScaled, 0)
            prediction=pnetModel.predict(imagePredict)
            classifications=prediction[1][0]
            boxes=prediction[0]
            boxes=boxes.reshape((-1 , 4))
            for index, box in enumerate(boxes):
                if(classifications[index]>threshold):
                    boxesPass.append(box)
            print(boxes.shape, imageScaled.shape)
            imShowRectangles(image, boxesPass, )

predictClassification()