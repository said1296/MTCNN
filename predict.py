import tensorflow as tf
import cv2
from numpy.random import randint as random
import os
import numpy as np

def predictClassification(positiveFlag=False, negativeFlag=False, partialFlag=False, trainDir=os.getcwd()+"/dataset/train/", modelsDir=os.getcwd()+"/models/", pnetFlag=False):
    positiveImagesFolder    =   trainDir+"positiveImages"
    negativeImagesFolder    =   trainDir+"negativeImages"
    partialImagesFolder     =   trainDir+"partialImages"

    if(positiveFlag):
        positiveImagesFile=open(os.path.join(positiveImagesFolder, "positiveImages.txt"), 'r')
        positivePaths=[]
        for index,record in enumerate(positiveImagesFile):
            record=record.strip().split(' ')
            positivePaths.append(record[0])

    if(negativeFlag):
        negativeImagesFile=open(os.path.join(negativeImagesFolder, "negativeImages.txt"),'r')
        negativePaths=[]
        for index,record in enumerate(negativeImagesFile):
            record=record.strip().split(' ')
            negativePaths.append(record[0])
    
    if(partialFlag):
        negativeImagesFile=open(os.path.join(negativeImagesFolder, "negativeImages.txt"),'r')
        negativePaths=[]
        for index,record in enumerate(negativeImagesFile):
            record=record.strip().split(' ')

    while(True):
        if(positiveFlag):
            index = random(0, len(positivePaths))
            image = cv2.imread(positivePaths[index])
            imagePredict = np.expand_dims(image,0)

            if(pnetFlag==True):
                modelPath=modelsDir+"pnetModel.h5"
                pnetModel=tf.keras.models.load_model(modelPath)
                prediction=pnetModel.predict(imagePredict)
                image=cv2.resize(image, (500,500))
                cv2.putText(image,str(prediction[0][0]), (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (20,40,200), 2)
                cv2.imshow("Prediction", image)
                keyPressed=cv2.waitKey(0)
                if(keyPressed==27):
                    break


        if(negativeFlag):
            index = random(0, len(negativePaths))
            image = cv2.imread(negativePaths[index])
            imagePredict = np.expand_dims(image,0)

            if(pnetFlag==True):
                modelPath=modelsDir+"pnetModel.h5"
                pnetModel=tf.keras.models.load_model(modelPath)
                prediction=pnetModel.predict(imagePredict)
                image=cv2.resize(image, (500,500))
                cv2.putText(image,str(prediction[0][0]), (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (20,40,200), 2)
                cv2.imshow("Prediction", image)
                keyPressed=cv2.waitKey(0)
                if(keyPressed==27):
                    break

        if(partialFlag):
            pass

predictClassification(positiveFlag=True, negativeFlag=True, pnetFlag=True)