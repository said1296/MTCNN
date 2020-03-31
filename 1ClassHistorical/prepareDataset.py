from utils import imagesLister, imagesResizer, imShowRectangles, iou
import re
import os
import numpy as np
from numpy.random import randint as random
import cv2
from math import sqrt, floor, ceil

size=24
trainDir=os.getcwd()+"/dataset/train/"

def prepareDataset(parseGroundTruthsFlag=False, negativesFlag=False, positivesAndPartialsFlag=False, limit=False):
    positiveImagesFolder    =   trainDir+"positiveImages"
    negativeImagesFolder    =   trainDir+"negativeImages"
    partialImagesFolder     =   trainDir+"partialImages"

    if(negativesFlag):
        if not os.path.exists(negativeImagesFolder):
            os.mkdir(negativeImagesFolder)
        negativeImagesFile  =   open(os.path.join(negativeImagesFolder, "negativeImages.txt"),'w')
        negativeId=0

    if(positivesAndPartialsFlag):
        if not os.path.exists(positiveImagesFolder):
            os.mkdir(positiveImagesFolder)
        if not os.path.exists(partialImagesFolder):
            os.mkdir(partialImagesFolder)
        positiveImagesFile  =   open(os.path.join(positiveImagesFolder, "positiveImages.txt"), 'w')
        partialImagesFile   =   open(os.path.join(partialImagesFolder, "partialImages.txt"),'w')
        positiveId=0
        partialId=0

    if(parseGroundTruthsFlag):
        parseGroundTruths(limit=limit)

    groundTruthsParsed = open(os.path.join(trainDir,"groundTruthsParsed.txt"), 'r')
    #Parse ground truths to np arrays
    for groundTruth in groundTruthsParsed:
        groundTruthArray = groundTruth.split(' ')
        if(len(groundTruthArray)<2):
            continue
        imageTest=cv2.imread(trainDir+'images/'+groundTruthArray[0])
        heightImage, widthImage, _ = imageTest.shape
        groundTruthBoxes = np.reshape(np.array(groundTruthArray[2:], dtype='f8'),(-1,4))
        groundTruthBoxesPurged = [box for box in groundTruthBoxes if (box[2]>40 and box[3]>40 and box[2]/box[3]>0.70 and box[3]/box[2]>0.70)] #Select boxes bigger than 40px and with an aspect ratio bigger than 0.7
        if(len(groundTruthBoxesPurged)==0):
            continue
        #Extract negative crops of images for training, less than 3% of IoU
        if(negativesFlag):
            if(negativeId%100==0):
                print("Negatives: %d" % negativeId) #Print count of negatives
            negatives=0
            negativesPerImage=50
            notFoundIteration=0
            while negatives<negativesPerImage:
                notFoundIteration+=1
                sizeCrop = random(40, min(widthImage,heightImage))
                xCrop = random(0,widthImage-sizeCrop)
                yCrop = random(0, heightImage-sizeCrop)
                cropImage = [xCrop, yCrop, sizeCrop]
                croppedBoxes=getCroppedBoxes(cropImage, groundTruthBoxes)
                #imShowRectangles(imageTest, [cropImage,groundTruthBoxes], color=(100,50,255), labels=['IoU: %f' % iou(cropImage,groundTruthBoxes,True), ['Ground']])
                iouValue=iou(cropImage,groundTruthBoxes)
                if(notFoundIteration>10000):
                    #imShowRectangles(imageCropped, croppedBoxes, color=(100,50,255), windowName=trainDir+'images/'+groundTruthArray[0])
                    imShowRectangles(imageTest, [cropImage,groundTruthBoxes], color=(100,50,255), labels=['IoU: %f' % iou(cropImage,groundTruthBoxes,True), 'Ground'])
                    notFoundIteration=0
                if(iouValue<20 and negatives<negativesPerImage):
                    imageCropped=imageTest[yCrop:yCrop+sizeCrop, xCrop:xCrop+sizeCrop, :]
                    #imShowRectangles(imageCropped, [croppedBoxes], color=(100,50,255))
                    negativeImagesFile.write('\n%s/%d.jpg' % (negativeImagesFolder, negativeId))
                    for groundTruth in croppedBoxes:
                        xCroppedBox, yCroppedBox, widthCroppedBox, heightCroppedBox = groundTruth
                        xStart=xCroppedBox/sizeCrop
                        yStart=xCroppedBox/sizeCrop
                        xEnd=(xCroppedBox+widthCroppedBox)/sizeCrop 
                        yEnd=(yCroppedBox+heightCroppedBox)/sizeCrop
                        negativeImagesFile.write(' %f %f %f %f' % (xStart, yStart,xEnd, yEnd))
                    imageCropped=cv2.resize(imageCropped, (size,size))
                    cv2.imwrite(negativeImagesFolder+'/%s.jpg'%negativeId, imageCropped)
                    negatives+=1
                    negativeId+=1
        if(positivesAndPartialsFlag):
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
                    croppedBoxes=getCroppedBoxes(cropImage, groundTruthBoxes)
                    if(notFoundIteration>10000):
                        imShowRectangles(imageTest, [cropImage,groundTruthBoxes], color=(100,50,255), labels=['IoU: %f' % iou(cropImage,groundTruthBoxes,True), 'Ground'])
                        imShowRectangles(imageCropped, croppedBoxes, color=(100,50,255), windowName=trainDir+'images/'+groundTruthArray[0])
                        notFoundIteration=0
                    notFoundIteration+=1
                    iouValue=iou(cropImage,groundTruthBoxes)
                    imageCropped=imageTest[yCrop:yCrop+sizeCrop, xCrop:xCrop+sizeCrop, :]
                    if(iouValue>=65 and positives<positivesPerBox):
                        #imShowRectangles(imageCropped, croppedBoxes, color=(100,50,255))
                        positiveImagesFile.write('\n%s/%d.jpg' % (positiveImagesFolder, positiveId))
                        for groundTruth in croppedBoxes:
                            xCroppedBox, yCroppedBox, widthCroppedBox, heightCroppedBox = groundTruth
                            xStart=xCroppedBox/sizeCrop
                            yStart=xCroppedBox/sizeCrop
                            xEnd=(xCroppedBox+widthCroppedBox)/sizeCrop
                            yEnd=(yCroppedBox+heightCroppedBox)/sizeCrop
                            positiveImagesFile.write(' %f %f %f %f' % (xStart, yStart,xEnd, yEnd))
                        imageCropped=cv2.resize(imageCropped, (size,size))
                        cv2.imwrite(positiveImagesFolder+'/%s.jpg'%positiveId, imageCropped)
                        positives+=1
                        positiveId+=1
                        if(positiveId%100==0):
                            print("Positives: %d" % positiveId) #Print count of negatives
                    elif(iouValue>=40 and partials<partialsPerBox):
                        #imShowRectangles(imageCropped, croppedBoxes, color=(100,50,255))
                        partialImagesFile.write('\n%s/%d.jpg' % (partialImagesFolder, partialId))
                        for groundTruth in croppedBoxes:
                                xCroppedBox, yCroppedBox, widthCroppedBox, heightCroppedBox = groundTruth
                                xStart=xCroppedBox/sizeCrop
                                yStart=xCroppedBox/sizeCrop
                                xEnd=(xCroppedBox+widthCroppedBox)/sizeCrop
                                yEnd=(yCroppedBox+heightCroppedBox)/sizeCrop
                                partialImagesFile.write(' %f %f %f %f' % (xStart, yStart,xEnd, yEnd))
                        imageCropped=cv2.resize(imageCropped, (size,size))
                        cv2.imwrite(partialImagesFolder+'/%s.jpg'%partialId, imageCropped)
                        partials+=1
                        partialId+=1
                        if(partialId%100==0):
                            print("Partials: %d" % partialId) #Print count of negatives
            

def getCroppedBoxes(cropImage, boxes):
    xCrop, yCrop, sizeCrop = cropImage
    croppedBoxes=[]
    for box in boxes:
        xBox, yBox, widthBox, heightBox = box
        if(xCrop+sizeCrop<xBox or yCrop+sizeCrop<yBox or xBox+widthBox<xCrop or yBox+heightBox<yCrop):
            continue
        xBoxCropped=max(0, xBox-xCrop)
        yBoxCropped=max(0,yBox-yCrop)
        widthBoxCropped=min(xCrop+sizeCrop,xBox+widthBox) - max(xCrop, xBox)
        heightBoxCropped=min(yCrop+sizeCrop, yBox+heightBox) - max(yCrop, yBox)
        cropBox=[xBoxCropped,yBoxCropped,widthBoxCropped,heightBoxCropped]
        croppedBoxes.append(cropBox)
    return croppedBoxes

def parseGroundTruths(limit):
    #Read ground truths
    groundTruths = open(os.path.join(trainDir,"groundTruths.txt"), 'r')
    groundTruthsParsed = open(os.path.join(trainDir,"groundTruthsParsed.txt"), 'w')
    #Parse ground truths, a record per line, save in groundTruthsParsed.txt
    for index, groundTruth in enumerate(groundTruths):
        if(limit!=False and index==limit):
            break
        groundTruth=groundTruth.strip()
        if(re.match(r'\d{1,2}--',groundTruth)):
            groundTruthsParsed.write('\n%s' % groundTruth)
        else:
            groundTruthArray=groundTruth.split()
            if(len(groundTruthArray)>1):
                groundTruthsParsed.write(' %s' % ' '.join(groundTruthArray[:4]))
            else:
                groundTruthsParsed.write(' %s' % groundTruth)