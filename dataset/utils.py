import os, sys
from cv2 import cv2
import numpy as np

def imagesLister(dir):
    imagesList = []
    listElems = os.listdir(dir)
    for elem in listElems:
        fullPath = os.path.join(dir, elem)
        if(os.path.isdir(fullPath)):
            return imagesLister(fullPath)
        else:
            imagesList.append(fullPath)
    return imagesList

def imagesResizer(imagesList, size):
    resizedImages=[]
    for image in imagesList:
        npImage = cv2.imread(image)
        npImageResized = cv2.resize(npImage, (size,size))
        resizedImages.append(npImageResized)
    return resizedImages

def iou(predicted, groundTruthBoxes, log=False):
    ious=[]
    try:
        xPred, yPred, widthPred, heightPred = predicted
    except:
        xPred, yPred, widthPred = predicted
        heightPred = widthPred
    for groundTruthBox in groundTruthBoxes:
        xGround, yGround, widthGround, heightGround = groundTruthBox
        xOverlap=max(0, min(xPred+widthPred,xGround+widthGround) - max(xPred, xGround))
        yOverlap=max(0, min(yPred+heightPred, yGround+heightGround) - max(yPred, yGround))
        areaOverlap=xOverlap*yOverlap
        areaPred=widthPred*heightPred
        areaGround=widthGround*heightGround
        iou=(areaOverlap*100)/((areaPred+areaGround)-areaOverlap)
        ious.append(iou)
    iou=max(ious)
    if(log):
        print("IoU: %f" % iou)
    return iou
    

def imShowRectangles(image, boxes, windowName="Drawn image", labels=False, color=(255,255,255), thickness=2, coords=False):
    image = image.copy()
    cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 900,900)
    for index, box in enumerate(boxes):
        label=''
        if(isinstance(box[index],np.ndarray)):
            for nestedBox in box:
                nestedBox = list(map(int, nestedBox))
                if(labels):
                    label=labels[index]
                imageDrawn=drawRectangle(image, nestedBox, label, color, thickness, coords)
        else:
            box = list(map(int, box))
            if(labels):
                label=labels[index]
            imageDrawn=drawRectangle(image,box, label, color, thickness, coords)
    cv2.imshow(windowName, imageDrawn)
    cv2.waitKey(0)

def drawRectangle(image, box, label, color=(255,255,255), thickness=2, coords=False):
    if(coords):
        startX, startY, endX, endY = box
    else:
        startX, startY, width=box[:3]
        try:
            height=box[3]
        except:
            height=width
        endX=startX+width
        endY=startY+height
    startPoint=(startX,startY)
    endPoint=(endX-1, endY-1) #minus one to compensate
    if(label):
        cv2.putText(image, label, (startX+6*thickness, startY+15*thickness), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1)
    return cv2.rectangle(image, startPoint, endPoint, color, thickness)