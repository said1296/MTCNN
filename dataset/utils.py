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

def iou(predicted, groundTruthBoxes, log=False, threshold=False, coords=False):
    ious=[]
    indexes=[]
    if(coords):
        xPred, yPred, xEndPred, yEndPred = predicted
        widthPred=xEndPred-xPred
        heightPred=yEndPred-yPred
    else:
        try:
            xPred, yPred, widthPred, heightPred = predicted
        except:
            xPred, yPred, widthPred = predicted
            heightPred = widthPred
    for index, groundTruthBox in enumerate(groundTruthBoxes):
        if(coords):
            xGround, yGround, xEndGround, yEndGround = groundTruthBox
            widthGround=xEndGround-xGround
            heightGround=yEndGround-yGround
        else:
            xGround, yGround, widthGround, heightGround = groundTruthBox
        xOverlap=max(0, min(xPred+widthPred,xGround+widthGround) - max(xPred, xGround))
        yOverlap=max(0, min(yPred+heightPred, yGround+heightGround) - max(yPred, yGround))
        areaOverlap=xOverlap*yOverlap
        areaPred=widthPred*heightPred
        areaGround=widthGround*heightGround
        iou=(areaOverlap)/((areaPred+areaGround)-areaOverlap)
        ious.append(iou)
        if(threshold and (areaOverlap>threshold*areaGround) and areaOverlap!=0):
            indexes.append(index)
    if(threshold):
        return indexes
    iou=max(ious)
    if(log):
        print("IoU: %f" % iou)
    return iou*100
    

def imShowRectangles(image, boxes, windowName="Drawn image", labels=False, color=(255,255,255), thickness=2, coords=False, wait=True):
    image = image.copy()
    cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 600,600)
    for index, box in enumerate(boxes):
        label=''
        if(isinstance(box[index],np.ndarray) or isinstance(box[index], list)):
            for nestedIndex, nestedBox in enumerate(box):
                nestedBox = list(map(int, nestedBox))
                if(labels):
                    if(isinstance(labels[index], list) or isinstance(labels[index], np.ndarray)):
                        label=labels[index][nestedIndex]
                    else:
                        label=labels[index]
                imageDrawn=drawRectangle(image, nestedBox, label, color, thickness, coords)
        else:
            box = list(map(int, box))
            if(labels):
                label=labels[index]
            imageDrawn=drawRectangle(image,box, label, color, thickness, coords)
    cv2.imshow(windowName, imageDrawn)
    if(wait):
        keyPressed=cv2.waitKey(0)
        if(keyPressed==27):
            cv2.destroyAllWindows()

def drawRectangle(image, box, label=False, color=(255,255,255), thickness=2, coords=False):
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
        if not isinstance(label, str):
            label=str(label)
        cv2.putText(image, label, (startX+6*thickness, startY+30*thickness), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
    return cv2.rectangle(image, startPoint, endPoint, color, thickness)