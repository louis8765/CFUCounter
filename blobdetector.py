# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 10:05:58 2021

@author: Louis
"""

import cv2
import numpy as np
import matplotlib as plt
    
def nothing(x):
    pass

def backgroundRemove(image):
    img=cv2.imread(image)
    imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    sizes=[cv2.contourArea(x) for x in contours]
    index=sizes.index(max(sizes))
    cnt=contours[index]


    black=np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(black, [max(contours,key=cv2.contourArea)], 0, 255, -1)
    ret,black=cv2.threshold(black,0,255,cv2.THRESH_BINARY)
    final=cv2.bitwise_and(img,img,mask=black)
    return final

#read in image
img=backgroundRemove('exp1.jpg')
img=cv2.resize(img,None,fx=.35,fy=.35)
img2=backgroundRemove('exp1.jpg')
img2=cv2.resize(img2,None,fx=.35,fy=.35)
img=cv2.GaussianBlur(img, (3,3),0)
img2=cv2.GaussianBlur(img2, (3,3),0)
cv2.imshow('final',img)

#create window and trackbar
cv2.namedWindow('circularity')
cv2.createTrackbar('minDistBetweenBlobs', 'circularity', 0, 10, nothing)
cv2.createTrackbar('minTh','circularity',31,100,nothing)
cv2.createTrackbar('maxTh','circularity',200,256,nothing)
cv2.createTrackbar('minArea','circularity',5,30,nothing)
cv2.createTrackbar('maxArea','circularity',78,200,nothing)
# =============================================================================
# cv2.createTrackbar('minCircularity','circularity',5,10,nothing)
# cv2.createTrackbar('minConvexity','circularity',.87,1,nothing)
# cv2.createTrackbar('minInertia','circularity',0.1,1,nothing)
# =============================================================================


cv2.imshow('gray',img)
while(1):
    #set simpleblobdetector parameters
    params=cv2.SimpleBlobDetector_Params()

    params.minThreshold=0
    params.maxThreshold=200

    params.filterByColor=1
    params.blobColor=255

    params.filterByArea=1
    params.minArea=10
    params.maxArea=30

    params.filterByCircularity=1
    params.minCircularity=.7

    params.filterByConvexity=1
    params.minConvexity=.9

    params.filterByInertia=0
    params.minInertiaRatio=.4

    #set parameters of interest to trackbar position
    params.minDistBetweenBlobs=cv2.getTrackbarPos('minDistBetweenBlobs', 'circularity')
    params.minThreshold=cv2.getTrackbarPos('minTh', 'circularity')
    params.minArea=cv2.getTrackbarPos('minArea', 'circularity')
    params.maxArea=cv2.getTrackbarPos('maxArea', 'circularity')
# =============================================================================
#     params.maxCircularity=cv2.getTrackbarPos('minCircularity', 'circularity')
#     params.minConvexity=cv2.getTrackbarPos('minConvexity', 'circularity')
#     params.minInertiaRatio=cv2.getTrackbarPos('minInertia', 'circularity')
#     
# =============================================================================
    
    #detect blobs using parameters of interest
    
    detector=cv2.SimpleBlobDetector_create(params)
    keypoints=detector.detect(img)
    keyimg=cv2.drawKeypoints(img2, keypoints, np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
    
    cv2.imshow('keyimglight',keyimg)
# =============================================================================
#     print(len(keypoints))
# =============================================================================

    params.blobColor=0
    detector=cv2.SimpleBlobDetector_create(params)
    keypoints2=detector.detect(img)
    keyimg2=cv2.drawKeypoints(img2, keypoints2, np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('keyimgdark',keyimg2)
    
    k=cv2.waitKey(1) & 0xFF
    if k==27:
        break
    elif k==113:
        print(len(keypoints))
        cv2.putText(keyimg, 'Colony count: '+str(len(keypoints)), (550,700), cv2.FONT_HERSHEY_COMPLEX, .3, (0,0,0))
        cv2.putText(keyimg2, 'Colony count: '+str(len(keypoints2)), (550,700), cv2.FONT_HERSHEY_COMPLEX, .3, (0,0,0))
        cv2.imwrite('blue.jpg',keyimg2)
        cv2.imwrite('white.jpg',keyimg)
        print(len(keypoints2))
    
    


cv2.destroyAllWindows()

def colorRange(minColor,maxColor):
    colorRange=np.arange(minColor,maxColor+1)
    imgdict={}
    for i in colorRange:
        keyimg=[]
        params.blobColor=i
        detector=cv2.SimpleBlobDetector_create(params)
        keypoints=detector.detect(img)
        
        keyimg=cv2.drawKeypoints(img2,keypoints,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if len(keypoints)>0:
            imgdict[keyimg]=keypoints
    return imgdict
