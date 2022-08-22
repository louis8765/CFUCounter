# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 21:42:09 2021

@author: Louis
"""
import cv2
import numpy as np

global colorHSV

def colorSelector(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        colorHSV=cv2.cvtColor(np.array([[img[x][y]]]), cv2.COLOR_BGR2HSV)
        font=cv2.FONT_HERSHEY_SIMPLEX
        strColor=str(colorHSV)
        cv2.putText(img, strColor, (x,y), font, 1, (255,255,0),2)
        cv2.imshow('image',img)
        print(img[x][y])
        print(colorHSV)
        

img = cv2.imread('exp5.jpg',1)
img=cv2.resize(img,None,fx=.25,fy=.25)
blur=cv2.GaussianBlur(img, (1,1), cv2.BORDER_DEFAULT)
hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)


cv2.imshow('image',img)
cv2.setMouseCallback('image', colorSelector)


lowerRange=np.array([0,0,0])
upperRange=np.array([255,255,80])
mask=cv2.inRange(hsv,lowerRange,upperRange)
result=cv2.bitwise_and(img,img, mask=mask)


cv2.imshow('blurred',blur)
cv2.imshow('masked',result)
cv2.waitKey(0)
cv2.destroyAllWindows()

