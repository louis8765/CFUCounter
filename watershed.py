# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 12:27:27 2021

@author: Louis
"""

import cv2
import numpy as np

img=cv2.imread('fourth.jpg')
img=cv2.resize(img,None,fx=.35,fy=.35)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
ret,thresh=cv2.threshold(gray,135,255,cv2.THRESH_BINARY_INV)

cv2.imshow('thresh',thresh)

kernel=np.ones((3,3))
notNoise=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=1)

sure_bg=cv2.dilate(notNoise,kernel,iterations=3)

dist_transform = cv2.distanceTransform(notNoise,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
sure_fg=np.uint8(sure_fg)
unknown=cv2.subtract(sure_bg,sure_fg)


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

cv2.imshow('final',img)

cv2.waitKey(0)
cv2.destroyAllWindows()