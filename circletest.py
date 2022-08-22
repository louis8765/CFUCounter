# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:22:15 2021

@author: Louis
"""


import numpy as np
import cv2 as cv
img = cv.imread('pos.jpg',0)
img=cv.resize(img,None,fx=.25,fy=.25)
cv.imshow('img',img)
# =============================================================================
# img = cv.medianBlur(img,5)
# cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
# circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1.2,100)
# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # draw the outer circle
#     cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
# cv.imshow('detected circles',cimg)
# cv.waitKey(0)
# cv.destroyAllWindows()
# =============================================================================


ret,thresh=cv.threshold(img,100,255,cv.THRESH_BINARY)
cv.imshow('thresh',thresh)
cv.waitKey(0)
cv.destroyAllWindows()