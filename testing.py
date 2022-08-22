# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:24:09 2022

@author: Louis
"""

import cv2 as cv
import numpy as np



img=cv.imread('scene2.jpg')
img=cv.resize(img,None,fx=.5,fy=.5)
lab=cv.cvtColor(img,cv.COLOR_BGR2LAB)
# =============================================================================
# final=np.zeros(img.shape[:2],np.uint8)
# for i in range(3):
#     cv.imshow(str(i),img[:,:,i])
#     if i==0:
#         final=cv.Canny(img[:,:,i],5,10)
#     else:
#         final=cv.bitwise_and(final,final,mask=cv.Canny(img[:,:,i],5,30))
# 
# cv.imshow('combiuned',final)
# =============================================================================
# =============================================================================
# img=cv.GaussianBlur(img,(3,3),0)
# final=0
# for i in range(3):
#     lap=cv.Laplacian(img[:,:,i],cv.CV_16S,3)
#     abs_dst = cv.convertScaleAbs(lap)
#     
#     ret,thresh=cv.threshold(abs_dst, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
#     if i==0:
#         final=thresh
#     else:
#         final=cv.bitwise_and(final,final,mask=thresh)
# cv.imshow('final',final)
# 
# =============================================================================
clahe=cv.createCLAHE()
final=0
# =============================================================================
# for i in range(3):
# # =============================================================================
# #     cv.imshow(str(i),clahe.apply(img[:,:,i]))
# # =============================================================================
#     clade=clahe.apply(img[:,:,i])
#     ret,thresh=cv.threshold(clade,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
#     cv.imshow(str(i),clade)
# =============================================================================
cv.imshow('lightness',cv.cvtColor(img,cv.COLOR_BGR2GRAY))

lab[:,:,0]=clahe.apply(lab[:,:,0])
bgr=cv.cvtColor(lab,cv.COLOR_LAB2BGR)
cv.imshow('new',bgr)



cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)