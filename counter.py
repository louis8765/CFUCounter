#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 21:01:49 2022

@author: louiszhang
"""
import cv2


class Counter():
    def __init__(self,fName):
        self.imgname=fName
        self.colorList=[]
        self.minWidth=[]
        self.contourList=[]
        self.img=cv2.imread(self.imgname)
        self.imgforwatershed=self.img.copy()
        
        self.img=cv2.medianBlur(self.img, 3)
        self.imgray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        
    def test(self):
        cv2.imshow('sds',self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        


def detectColony(x,y,image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    print(hsv.shape)
    h=hsv[:,:,0]
    print(h.shape)
    R,C=h.shape
    color=h[x][y]
    
    
    def dfs(r,c,col):
        if h[r][c]==col:
            hue=h[r][c]
            h[r][c]=-5
            if r>=1:
                dfs(r-1,c,hue)
            if r+1<R:
                dfs(r+1,c,hue)
            if c>=1:
                dfs(r,c-1,hue)
            if c+1<C:
                dfs(r,c+1,hue)
    dfs(x,y,color)
    h[h>=0]=0
    h[h==-5]=255
    
    cv2.imshow('does this work',h)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


# =============================================================================
# 
# img=cv2.imread('3.png')
# detectColony(191,317,img)
# =============================================================================
# =============================================================================
# counter=Counter('3.png')
# cv2.imshow('sdds',counter.img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
#     
# =============================================================================
    
        
