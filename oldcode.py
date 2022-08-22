#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:51:34 2022

@author: louiszhang
"""

def localMinima(temp,index):
    temp=cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
    cv2.imshow('first'+str(index),temp)
    #find x and y derivatives
    xderv=cv2.Sobel(temp, cv2.CV_16S, 1, 0)
    yderv=cv2.Sobel(temp, cv2.CV_16S, 0, 1)

    xderv=cv2.convertScaleAbs(xderv)
    yderv=cv2.convertScaleAbs(yderv)
    
    #find gradient roughly sum of half of each derivative
    grad = cv2.addWeighted(xderv, 0.5, yderv, 0.5, 0)  
# =============================================================================
#     cv2.imshow('gradient'+str(index),grad)
# =============================================================================
# =============================================================================
#     cv2.imwrite('grayscale'+str(xderv[5][5])+'.jpg',temp)
# =============================================================================
    
    blank,grad=cv2.threshold(grad, 20, 255, cv2.THRESH_BINARY_INV)
    
# =============================================================================
#     cv2.imshow('thresholdedgrad'+str(index),grad)
# =============================================================================
# =============================================================================
#     cv2.imwrite('gradient'+str(xderv[5][5])+'.jpg',grad)
# =============================================================================
    grad=clear_border(grad)    
# =============================================================================
#     grad=cv2.dilate(grad, (3,3))
#     grad=cv2.erode(grad,(3,3))
# =============================================================================
    grad=cv2.morphologyEx(grad, cv2.MORPH_CLOSE, (3,3))
    final=cv2.bitwise_and(temp,temp,mask=grad)
# =============================================================================
#     cv2.imshow('preotsu'+str(index),final)
# =============================================================================
# =============================================================================
#     cv2.imwrite('overlay'+str(xderv[5][5])+'.jpg',final)
# =============================================================================
    value=Otsu(final)
    print(index,value)
    if value:   
        _,final=cv2.threshold(final,value,255,cv2.THRESH_TOZERO_INV)
    _,final=cv2.threshold(final,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)     
    cv2.imshow('final'+str(index),final)
# =============================================================================
#     cv2.imwrite('final'+str(xderv[5][5])+'.jpg',final)
# =============================================================================
    return final


def generalWatershed(image,threshold,x,y,w,h,contour):
    #produces a bounding box of colony w/ black background
    mask=np.zeros(image.shape[:2],np.uint8)
    cv2.drawContours(mask,contour,0,255,-1)     
    newimg=cv2.bitwise_and(image,image,mask=mask)
    newimg[mask==0]=255
    cropped=newimg[y:y+h,x:x+w]
    temp=image[y:y+h,x:x+w]
    gray=cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
    
    _,threshTemp=cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel=np.ones((3,3),np.uint8)
    close=cv2.morphologyEx(threshTemp, cv2.MORPH_CLOSE, kernel)
    #find sure background and sure foreground
    bg=cv2.dilate(close,kernel)
    
    
    ret, fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# =============================================================================
#     fg=localMinima(cropped,x)
# =============================================================================
    fg=np.uint8(fg)

    #find unknown
    unknown=cv2.subtract(bg,fg)

    #generate markers for foreground
    ret, markers=cv2.connectedComponents(fg)
    markers=markers+1
    markers[unknown==255]=0
    markers=cv2.watershed(cropped,markers)
    
    
    cropped[markers==-1]=[255,0,0]
    temp[markers==-1]=[255,0,255]
    image[y:y+h,x:x+w]=temp
    return markers


    fullimage=cv2.calcHist([imgray],[0],None,[252],[4,256])
    fullimage=fullimage.ravel()
    first=0
    second=0
    largest=0
    yhat=[]
    threshlist=[]
    yhat2=[]
# =============================================================================
#     optimalValues={}
# =============================================================================
    for i in range(5,40,2):
        for j in range(5,30,6):
            thresh=cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, i, j)
# =============================================================================
#             what=cv2.bitwise_and(imgray,imgray,mask=thresh)
#             what=cv2.cvtColor(what,cv2.COLOR_BGR2RGB)
# =============================================================================
# =============================================================================
#             cv2.imshow('thresh'+str(i)+" "+str(j),what)
# =============================================================================
            hist = cv2.calcHist([imgray],[0],thresh,[252],[4,256])
            hist=hist.ravel()
            yhat = savgol_filter(hist, 51, 3)
            right,left,area,maxidx=maxRange(yhat,hist)
            peaks,_=find_peaks(yhat,height=yhat[maxidx]*.2,width=(right-left)/4)
            if area>largest and (right-left)<200 and len(peaks)==1:
                largest=area
                first,second=i,j
                yhat2=yhat
                threshlist.append((i,j))
            
# =============================================================================
#             yhat2=yhat/yhat[maxidx]
#             rightslope=abs(yhat2[maxidx]-yhat2[right])/(maxidx-right)
#             leftslope=abs(yhat2[maxidx]-yhat2[left])/(maxidx-left)
# =============================================================================
        
            
            
            fig,axs=plt.subplots(1,2,gridspec_kw={'width_ratios': [2, 3]})
            fig.set_figheight(20)
            fig.set_figwidth(30)
            fig.suptitle(str(i)+' '+str(j)+'sum: '+str(sum(hist))+'max: '+str(np.argmax(hist))+'leftright: '+str(left)+' '+str(right))
            axs[0].plot(yhat2)
            for k in peaks:
                axs[0].axvline(x=k)
            axs[1].imshow(thresh)
            plt.show()
            
# =============================================================================
#             optimalValues[(i,j)]=[leftslope,area]
#     print(optimalValues)
#     print()
#     print()
#     sorted_list= list(sorted(optimalValues.items(), key=lambda item: item[1][1]))
#     top20=sorted_list[-25:]
#     top20=dict(top20)
#     print(top20)
#     max_value=max(top20.items(),key=lambda item: item[1][0])
#     print(max_value[0][0],max_value[0][1])
#     return max_value[0][0],max_value[0][1]
# =============================================================================
    

            

# =============================================================================
#                 plt.plot(yhat)
#                 plt.title(str(i)+' '+str(j)+'sum: '+str(sum(hist))+'max: '+str(np.argmax(hist))+'leftright: '+str(left)+' '+str(right))
#                 for k in peaks:
#                     plt.axvline(x=k)
#                 plt.show()
# =============================================================================
    
    plt.plot(yhat2)
    plt.title('area: '+str(largest)+'first:'+str(first)+'second: '+str(second))
    plt.show()
    print(threshlist)
    return first,second