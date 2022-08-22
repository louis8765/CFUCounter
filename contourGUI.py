# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:14:47 2022

@author: Louis
"""

#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
import math
from findmaxima2d import find_maxima, find_local_maxima
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from skimage.segmentation import watershed
from sklearn.naive_bayes import GaussianNB



def big(NAME):
    def backgroundRemove(image):
        imgray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #35,10 for .4
        thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,35,1)
    # =============================================================================
    #     se,thresh=cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # =============================================================================
        contours,heirarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        black=np.zeros(img.shape[:2], np.uint8)
        largest=max(contours,key=cv2.contourArea)
        (centerX,centerY),radius=cv2.minEnclosingCircle(largest)
        if cv2.contourArea(largest)>.2*imgray.shape[0]*imgray.shape[1]:
            cv2.drawContours(black, [largest], 0, 255, -1)
            ret,black=cv2.threshold(black,0,255,cv2.THRESH_BINARY)
            final=cv2.bitwise_and(img,img,mask=black)
        else:
            cv2.circle(black,(int(centerX),int(centerY)),int(radius),255,-1)
            final=cv2.bitwise_and(img,img,mask=black)
            
# =============================================================================
#         cv2.imshow('final',final)
# =============================================================================
        return final,centerX,centerY,radius
        
    def changeBackgroundColor(image):
        black_pixels = np.where(
        (image[:, :, 0] == 0) & 
        (image[:, :, 1] == 0) & 
        (image[:, :, 2] == 0))
    
        # set those pixels to white
        image[black_pixels] = [255, 255, 255]
        return image
    
    def watershed2(image,df):
        
        #create white background image containing only suspicious colonies
        mask=np.zeros(image.shape[:2],np.uint8)
        cv2.drawContours(mask,df['Contours'].tolist(),-1,255,-1)     
        newimg=cv2.bitwise_and(image,image,mask=mask)
        newimg[mask==0]=255
# =============================================================================
#         cv2.imshow('whatisthis',newimg)
# =============================================================================
        
        threshTemp=thresh
        threshTemp=cv2.bitwise_and(threshTemp,threshTemp,mask=mask)
# =============================================================================
#         cv2.imshow('threshtemp',threshTemp)
# =============================================================================
        kernel=np.ones((3,3),np.uint8)
        close=cv2.morphologyEx(threshTemp, cv2.MORPH_CLOSE, kernel)
        #find sure background and sure foreground
        bg=cv2.dilate(close,kernel)
# =============================================================================
#         cv2.imshow('background',bg)
# =============================================================================
        
        fg=localMinima2(newimg)
        fg=np.uint8(fg)
    
        #find unknown
        unknown=cv2.subtract(bg,fg)
        #generate markers for foreground and watershed
        ret, markers=cv2.connectedComponents(fg)
        markers=markers+1
        markers[unknown==255]=0
        threshTemp[threshTemp==255]=True
        threshTemp[threshTemp==0]=False
        markers=watershed(imgray,markers=markers,mask=threshTemp,compactness=.1)
        #show watershed results
    # =============================================================================
    #     imagecopy=image.copy()
    #     imagecopy[markers==-1]=[255,0,255]
    #     cv2.imshow('debugging',imagecopy)
    # =============================================================================
        copyyy=img.copy()
        contours=[]
        for label in np.unique(markers):
            if label==0:
                continue
    
            mask = np.zeros(imgray.shape, dtype="uint8")
            mask[markers == label] = 255
           	# detect contours in the mask and grab the largest one
            cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(copyyy,cnts,0,(0,0,255),1)
            contours.append(cnts[0])
# =============================================================================
#         cv2.imshow('debuggingcontours',copyyy)
# =============================================================================
    
        return contours
    
    def nothing(x):
        pass
    
    def localMinima2(temp):
        temp=cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
        temp=255-temp
        localmax=find_local_maxima(temp)
         
        plt.figure(figsize=(60,60))
        y,x,hehe=find_maxima(temp,localmax,3)

        black=np.zeros(temp.shape,np.uint8)
        for i in range(len(y)):
            black[y,x]=255
# =============================================================================
#         cv2.imshow('sureforeground',black)
# =============================================================================
        return black
    
    
    def calcInertia(row):
        if len(row[0])>=5:
             (x,y),(MA,ma),angle=cv2.fitEllipse(row['Contours'])
             return MA/ma
        return None
    
    def calcCircularity(row):
        area = row['Area']
        perimeter=cv2.arcLength(row['Contours'], True)
        if perimeter==0:
            return None
        circularity=4*math.pi*area/perimeter**2
        return circularity
    
    def calcCircularity2(contour):
        area=cv2.contourArea(contour)
        perimeter=cv2.arcLength(contour,True)
        if perimeter==0:
            return None
        circularity=4*math.pi*area/perimeter**2
        return circularity
    
    #look into this and what to do if there are three peaks
    def Otsu(image): 
        hist = cv2.calcHist([image],[0],None,[255],[1,255])
        hist=hist.ravel()
        hist_norm = hist/hist.sum()
        Q = hist_norm.cumsum()
        bins = np.arange(256)[1:]
        fn_min = np.inf
        thresh = -1
        for i in range(1,254):
            p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
            q1,q2 = Q[i],Q[253]-Q[i] # cum sum of classes
            if q1 < 1.e-6 or q2 < 1.e-6:
                continue
            b1,b2 = np.hsplit(bins,[i]) # weights
            # finding means and variances
            m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
            v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
            # calculates the minimization function
            fn = v1*q1 + v2*q2
            if fn < fn_min:
                fn_min = fn
                thresh = i
        if largestDist(hist)<10:
            return False
        print('ehehe',thresh)
        return thresh
    
    def largestDist(lst):   
        distance=-1
        prev=-1
        for index, value in enumerate(lst):
            if value!=0:
                if prev==-1:
                    prev=index
                    continue
                if index-prev>distance:
                    distance=(index-prev)
                    prev=index
        return distance
    
    def calcColor(row,HSV):
        mask=np.zeros(HSV.shape[:2],np.uint8)
        cv2.drawContours(mask,[row['Contours']],0,255,-1)
        color= cv2.mean(HSV[:,:,0],mask=mask)[0]
        return color
    
    def calcAreaRatio(row):
        area=row['Area']
        center,radius=cv2.minEnclosingCircle(row['Contours'])
        circleArea=radius**2*math.pi
        return area/circleArea
    
    def calcCenter(row):
        M = cv2.moments(row['Contours'])
        if M['m00']==0:
            return None
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return cx,cy
        
    
    def contour2df(contours):
        #initializing DataFrame for outlier detection || maybe even cell classification in the future?
        df2=pd.DataFrame()
        df2['Contours']=contours
        df2['Area']=df2.apply(lambda row: cv2.contourArea(row['Contours']),axis=1)
        df2['Inertia']=df2.apply(lambda row: calcInertia(row),axis=1)
        df2['Area']=pd.to_numeric(df2['Area'])
        df2['Circularity']=df2.apply(lambda row: calcCircularity(row),axis=1)
        HSV=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        df2['Color']=df2.apply(lambda row: calcColor(row,HSV),axis=1)
        df2['areaRatio']=df2.apply(lambda row: calcAreaRatio(row),axis=1)
        LAB=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        df2['Lightness']=df2.apply(lambda row: calcColor(row,LAB),axis=1)
    
        return df2
    
    def equalizeContour(img,contour):
        mask=np.zeros(img.shape,dtype='uint8')
        cv2.drawContours(mask,contour,-1,255)
    # =============================================================================
    #     plt.imshow(mask)
    # =============================================================================
        hist=cv2.calcHist([img], [0], mask, [256], [0,256])
        plt.hist(hist,255,[0,255])
        img=cv2.equalizeHist(img[mask>0])
        hist=cv2.calcHist([img], [0], mask, [256], [0,256])
        plt.hist(hist,255,[0,255])
        return img
    
    def optimalParameters(imgray,color,minWidth):
        fullimage=cv2.calcHist([imgray],[0],None,[252],[4,256])
        fullimage=fullimage.ravel()
        first=0
        second=0
        yhat=[]
        threshlist=[]
        yhat2=[]
        maxList=[]
        startingValue=minWidth//2*2+1
        endingValue=startingValue*3
        image=[]
        prev=0
        prevThresh=[]
        for z in range(4,1,-1):
            largest=0
            peakHeight=0
            yhat=0
            peaks=[]
            
            print('z:',z)
            for i in range(startingValue,endingValue,4):
                for j in range(1,30,2):
                    thresh=cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, i, j)
                    hist = cv2.calcHist([imgray],[0],thresh,[252],[4,256])
                    hist=hist.ravel()
                    yhat = savgol_filter(hist, 25, 3)
                    right,left,area,maxidx=maxRange(yhat,hist)
                    #not trivial find the best parameters for this
                    peaks,_=find_peaks(yhat,height=yhat[maxidx]*.1,width=5)
                
                    
                    if len(peaks)<z and len([i for i in peaks if i>color[1] and i<color[2]])>0:
                        area=sum(yhat[color[1]:color[2]])
                        if area>largest or yhat[color[0]]>peakHeight:
                            largest=area
                            peakHeight=yhat[color[0]]
                            first,second=i,j
                            yhat2=yhat
                            threshlist.append((i,j))
                            image=thresh
                            print(len(peaks),i,j,area)
                            
                            
                            
            print('largest: ',largest)
            

            if largest<.4*prev:
                return first,second,prevThresh
            prev=largest
            prevThresh=image

            maxList.append(maxidx)

    

        print(threshlist)
        return first,second,image
         
            
            
    def maxRange(yhat,hist):
        big=True
        small=True
        maxidx=np.argmax(yhat)
        right=maxidx
        left=maxidx
        while big or small:
            if right+1<len(yhat) and yhat[right+1]<=yhat[right]:
                right+=1
            else:
                big=False
            if left-1>-1 and yhat[left-1]<=yhat[left]:
                left-=1
            else:
                small=False
        return right,left,sum(hist[left-1:right]),maxidx    
        
        
    
    def drawCircles(image,contours,color,thickness):
        for i in contours:
            (x,y),radius = cv2.minEnclosingCircle(i)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(image,center,radius,color,thickness)
        return image
    
    
    
    def detectColony(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            finalUp=0
            prevArea=0
            prevContour=0
            prevMask=0
            prevCircularity=0
            areaList=[]
            circularityList=[]
            
            for i in range(0,50):
                imgraycopy=imgray.copy()
                imgraycopy[imgraycopy==255]=254
                cv2.floodFill(imgraycopy, None, seedPoint=(x,y), newVal=255,loDiff=50,upDiff=i)
                imgraycopy[imgraycopy<255]=0
                contours,hierarchy=cv2.findContours(imgraycopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                contour=contours[0]
                area = cv2.contourArea(contour)
                circularity=calcCircularity2(contour)
                areaList.append(area)
                circularityList.append(circularity)
                print('area',area,'circularity',circularity)
                if area-prevArea>500:
                    break
                
                prevArea=area
                prevContour=contour
                prevMask=imgraycopy
                prevCircularity=circularity

            contourList.append(np.array(prevContour))
            #setting minimum box dimensions for interative thresholding
            _,_,width,height = cv2.boundingRect(prevContour)
            print('wdith: ',width,'height:',height)
            minWidth.append(max(width,height))
            print('mindWidth:',minWidth)
            finalUp=i-1
            color=cv2.mean(imgray,mask=prevMask)[0]
            print('final',finalUp)
                    
            #create dataframe with attributes of ground truth
            
    
            #return histogram of t
            hist = cv2.calcHist([imgray],[0],prevMask,[252],[4,256])
            hist=hist.ravel()
            first,last=firstLastIndex(hist)
            colorList.append([int(color),first,last])
            
            #display updated image
            imcopy=img.copy()
            cv2.drawContours(imcopy,contourList,len(contourList)-1,(0,0,255),1)
            cv2.imshow('img',imcopy)
        
        
    def detectColony2(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            h=hsv[:,:,2]
            R,C=len(h),len(h[0])
            color=h[y][x]
            print(x,y,color)
            def dfs(r,c,col):
                if h[r][c]>=col-1 and h[r][c]<=col+1:
                    print('row',r,'col',c,'color',h[r][c],'prevColor', col)
                    col=h[r][c]
                    h[r][c]=-5
                    if r>=1:
                        dfs(r-1,c,col)
                    if r+1<R:
                        dfs(r+1,c,col)
                    if c>=1:
                        dfs(r,c-1,col)
                    if c+1<C:
                        dfs(r,c+1,col)
            dfs(y,x,color)
            
            h[h==-5]=255
            cv2.imshow('binary',h)
              
    def calcDistance(point1,point2):
        return abs(point1[0]-point2[0])+abs(point1[1]-point2[1])
    
    def firstLastIndex(histogram):         
        lst=[i for i,e in enumerate(histogram) if e!=0]
        return lst[0],lst[-2]
            
    def resizeImg(img):
        height,width=img.shape[:2]
        pixels=height*width
        maxPixels=2000000
        ratio=1
        if pixels>maxPixels:
            ratio=maxPixels/pixels
            ratio=math.sqrt(ratio)
        img=cv2.resize(img,None,fx=ratio,fy=ratio)
        return img
    ##########################################################################################################################################################
    ##########################################################################################################################################################
    ##########################################################################################################################################################
    ##########################################################################################################################################################
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    imgname=NAME
    colorList=[]
    minWidth=[]
    contourList=[]
    img=cv2.imread(imgname)
    img=resizeImg(img)
    originalImg=img

    
    imgforwatershed=img.copy()
    
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', detectColony)
    
    img=cv2.medianBlur(img, 3)
    imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('img',img)
    
    img,midX,midY,radius=backgroundRemove(img)
    imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #wait until all a priori are selected
    while(1):
        k=cv2.waitKey(1) & 0xFF
        if k==113:
            break
        elif k==114:
            print('whatttt')
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return big(NAME)
            
    
    groundTruth=contour2df(contourList)
    print('pre loation length',len(groundTruth))
    groundTruth['Location']=groundTruth.apply(lambda row: calcCenter(row),axis=1)
    print('post liocation length',len(groundTruth))
    print('LOCATION: ',groundTruth['Location'])
    

    thresh=np.zeros(imgray.shape,np.uint8)
    for i in range(len(colorList)):
        first,second,threshold=optimalParameters(imgray,colorList[i],minWidth[i])

        thresh=cv2.bitwise_or(thresh, threshold)
    print(first,second)
    
    cv2.imshow("thresh",thresh)
    
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hierarchy=hierarchy[0,:,3]
    
    #create figure

    
    
    #testing out size of contours
    areas=[cv2.contourArea(x) for x in contours]
    
    
    #initializing DataFrame for outlier detection || maybe even cell classification in the future?
    df=pd.DataFrame()
    df['Contours']=contours
    df['Area']=df.apply(lambda row: cv2.contourArea(row['Contours']),axis=1)
    df['Inertia']=df.apply(lambda row: calcInertia(row),axis=1)
    df['Area']=pd.to_numeric(df['Area'])
    df['Circularity']=df.apply(lambda row: calcCircularity(row),axis=1)
    largest=df['Contours'][df['Area'].idxmax()]
    df['hierarchy']=hierarchy
    df['areaRatio']=df.apply(lambda row: calcAreaRatio(row),axis=1)
    df['Location']=df.apply(lambda row: calcCenter(row),axis=1)
    HSV=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    df['Color']=df.apply(lambda row: calcColor(row,HSV),axis=1)
    
    LAB=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    df['Lightness']=df.apply(lambda row: calcColor(row,LAB),axis=1)
    
    
    df = df.replace(to_replace='None', value=np.nan).dropna()
    df=df.reset_index(drop=True)
    print(df.isnull().any())
    
    
    realIndexList=[]
    #create new groundTruth
    for index,row in groundTruth.iterrows():
        distance=500
        realIndex=-1
        for index1,row1 in df.iterrows():
            if calcDistance(row['Location'],row1['Location'])<distance:
                distance=calcDistance(row['Location'],row1['Location'])
                realIndex=index1
                print(distance,row['Location'],row1['Location'])
        realIndexList.append(realIndex)
    
    groundTruth=df.iloc[realIndexList]
    
        
    
    #manual pruning by attributes
    df=df[df['Area']<radius**2*math.pi*.6]
    df=df[df['Area']>.5*(min(groundTruth['Area']))]
    df.reset_index(drop=True,inplace=True)
    #plotting areas
    
    #determine correct index for groundtruth now that df has been pruned
    groundTruth=df[df['Location'].isin(groundTruth['Location'].values)]
        
    imgcopy=img.copy()
    
    
    
    
    
        
    #DBSCAN TO FIND OUTLIERS AND CLASSIFY
    df_dbscan=df.copy().drop(['Contours','hierarchy','Lightness','Location'],axis=1)
    
    
    
    #nearest neighbors to find good eps for dbscan
    scaler=MinMaxScaler()
    df_dbscan[['Area','Circularity','Inertia','Color','areaRatio']]=scaler.fit_transform(df_dbscan[['Area','Circularity','Inertia','Color','areaRatio']])
    neigh=NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(df_dbscan)
    distances,indices=nbrs.kneighbors(df_dbscan)
    distances=np.sort(distances,axis=0)
    distances=distances[:,1]
    kn = KneeLocator(np.arange(len(distances)), distances, curve='convex', direction='increasing')
    print('knee: ',kn.knee,distances[kn.knee]*1000)
    
    

    df_copy=pd.DataFrame()
    
    
    singleIdx=-2

    imgcopy=img.copy()
    dbscan=DBSCAN(eps=distances[kn.knee],min_samples=int(len(df)*.05)).fit(df_dbscan)
    #plotting dbscan results
    color=iter([[0,0,255],[0,255,0],[255,0,0],[0,255,255],[0,0,0],[255,255,255],[125,125,0],[0,125,125]])
    plotColor=iter(['r','g','b','y','black'])
    df_copy=df.copy()
    df_copy['cluster']=dbscan.labels_
    groups=df_copy.groupby('cluster')
    lst=[]
    #show cluster for debugging purposes
    for name,group in groups:
        cv2.drawContours(imgcopy,group['Contours'].tolist(),-1,next(color),1)
        lst.append(group['Circularity'].mean()+len(group)/len(df_copy))           

    if lst.index(max(lst))-1==-1:
        lst2=lst[1:]
        print(lst2)
        print(lst)
        singleIdx=lst2.index(max(lst2))
    else:
        singleIdx=lst.index(max(lst))-1
        
        
    df_copy.reset_index(drop=True,inplace=True)
    
    #set groundTruth to include the DBSCAN cluster
    groundTruth['cluster']=df_copy[df_copy['Location'].isin(groundTruth['Location'].values)]['cluster']
    
    
    df_single=df_copy[df_copy['cluster'].isin(groundTruth['cluster'])]
    #set df_single to only contain the correct hierarchy
    common=df_single['hierarchy'].mode().iloc[0].item()
    df_single=df_single[df_single['hierarchy']==common]
    notCircular=df_single[df_single['Circularity']<min(groundTruth['Circularity'])-.1]
    df_single=df_single[df_single['Circularity']>=min(groundTruth['Circularity'])-.1]
    
    
    testImage=img.copy()
    cv2.drawContours(testImage,df_single['Contours'].tolist(),-1,(255,255,0),1)

    
    #generate bayes classifer based on color and lightness
    X=df_single[['Color','Lightness']]
    y=df_single['cluster']
    gnb=GaussianNB()
    gnb.fit(X,y)
    
    
    

    imgcopy=img.copy()
    marker=watershed2(img, df_copy[~df_copy.index.isin(df_single.index)])
    df_final=contour2df(marker)

    df_final=df_final[(df_final['Area']<df_single['Area'].max()*2) & (df_final['Area']>df_single['Area'].min())]
    df_final=df_final[df_final['areaRatio']>df_single['areaRatio'].min()-.3]
    if len(df_final):
        df_final['cluster']=gnb.predict(df_final[['Color','Lightness']])
        

    clusterList=df_single['cluster'].unique()
    clusterSize=[]
    for count,i in enumerate(clusterList):
        if len(df_final):
            cv2.drawContours(originalImg,df_final[df_final['cluster']==i]['Contours'].tolist(),-1,(255*count,255,0),1)
            cv2.drawContours(originalImg,df_single[df_single['cluster']==i]['Contours'].tolist(),-1,(255*count,255,0),1)
            clusterSize.append(len(df_final[df_final['cluster']==i])+len(df_single[df_single['cluster']==i]))
            print('cluster',count,':',clusterSize[-1])
        else:
            cv2.drawContours(originalImg,df_single[df_single['cluster']==i]['Contours'].tolist(),-1,(255*count,255,0),1)
            clusterSize.append(len(df_single[df_single['cluster']==i]))
            print('cluster',count,':',clusterSize[-1])
            
            
    
    totalColonies=len(df_single)+len(df_final)

    print('original: ',len(df_copy[df_copy['cluster']==singleIdx]))
    print('new',len(df_final))
    img=changeBackgroundColor(img)
    print('Total Colony Count: '+str(totalColonies), (int(midX)-200,int(midY-radius-50)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    return originalImg,clusterSize
        