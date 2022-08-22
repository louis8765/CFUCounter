# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 12:55:06 2022

@author: Louis
"""

from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df=pd.read_excel('cellseg data.xlsx')

plotList=[]
f=plt.figure()
plt.figure(figsize=(7,7))
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=1,hspace=.5)
ax1 = plt.subplot(gs[0, :2], )
ax2 = plt.subplot(gs[0, 2:])
ax3 = plt.subplot(gs[1, 1:3])
plotList=[ax1,ax2,ax3]


x = df['manual'].values
for count,column in enumerate(df.columns[2:]):
    y = df[column].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    
    confidence_interval = 1.96*std_err
    print(column)
    print('CI: ',confidence_interval)
    print('CI Range:',str(slope-confidence_interval),str(slope+confidence_interval))
    print('slope: ',slope,'intercept:',intercept,'r_value',r_value,'p-value:',p_value,'std error:',std_err)
    
    
    
    plotList[count].plot(x, y, 'o', label='original data')
    plotList[count].plot(x, intercept + slope*x, 'r--', label='y={:.2f}x+{:.2f}'.format(slope,intercept))
    print()


ax1.legend()
ax2.legend()
ax3.legend()
ax3.set_xlabel('Manual Count')
ax3.set_ylabel('Automated Count')
ax2.set_xlabel('Manual Count')
ax2.set_ylabel('Automated Count')
ax1.set_xlabel('Manual Count')
ax1.set_ylabel('Automated Count')

ax1.set_title('CFUCounter')
ax2.set_title('OpenCFU')
ax3.set_title('Autocellseg')
plt.figure(figsize=(50,50))

plt.show()

