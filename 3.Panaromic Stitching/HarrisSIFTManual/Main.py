#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""


@author: Aditya Dhookia
"""
import cv2
from descriptor import detect_descriptor
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('BK_left.JPG',0)
N = 50  #top N features
rows,cols = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),1,1)  #1 degree
dst = cv2.warpAffine(img,M,(cols,rows))
img_tilt=dst

# Get the values of co-ordinates corresponding to these images
kpd,ind = detect_descriptor(img,N)
kpd2,ind2 = detect_descriptor(img_tilt,N)

#Plotting individual
n1,n2 = img.shape
final_image = np.zeros([n1,n2])
final_image1 = np.zeros([n1,n2])
final_image[ind]=255
for i in range(0,n1):
    for j in range(0,n2):
        if final_image[i,j] == 255:
            cv2.circle(img,(j,i), 2, (255,0,0), 1)
final_image1[ind2]=255
for i in range(0,n1):
    for j in range(0,n2):
        if final_image1[i,j] == 255:
            cv2.circle(img_tilt,(j,i), 2, (255,0,0), 1)
            
#Finding correlation between the two - Algorithm
index_corr = np.zeros([2,50])            
d = np.zeros([50,50]) 
z=0
for p in range(0,50):
    for q in range(0,50):
        d[p,q] = np.linalg.norm(kpd[p,:]-kpd2[q,:])
    dsorted = sorted(d[p,:])
    d1 = dsorted[0]
    d2 = dsorted[1]
    r = d1/d2
    if r<0.9:
        sm_ind = np.where(d[p,:]==d1) #save i and sm_index which are corresponding points
        
        index_corr[0,z]=p
        index_corr[1,z] = sm_ind[0][0] #there will be some zeros at the end of index_corr
        z=z+1
corr_index = (index_corr[:,0:z]).astype(int)

#Plotting of the concatenated images
con_image = np.concatenate((img,img_tilt), axis=1)
x1 = np.zeros([1,len(corr_index[0])]).astype(int)  # as co-ordinates are integers
y1 = np.zeros([1,len(corr_index[0])]).astype(int)
x2 = np.zeros([1,len(corr_index[0])]).astype(int)
y2 = np.zeros([1,len(corr_index[0])]).astype(int)
l = len(corr_index[0])
for x in range(0,l):
    x1[0,x] = ind[0][corr_index[0,x]].astype(int)
    y1[0,x] = ind[1][corr_index[0,x]].astype(int)
    x2[0,x] = ind2[0][corr_index[1,x]].astype(int)
    y2[0,x] = ind2[1][corr_index[1,x]].astype(int)
for q in range(0,l):
    cv2.line(con_image,(y1[0,q],x1[0,q]),(img.shape[1]+y2[0,q],x2[0,q]),(255,255,255),1)
plt.imshow(con_image,cmap=plt.cm.gray)
cv2.imwrite('correspondence.png', con_image)
