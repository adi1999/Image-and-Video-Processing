#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""


@author: Aditya Dhookia
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage import color
from skimage import io


cap = cv2.VideoCapture("Q1.avi")

fgbg = cv2.createBackgroundSubtractorMOG2() #BakcgroundSubtractorMOG() does not work in my machine hence replaced with this.
count = 0

while count<10:
    count = count+1
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.imwrite('frame10.png',frame)
abcd =fgmask

cv2.destroyAllWindows()

gray = cv2.GaussianBlur(abcd, (21, 21), 0)

erosion = cv2.erode(gray,None,iterations = 8)
erosion = cv2.dilate(erosion, None, iterations=1)
plt.imshow(erosion,cmap=plt.get_cmap('gray'))

_, contours, _ = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 

for c in contours:
    if cv2.contourArea(c) > 2000:
        max_contour = c
        
                   

rect = cv2.boundingRect(max_contour)
    
x,y,w,h = rect
cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
plt.imshow(frame,cmap=plt.get_cmap('gray'))

frame_gray = color.rgb2gray(io.imread('frame10.png'))
b_box = np.zeros([h,w])
b_box = frame_gray[y:y+h+1,x:x+w+1]
sad = 0
temp_sum = 1000000000000000
x1 = 0
y1 = 0


while count < 20:
    count = count+1
    ret, fr_1 = cap.read()
    fr = cv2.cvtColor( fr_1, cv2.COLOR_RGB2GRAY )
    for i in range(70,110):
        for j in range(150, 200):
            for k in range(0, len(b_box)):
                for l in range(0, len(b_box[0])):
                    sad = sad + abs((fr[i+k,j+l]-b_box[k,l]))
            if sad < temp_sum:
                temp_sum = sad
                x1 = i
                y1 = j
            sad = 0
    plt.figure(count)
    cv2.rectangle(fr_1,(y1,x1),(y1+w,x1+h),(0,255,0),2)
    plt.imshow(fr_1,cmap=plt.get_cmap('gray'))

    
