# -*- coding: utf-8 -*-
"""


@author: Aditya Dhookia
"""
#Header files
import cv2
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from quant import quant
from cnt import cnt 
#Reading image and normalising
frame1 = cv2.imread('foreman001.jpg',0) 
frame2 = cv2.imread('foreman060.jpg',0) 
frame1 = np.float64(frame1)
frame2 = np.float64(frame2)
#plt.imshow(frame1,cmap=plt.cm.gray)
#plt.imshow(frame2,cmap=plt.cm.gray)
#frame1 = np.round((frame-min_im)*255/(max_im-min_im));
#frame2 = frame2/255
m,n = frame1.shape
PSNR = 0
qs = 2 #step size
#Initialisation
blk_size = 8
row_ind = 0
col_ind = 0
T = np.zeros((8,1))
L = np.zeros((8,1))
out1 = np.zeros((8,8))
out2 = np.zeros((8,8))
out3 = np.zeros((8,8))
out4 = np.zeros((8,8))
output_image = np.zeros((m,n))
sad = 0
temp_sum = 1000000000000000
x1 = 0
y1 = 0
i=8
j=8
K=0
vari = np.zeros((4,1))
#Intra-prediction
while i < m-8:
     while j < n-8:
        T = frame2[      i      , j:j+blk_size]
        L = frame2[i:i+blk_size ,     j       ]

        T1 = frame1[i:i+blk_size , j:j+blk_size]
        L1 = frame2[i:i+blk_size , j:j+blk_size]
        
        for p in range(i-8,i+8):
            for q in range(j-8, j+8):
                for r in range(0, 8):
                    for s in range(0, 8):
                        sad = sad + abs((frame2[p+r,q+s]-T1[r,s]))
                if sad < temp_sum:
                    temp_sum = sad
                    x1 = p
                    y1 = q
                sad = 0
                
        out4 = frame1[x1:x1+8,y1:y1+8]
        temp_sum = 1000000000000000
        ########## ACCESSING INDIVIDUAL PIXELS 
        for k in range(0,blk_size):
            for l in range(0,blk_size):
                ###### CALC keeping in mind the INDICES of IMAGE
                out1[l,k]=T[k]                                  #vertical
                out2[l,k]=L[l]                                  #horizontal
                a = np.array([L,T])                            
                out3[l,k]=np.mean(a)                            #DC/mean    
        err1 = L1-out1
        err2 = L1-out2
        err3 = L1-out3
        err4 = L1-out4 
        vari[0] = np.var(err1)
        vari[1] = np.var(err2)
        vari[2] = np.var(err3)
        vari[3] = np.var(err4)
        ind = np.argmin(vari)
        if ind==0:
            out_blk = out1
            err_blk = err1
        elif ind==1:
            out_blk = out2
            err_blk = err2
        elif ind==2:
            out_blk = out3
            err_blk = err3
        else:
            out_blk = out4
            err_blk = err4
        
        if (np.max(np.abs(err_blk))!=0):
            dct_blk = cv2.dct(err_blk)
            dct_blk = (dct_blk-np.min(dct_blk))*16/(np.max(dct_blk)-np.min(dct_blk))
            dct_blk=quant(dct_blk,qs)
            K = K + cnt(dct_blk) 
            rec_err_blk = cv2.idct(dct_blk)
        else:
            rec_err_blk = err_blk
        output_blk = out_blk+rec_err_blk
        output_image[i:i+8,j:j+8] = output_blk
        j=j+8 
     i=i+8 
     j=0
plt.imshow(np.uint8(output_image),cmap=plt.cm.gray)
#PSNR = signoise(frame2,output_image)
a=0
mse= ((frame2[8:711,8:1271]-output_image[8:711,8:1271])**2).mean(axis=None)
ps= 20*np.log10(255)-10*np.log10(mse)
difference = frame2-output_image
np.mean(difference)
cv2.imwrite('q_2.png', output_image)
