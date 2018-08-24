# -*- coding: utf-8 -*-
"""
author: Aditya Dhookia
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
#import skimage
#import scipy as sp

from ista import ista
from DCT_basis_gen import DCT_basis_gen
#from matplotlib import pyplot as plt

img = cv2.imread('128x128-facebook.png',0)
img = np.float64(img)
#creating noisy image
mean = 0;
var = 0.1
sigma = var**0.5
a = np.random.normal(0,sigma,(img.shape))
max_im = np.max(img)
min_im = np.min(img)
image_norm = np.float64((img-min_im)/(max_im-min_im))
noisy_img = image_norm+a
max_im_n = np.max(noisy_img)
min_im_n = np.min(noisy_img)
noisy = np.round((noisy_img-min_im_n)*255/(max_im_n-min_im_n))
d = np.float64((noisy-np.min(noisy))/(np.max(noisy)-np.min(noisy)))

#initialising
B=np.zeros((64,64))
x3=np.zeros((128,128))
x2=np.zeros((8,8))



for p in range (0,121):  # for a 128x128 image
    for q in range (0,121):
        
        y=d[p:8+p,q:8+q]  # 8x8 blocks in image
       
        y=np.reshape(y,(64,1)) #converting to vector
        
        B1 = DCT_basis_gen(8) # creating Bases
        
        L=0
        for j in range(0,8):   #creating dictionary
            for k in range(0,8):
                z = B1[0:8,j]
                z = np.reshape(z,(8,1))
                z1 = B1[0:8,k]
                z1 = np.reshape(z1,(8,1))
                z1 = np.transpose(z1)
                B2= z*z1
                B2=np.reshape(B2,(1,64))
                B[0:64,L]=B2
               
                L=L+1
        #end
        #lamda and alpha parameters
        lamda = 0.1
        alpha = np.round((np.max(np.linalg.eigvals(np.dot(np.transpose(B),B)))).real)
        
        #calling ista
        x = ista (y, B, lamda, alpha)

        #creating back the image and placing it correctly         
        x1=np.dot(np.transpose(x),B)
        x2[0:8,0:8]=np.reshape(x1,(8,8))
        x3[p:8+p,q:8+q]=x2
        q=q+8
    #end 
    p=p+8

fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
ax1.imshow(noisy,cmap=plt.cm.gray)
ax1.set_title('Noisy image')
ax2 = fig1.add_subplot(122)
ax2.imshow(x3,cmap=plt.cm.gray)
ax2.set_title('Denoised image')
