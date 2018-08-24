# -*- coding: utf-8 -*-
"""
@author : Aditya Dhookia
"""
import cv2
import numpy as np
from ista_wavelet import ista_wavelet
from matplotlib import pyplot as plt

img = cv2.imread('lena512color.tiff',0)
img = np.float64(img)

sigma = 25.55

noise=np.random.normal(0,sigma,(img.shape))
noisy_img=noise+img
d=noisy_img

#parameters lamda and type
lamda =30
type = 'db8'


denoised_img = ista_wavelet(img,d,type,lamda)


fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
ax1.imshow(noisy_img,cmap=plt.cm.gray)
ax1.set_title('Noisy image')
ax2 = fig1.add_subplot(122)
ax2.imshow(denoised_img,cmap=plt.cm.gray)
ax2.set_title('Denoised image')


cv2.imwrite('Noisy image wavelet.png',d)
cv2.imwrite('Denoised image wavelet.png',denoised_img)









