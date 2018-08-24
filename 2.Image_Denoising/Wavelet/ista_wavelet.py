# -*- coding: utf-8 -*-
"""
@author : Aditya Dhookia
"""

import numpy as np
import pywt

def ista_wavelet(img,d,type,lamda):    
    error_ratio=10
    error_old=1
    
   
    while(error_ratio >0.05 ): 
        
        coeffs = pywt.wavedecn(d,type,mode='symmetric',level=3)
        cA= coeffs[0]
        
        Level3 = coeffs[1]
        V3=Level3['ad']
        H3=Level3['da']
        D3=Level3['dd']
        
        Level2 = coeffs[2]
        V2=Level2['ad']
        H2=Level2['da']
        D2=Level2['dd']
        
        Level1 = coeffs[3]
        V1=Level1['ad']
        H1=Level1['da']
        D1=Level1['dd']
        
        
        #do processing that is thresholding
        
    
        T= lamda/2
        
        V3=pywt.threshold(V3, T, mode='soft', substitute=0)
        H3=pywt.threshold(H3, T, mode='soft', substitute=0)
        D3=pywt.threshold(D3, T, mode='soft', substitute=0)
        
        V2=pywt.threshold(V2, T, mode='soft', substitute=0)
        H2=pywt.threshold(H2, T, mode='soft', substitute=0)
        D2=pywt.threshold(D2, T, mode='soft', substitute=0)
        
        V1=pywt.threshold(V1, T, mode='soft', substitute=0)
        H1=pywt.threshold(H1, T, mode='soft', substitute=0)
        D1=pywt.threshold(D1, T, mode='soft', substitute=0)
        
        new_coeffs = [cA,{'ad':V3,'da':H3,'dd':D3},{'ad':V2,'da':H2,'dd':D2},{'ad':V1,'da':H1,'dd':D1}]
        denoised_img = pywt.waverecn(new_coeffs,type,mode='symmetric')
        
        d= denoised_img   
        
        error_new = np.sum(np.abs(d[:] - img[:]))
        #error_new = np.sum(np.abs(Hx - y)
        error_ratio = (error_old - error_new)/error_old
        error_old = error_new
        
        
    return denoised_img
        
