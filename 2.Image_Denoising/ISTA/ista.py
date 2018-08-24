# -*- coding: utf-8 -*-
"""

@author: Aditya Dhookia
"""
import numpy as np
def ista(y,H,lamda,alpha):
    
    x = np.zeros((64,1))
    T = lamda/(2*alpha)
    error_old = 1
    error_ratio = 10
    k=1
    Hx = np.zeros((64,1))
    while( error_ratio > 10^-16):
        Hx = np.dot(H,x)
        #x = wthresh(x + ((H.transpose()*(y - Hx))/alpha),'s', T)
        X = x + np.dot(np.transpose(H)/alpha,(y - Hx)/alpha)
        res = (abs(X) - T)
        res = (res + abs(res))/2
        x  = np.sign(X)*res
        error_new = np.sum(np.abs(Hx[:] - y[:]))
        #error_new = np.sum(np.abs(Hx - y)
        error_ratio = (error_old - error_new)/error_old
        error_old = error_new
        k=k+1
    #end
    return x
    
