# -*- coding: utf-8 -*-
"""
@author : Aditya Dhookia
"""
import numpy as np
import math



def DCT_basis_gen(N):
    h = np.zeros((N, N))
    temp=np.zeros((N))
    
    for k  in range (0,N):
        if k == 0:
            a = math.sqrt(1.0/N)
        else:
            a = math.sqrt(2.0/N)
        
        for n in range (0,N):
            temp[n] = a * (math.cos((((2*n)+1)*(k)*(math.pi))/(2*N)))
           
        #end
        h[0:N,k] = temp[0:N]
        #end
    return h

    
