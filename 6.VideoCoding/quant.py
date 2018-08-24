# -*- coding: utf-8 -*-
"""


@author: Aditya Dhookia
"""
def quant(matrix,q):
    for i in range(0,8):
        for j in range(0,8):
            if q==1:
                if matrix[i,j]>=0 and matrix[i,j] < 1 :
                    matrix[i,j]=0 
                elif matrix[i,j]>=1 and matrix[i,j] < 2 :
                    matrix[i,j]=1 
                elif matrix[i,j]>=2 and matrix[i,j] < 3 :
                    matrix[i,j]=2 
                elif matrix[i,j]>=3 and matrix[i,j] < 4 :
                    matrix[i,j]=3 
                elif matrix[i,j]>=4 and matrix[i,j] < 5 :
                    matrix[i,j]=4 
                elif matrix[i,j]>=5 and matrix[i,j] < 6 :
                    matrix[i,j]=5 
                elif matrix[i,j]>=6 and matrix[i,j] < 7 :
                    matrix[i,j]=6 
                elif matrix[i,j]>=7 and matrix[i,j] < 8 :
                    matrix[i,j]=7 
                elif matrix[i,j]>=8 and matrix[i,j] < 9 :
                    matrix[i,j]=8 
                elif matrix[i,j]>=9 and matrix[i,j] < 10 :
                    matrix[i,j]=9 
                elif matrix[i,j]>=10 and matrix[i,j] < 11 :
                    matrix[i,j]=10 
                elif matrix[i,j]>=11 and matrix[i,j] < 12 :
                    matrix[i,j]=11 
                elif matrix[i,j]>=12 and matrix[i,j] < 13 :
                    matrix[i,j]=12 
                elif matrix[i,j]>=13 and matrix[i,j] < 14 :
                    matrix[i,j]=13 
                elif matrix[i,j]>=14 and matrix[i,j] < 15 :
                    matrix[i,j]=14 
                elif matrix[i,j]>=15 and matrix[i,j] < 16 :
                    matrix[i,j]=15
                elif matrix[i,j]==16:
                    matrix[i,j]=16
            elif q==2:
                if matrix[i,j]>=0 and matrix[i,j] < 2 :
                    matrix[i,j]=0 
                elif matrix[i,j]>=2 and matrix[i,j] < 4 :
                    matrix[i,j]=2 
                elif matrix[i,j]>=4 and matrix[i,j] < 6 :
                    matrix[i,j]=4 
                elif matrix[i,j]>=6 and matrix[i,j] < 8 :
                    matrix[i,j]=6 
                elif matrix[i,j]>=8 and matrix[i,j] < 10 :
                    matrix[i,j]=8
                elif matrix[i,j]>=10 and matrix[i,j] < 12 :
                    matrix[i,j]=10 
                elif matrix[i,j]>=12 and matrix[i,j] < 14 :
                    matrix[i,j]=12 
                elif matrix[i,j]>=14 and matrix[i,j] < 16 :
                    matrix[i,j]=14
                elif matrix[i,j]==16:
                    matrix[i,j]=16
            elif q==4:
                if matrix[i,j]>=0 and matrix[i,j] < 4 :
                    matrix[i,j]=0 
                elif matrix[i,j]>=4 and matrix[i,j] < 8 :
                    matrix[i,j]=4 
                elif matrix[i,j]>=8 and matrix[i,j] < 12 :
                    matrix[i,j]=8 
                elif matrix[i,j]>=12 and matrix[i,j] < 16 :
                    matrix[i,j]=12 
                elif matrix[i,j]==16:
                    matrix[i,j]=16
    return matrix
