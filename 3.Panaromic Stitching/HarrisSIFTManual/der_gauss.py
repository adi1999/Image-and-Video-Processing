# -*- coding: utf-8 -*-
"""


@author: Aditya Dhookia
"""

import numpy as np

def der_gauss(shape,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    nx, ny = shape
    xv = np.linspace(-1, 1, nx)
    yv = np.linspace(-1, 1, ny)
    x1, y1 = np.meshgrid(xv, yv)
           
    
    h = np.exp( -(x1*x1 + y1*y1) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    gauss_x = -x1/sigma*sigma
    h1 = gauss_x * h
    return h1
