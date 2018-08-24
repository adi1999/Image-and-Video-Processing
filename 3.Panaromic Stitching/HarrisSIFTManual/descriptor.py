from gaussian_filt import matlab_style_gauss2D
from der_gauss import der_gauss
from der_gauss_y import der_gauss_y
import numpy as np

#import cv2
from scipy import signal
from largest_indices import largest_indices as li
import math

def detect_descriptor(img, N):

    
    h= matlab_style_gauss2D(shape=(21,21),sigma=2)
    h1=der_gauss(shape=(5,5),sigma=1)
    h2=der_gauss_y(shape=(5,5),sigma=1)
    
    #FFT_filter1 = np.abs(np.fft.fft2(np.pad(h,((64,64),(64,64)),'constant',constant_values=(0,0))))               #Change variable according to H1, H2, H3
    #FFT_filter2 = np.abs(np.fft.fft2(np.pad(h2,((64,64),(64,64)),'constant',constant_values=(0,0))))               #Change variable according to H1, H2, H3
    
    
    
    img = np.float64(img)
    n1,n2 = img.shape
    
    
    Ix = signal.convolve2d(img, h1, boundary='symm', mode='same')
    Iy = signal.convolve2d(img, h2, boundary='symm', mode='same')
    
    mag = np.zeros([n1,n2])
    orient = np.zeros([n1,n2])
    bins = np.zeros([n1,n2])
    q = 45
    for i in range(0,n1):
        for j in range(0,n2):
            mag[i,j] = math.sqrt(Ix[i,j]**2 + Iy[i,j]**2)
            if Ix[i,j]==0:
                orient[i,j] = 90
            else:
                orient[i,j] = math.degrees(math.atan2(Iy[i,j],Ix[i,j]))
               
            orient[i,j] = (orient[i,j] + 360) % 360
            bins[i,j] = np.floor((orient[i,j]+(q/2))/q)
            bins=bins.astype(int)
            if bins[i,j]==8:
                bins[i,j]=0
                    
    mag = np.pad(mag,((8,8),(8,8)),'symmetric')
    bins = np.pad(bins,((8,8),(8,8)),'symmetric')
    
    Ixx = Ix*Ix
    Iyy = Iy*Iy
    Ixy = Ix*Iy
    
    Ixx = signal.convolve2d(Ixx, h, boundary='symm', mode='same')
    Iyy = signal.convolve2d(Iyy, h, boundary='symm', mode='same')
    Ixy = signal.convolve2d(Ixy, h, boundary='symm', mode='same')
    
    A = np.zeros([2*n1,2*n2])
    H = np.zeros([n1,n2])
    final_image = np.zeros([n1,n2])
    final_image1 = np.zeros([n1,n2])
    temp = np.zeros([2,2])
    temp1 = np.zeros([3,3])
    alpha = 0.06
    for i in range (0, n1):
        for j in range(0,n2):
            for k in range(0,2):
                for l in range(0,2):
                    if k == 0 and l == 0:
                        A[2*i+k,2*j+l] = Ixx[i,j]
                        temp[k,l] = Ixx[i,j]
                    elif k == 0 and l == 1:
                        A[2*i+k,2*j+l] = Ixy[i,j]
                        temp[k,l] = Ixy[i,j]
                    elif k == 1 and l == 0:
                        A[2*i+k,2*j+l] = Ixy[i,j]
                        temp[k,l] = Ixy[i,j]
                    elif k == 1 and l == 1:
                        A[2*i+k,2*j+l] = Iyy[i,j]
                        temp[k,l] = Iyy[i,j]
    
            H[i,j] = np.linalg.det(temp) - alpha*np.trace(temp*temp)
    #end
    max_im = np.max(H)
    min_im = np.min(H)
    H = np.round((H-min_im)*255/(max_im-min_im));
    
    H = np.pad(H,((1,1),(1,1)),'constant',constant_values=(0,0))
    
    for i in range(1,n1+1):
        for j in range(1,n2+1):
            for k in range(0,3):
                for l in range(0,3):
                    temp1[k,l] = H[i-(((3-1)/2))+k, j-(((3-1)/2))+l]
                                                        
            if temp1[1,1] == np.max(temp1) and np.max(temp1)>=50:
                 
                 final_image1[i-1,j-1] = temp1[1,1]
    
    ind = li(final_image1, N)
    final_image[ind]=255
    #partb end
    
    
    h_n = matlab_style_gauss2D(shape=(16,16),sigma=8)
    temp1 = np.zeros([16,16])
    temp2 = np.zeros([16,16]).astype(int)
    temp3 = np.zeros([4,4])
    temp4 = np.zeros([4,4]).astype(int)
    hog = np.zeros([1,8])
    hog1 = np.zeros([1,8])
    hog_shift = np.zeros([1,8])
    kpd = np.zeros([N,128])
    kpd_1 = np.zeros([1,128])
    kpd_2 = np.zeros([N,128])
    p=0
    a=0
    for i in range(0,n1):
        for j in range(0,n2):
            if final_image[i,j]==255:
                
                hog[:,:]=0
                for k in range(0,16):
                    for l in range(0,16):
                        temp1[k,l] = mag[i-(7-k), j-(7-l)]*h_n[k,l]
                        temp2[k,l] = bins[i-(7-k), j-(7-l)]
                        hog[0,temp2[k,l]] = hog[0,temp2[k,l]] + temp1[k,l]
                
                index = np.argmax(hog)
                
            
                for i1 in range(0,4):
                    for j1 in range(0,4):
                        
                        for k1 in range(0,4):
                            for l1 in range(0,4):
                                temp3[k1,l1] = temp1[4*i1+k1,4*j1+l1]
                                temp4[k1,l1] = temp2[4*i1+k1,4*j1+l1]
                                hog1[0,temp4[k1,l1]] = hog1[0,temp4[k1,l1]] + temp3[k1,l1]
                        #bewakoof = hog1
                        for s in range(0,8):
                            hog_shift[0,s] = hog1[0,s-(8-index)]
                        kpd[a,p:8+p] = hog_shift
                        #bewakoof2 = hog_shift
                        hog1[0,:] = 0
                        p = p+8
                
                L2Norm = np.linalg.norm(kpd[a,:])       
                kpd_1=np.divide(kpd[a,:],L2Norm)            
                value_index = kpd_1 > 0.20 
                kpd_1[value_index] = 0.20        
                #Renormailse
                L2Norm_1 = np.linalg.norm(kpd_1)
                kpd_2[a,:]=np.divide(kpd_1,L2Norm_1)
                p=0
                a = a+1
    return kpd_2, ind
            
          