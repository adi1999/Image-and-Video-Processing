# -*- coding: utf-8 -*-
"""


@author: Aditya Dhookia
"""

import cv2 
import numpy as np
from matplotlib import pyplot as plt 
from  PIL import Image

img1 = cv2.imread('BK_left.JPG',0)
img2 = cv2.imread('BK_right.JPG',0)
cv2.imwrite('right.png',img2)

detector = cv2.FeatureDetector_create("SIFT")
descriptor = cv2.DescriptorExtractor_create("SIFT")

skp1 = detector.detect(img1)
skp1 ,sd1 = descriptor.compute(img1,skp1)

skp2 = detector.detect(img2)
skp2 ,sd2 = descriptor.compute(img2,skp2)

#skp[element_index].pt for co-ordinates
#skp[element_index].size for scale of point

#for i in range(0,len(skp1)):
#    cv2.circle(img1, (int(skp1[i].pt[0]),int(skp1[i].pt[1])), int(skp1[i].size), (255,0,0), 1)
#plt.imshow(img1,cmap=plt.cm.gray)
#img1=np.uint8(img1)
#cv2.imwrite('skp1.JPG',img1)

# for min features
if int(sd1.shape[0]) > int(sd2.shape[0]):
    l= int(sd2.shape[0])
    m= int(sd1.shape[0])
else:
    l= int(sd1.shape[0])
    m=int(sd2.shape[0])

index_corr1 = np.zeros([2,m+1])
index_corr2 = np.zeros([2,m+1])            
#d = np.zeros([int(sd1.shape[0]),[int(sd1.shape[0])]])
d = np.zeros([l,m])  # Here rows=l feature points and m=columns to check all possible comb  
z=0


for p in range(0,l):
    for q in range(0,m):
        d[p,q] = np.linalg.norm ( sd1[p,:] - sd2[q,:] )
    dsorted = sorted(d[p,:])
    d1 = dsorted[0]
    d2 = dsorted[1]
    r = d1/d2
    if r<0.5:
        sm_ind = np.where( d[p,:]== d1)
        index_corr1[0,z] = int(skp1[p].pt[0])
        index_corr1[1,z] = int(skp1[p].pt[1])  
        index_corr2[0,z] = int(skp2[sm_ind[0][0]].pt[0])
        index_corr2[1,z] = int(skp2[sm_ind[0][0]].pt[1])
        z=z+1
#end

src = np.zeros([z,2]).astype(np.float32)
dest = np.zeros([z,2]).astype(np.float32)

for a in range(0,z): 
    src[a,0]=  index_corr1[0,a]    
    src[a,1]=  index_corr1[1,a]
    dest[a,0]= index_corr2[0,a]
    dest[a,1]= index_corr2[1,a]     

con_image = np.concatenate((img1,img2), axis=1)
for s in range(0,z):
    cv2.line(con_image,(int(src[s,0]),int(src[s,1])),(400+int(dest[s,0]),int(dest[s,1])),(255,255,255),1)

plt.imshow(con_image,cmap = plt.cm.gray) 
cv2.imwrite("con_image.png",con_image)   
#cv2.findHomography( src, dest, cv2.RANSAC)
# Calculate Homography
h, status = cv2.findHomography(src, dest,cv2.RANSAC)
# Warp source image to destination based on homography
img_out = cv2.warpPerspective(img2, h, (800,300), flags = cv2.WARP_INVERSE_MAP)
plt.imshow(img_out,cmap=plt.cm.gray)
cv2.imwrite('warped_right.png',img_out)
img_ext = np.zeros([300,800])
img_ext[:,0:400] = img1[:,:]
#img_ext[:,191:590] = img_out[:,201:600]
cv2.imwrite('left.png',img_ext)
background = Image.open("left.png")
overlay = Image.open("warped_right.png")

background = background.convert("L")
overlay = overlay.convert("L")

new_img = Image.blend(background, overlay, 0.5)
plt.imshow(new_img,cmap=plt.cm.gray)
new_img.save("new.png","PNG")
