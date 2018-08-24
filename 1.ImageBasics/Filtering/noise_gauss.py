import cv2
import numpy as np

from  FUNC_2Dconvolution import conv2D
from matplotlib import pyplot as plt
# input

img = cv2.imread('airplane.png',0)
img = np.float64(img)
mean = 0;
var = 0.01
sigma = var**0.5
a = np.random.normal(0,sigma,(img.shape))
max_im = np.max(img)
min_im = np.min(img)
image_norm = np.float64((img-min_im)/(max_im-min_im))
noisy_img = image_norm+a
max_im_n = np.max(noisy_img)
min_im_n = np.min(noisy_img)
noisy_img_n = np.round((noisy_img-min_im_n)*255/(max_im_n-min_im_n));

### ***** Use the below 3 average filters for lengths 3, 5 and 7 ****** IMPORTANT                      

H1 = np.float64(np.array([[1.0/9,1.0/9,1.0/9],[1.0/9,1.0/9,1.0/9],[1.0/9,1.0/9,1.0/9]]));
H2 = np.float64(np.array([[1.0/25,1.0/25,1.0/25,1.0/25,1.0/25],[1.0/25,1.0/25,1.0/25,1.0/25,1.0/25],[1.0/25,1.0/25,1.0/25,1.0/25,1.0/25],[1.0/25,1.0/25,1.0/25,1.0/25,1.0/25],[1.0/25,1.0/25,1.0/25,1.0/25,1.0/25]]))
H3 = np.float64(np.array([[1.0/49,1.0/49,1.0/49,1.0/49,1.0/49,1.0/49,1.0/49],[1.0/49,1.0/49,1.0/49,1.0/49,1.0/49,1.0/49,1.0/49],[1.0/49,1.0/49,1.0/49,1.0/49,1.0/49,1.0/49,1.0/49],[1.0/49,1.0/49,1.0/49,1.0/49,1.0/49,1.0/49,1.0/49],[1.0/49,1.0/49,1.0/49,1.0/49,1.0/49,1.0/49,1.0/49],[1.0/49,1.0/49,1.0/49,1.0/49,1.0/49,1.0/49,1.0/49],[1.0/49,1.0/49,1.0/49,1.0/49,1.0/49,1.0/49,1.0/49]]))



H=input("Enter 1 for average and 2 for gaussian (default is average)")
g=input("Enter 3 for 3X3, 5 for 5X5 and 7 for 7X7 (default is 3X3)")


if H==1:
        
    if g==3:
        H=H1
    elif g==5:
        H=H2
    elif g==7:
        H=H3
    else:
        H=H1 ##Default filter

elif H==2:
    
    ### ***** Gaussian filter, change shape values foe 3, 5 and 7 length ***** IMPORTANT
    shape=(g,g)
    
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    H = np.exp( -(x*x + y*y) / (2.*1*1) )
    H[ H < np.finfo(H.dtype).eps*H.max() ] = 0
    sumh = H.sum()
    
    if sumh != 0:
        H /= sumh
else:
    H = H1
    
    
#Calling the manual 2D convolution
filtered_image = conv2D(noisy_img_n,H)


fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
ax1.imshow(np.uint8(noisy_img_n),cmap=plt.cm.gray)
ax1.set_title('Noisy image')
ax2 = fig1.add_subplot(122)
ax2.imshow(np.uint8(filtered_image),cmap=plt.cm.gray)
ax2.set_title('Denoised image')


