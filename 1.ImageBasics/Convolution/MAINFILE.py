import cv2
import numpy as np
from  FUNC_2Dconvolution import conv2D
from matplotlib import pyplot as plt
# input

img = cv2.imread('airplane.png',0)

H1 = np.float64(np.array([[1.0/16,2.0/16,1.0/16],[2.0/16,4.0/16,2.0/16],[1.0/16,2.0/16,1.0/16]]))
H2 = np.float64(np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]))
H3 = np.float64(np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))


n=input("Enter 1 for H1, 2 for H2 and 3 for H3 (default is H1)")

if n==1:
    H=H1
elif n==2:
    H=H2
elif n==3:
    H=H3
else:
    H=H1 ##Default filter

#H = np.float64(np.array([[1.0/16,2.0/16,1.0/16],[2.0/16,4.0/16,2.0/16],[1.0/16,2.0/16,1.0/16]]))
#H = np.float64(np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]))
#1H = np.float64(np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))

#Calling the manual 2D convolution
filtered_image = conv2D(img,H)
#######Plotting
#plt.imshow(filtered_image,cmap=plt.cm.gray)
cv2.imwrite('Filtered_image.png',filtered_image)

print ("DONE!!!!!")
print ("Image file saved as Filtered_image.png")
##### Calculating FFTs


FFT_original = np.abs(np.fft.fft2(np.float64(img)))
FFT_filtered = np.abs(np.fft.fft2(filtered_image))
FFT_filter = np.abs(np.fft.fft2(np.pad(H,((254,254),(254,254)),'constant',constant_values=(0,0))))               #Change variable according to H1, H2, H3

fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
ax1.imshow(img,cmap=plt.cm.gray)
ax1.set_title('original image')
ax2 = fig1.add_subplot(122)
ax2.imshow(np.uint8(filtered_image),cmap=plt.cm.gray)
ax2.set_title('Filtered image')


fig2 = plt.figure()
ax1 = fig2.add_subplot(121)
ax1.imshow(np.fft.fftshift(np.log(FFT_original+1)),cmap=plt.cm.gray)
ax1.set_title('original image')
ax2 = fig2.add_subplot(122)
ax2.imshow(np.fft.fftshift(np.log(FFT_filtered+1)),cmap=plt.cm.gray)
ax2.set_title('Filtered image')


fig3 = plt.figure()
plt.imshow(np.fft.fftshift(np.log(FFT_filter+1)),cmap=plt.cm.gray)

