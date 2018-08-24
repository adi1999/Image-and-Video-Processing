import cv2
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

frame1 = cv2.imread('1.jpg',0)             
p0 = cv2.goodFeaturesToTrack(frame1,100,0.01,1,useHarrisDetector=True)


frame4 = cv2.imread('4.jpg',0)        
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
p1, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame4, p0, None, **lk_params)

goodp1 = np.array(p1 , dtype='int')
goodp0 = np.array(p0, dtype='int')

#Plotting 
rows, cols = frame1.shape                                  
corrImage = np.zeros((rows,(2*cols)), dtype = np.uint8)       
corrImage[0:rows , 0:cols] = frame1[:,:]
corrImage[0:rows , cols:(2*cols)] = frame4[:,:]
pairs = zip(goodp0,goodp1)                           
u,v=  [],[]                                            
x,y = [],[]
for (old,new) in pairs:
    a,b = old.ravel()                                          
    c,d = new.ravel()                                          
    cv2.line(corrImage, (a,b),(c+cols,d), [0,255,0], 1)
    #Taking a patch at the center of the image for better homography as only 8 points are being considered
    if a > (cols/3) and b>(rows/3):
        u.append(a)
        v.append(b)
        x.append(c)
        y.append(d)

#setting up the equation in matric form        
H = np.zeros((2*len(u),8), dtype = np.float32)
midIndex = len(u)
H[0:midIndex,0] = 1
H[0:midIndex,1] = u
H[0:midIndex,2] = v
H[0:midIndex,6] = -1 * np.array(u)*np.array(x)
H[0:midIndex,7] = -1 * np.array(v)*np.array(x)
H[midIndex:(2*midIndex),3] = 1
H[midIndex:(2*midIndex),4] = u
H[midIndex:(2*midIndex),5] = v
H[midIndex:(2*midIndex),6] = -1 * np.array(u)*np.array(y)
H[midIndex:(2*midIndex),7] = -1 * np.array(v)*np.array(y)

#Getting x and y co-ordinates
q = np.zeros((2*midIndex,1), dtype = np.float32)
q[0:midIndex,0] = x
q[midIndex:(2*midIndex),0] = y
#Least squares formula
HT = np.transpose(H)
p_invq= inv(HT.dot(H)).dot(HT) 
a = p_invq.dot(q)
#Arranging the parameters in correct format
M = np.zeros((3,3), dtype = np.float32)
M[0,0] = a[1]
M[0,1] = a[2]
M[0,2] = a[0]
M[1,0] = a[4]
M[1,1] = a[5]
M[1,2] = a[3]
M[2,0] = a[6]
M[2,1] = a[7]
M[2,2] = 1

WI = cv2.warpPerspective(frame4, M, (cols,rows))
cv2.imwrite('warpedImage.png',WI)
plt.imshow(corrImage,cmap=plt.cm.gray) 
plt.figure()
plt.imshow(WI,cmap = plt.cm.gray ) 
fgbg = cv2.BackgroundSubtractorMOG()             
fgmask1 = fgbg.apply(frame1)                           
fgmask4 = fgbg.apply(WI)
plt.figure()
plt.imshow(fgmask4,cmap = plt.cm.gray )                              
cv2.imwrite('Binary_image.png',fgmask4)    
            



