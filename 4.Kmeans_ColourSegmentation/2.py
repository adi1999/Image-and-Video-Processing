# -*- coding: utf-8 -*-
"""


@author: Aditya Dhookia
"""



print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

n_colors = 64

# Load the Summer Palace photo
china = load_sample_image("china.jpg")

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
china = np.array(china, dtype=np.float64) / 255
wt  = 1




# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(china.shape)
#image_array1 = np.reshape(china, (w * h, 3))


x_corr = np.zeros([w,h])
y_corr = np.zeros([w,h])

for i in range(0,w):
    for j in range(0,h):
        x_corr[i,j] = (np.float64(i)/w)*1*wt
        y_corr[i,j] = (np.float64(j)/h)*1*wt  

new_image = np.zeros([w,h,5]) #
new_image[:,:,0:3] = china
new_image[:,:,3] = x_corr
new_image[:,:,4] = y_corr
                     
image_array = np.reshape(new_image, (w * h, 5))


print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]

kmeans = KMeans(n_clusters=n_colors, init='k-means++',verbose=1,random_state=0).fit(image_array_sample)
     
    
print("done in %0.3fs." % (time() - t0))
# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))


codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array,
                                          axis=0)
print("done in %0.3fs." % (time() - t0))
#codebook_random1 = shuffle(image_array1, random_state=0)[:n_colors + 1]


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    #d = 3
    image = np.zeros((w, h, 3))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image
image_kmeans = recreate_image(kmeans.cluster_centers_[:,0:3], labels, w, h)
image_random = recreate_image(codebook_random[:,0:3], labels_random, w, h)
# Display all results, alongside original image

plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(china)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(image_kmeans)

plt.figure(3)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, Random)')
plt.imshow(image_random)
plt.show()

#a = labels_random.reshape(427,640)
#plt.imshow(a)
