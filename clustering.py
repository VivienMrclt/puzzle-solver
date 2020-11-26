import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter

def gauss(image, sig):
    img = np.zeros(image.shape)
    if len(image.shape) == 3:
        for i in range(3):
            img[:, :, i] = gaussian_filter(image[:, :, i], sigma=sig)
    else:
        img[:, :] = gaussian_filter(image[:, :], sigma=sig)
    return img


img = cv.imread('truth.jpg')
img = gauss(img, 4)

Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv.imshow('res2',res2)
cv.waitKey(0)
cv.destroyAllWindows()