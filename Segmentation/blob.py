from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

img=np.load("./result.npy", allow_pickle=True)
print(img.shape)

img[img<0.2]=int(0)
img[img>=0.2]=int(255)
img=np.array(img).astype('uint8')


# convert image to grayscale image
#gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
ret,thresh = cv2.threshold(img,127,255,0)
 
# find contours in the binary image
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
for c in contours:
   # calculate moments for each contour
   M = cv2.moments(c)
 
   # calculate x,y coordinate of center
   cX = int(M["m10"] / M["m00"])
   cY = int(M["m01"] / M["m00"])
   print(cX, cY)
   print()
