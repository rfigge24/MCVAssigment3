import numpy as np
import cv2 as cv
import math

def makeColorHistogram(imgCords,frame):
    #initilizing the histogram with its bins:
    histogram = np.zeros(36,np.float32)  
    #change the colorspace to HSV:
    hsvImg = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #fill the bins:
    for imgX,imgY in imgCords:
        if imgX < hsvImg.shape[1] and imgY < hsvImg.shape[0]:
            color = hsvImg[imgY,imgX]
            histogram[math.floor(color[0]/5)] += 1   
    #normalizing the histogram:
    totalsum = np.sum(histogram)
    histogram = histogram / totalsum

    return histogram


def compareColorHistograms(hist1, hist2):
    #uses chi-squared distance:
    distance = 0
    for h1,h2 in zip(hist1,hist2):
        if not (h1+h2) == 0:
            distance += (h1 - h2)**2 / (h1 + h2)
    return distance
