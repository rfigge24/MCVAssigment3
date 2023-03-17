import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

def makeColorHistogram(imgCords,frame):
    
    hsvImg = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2],np.uint8)
    for x,y in imgCords:
        mask[y,x] = 255
    
    #cv.imshow('mask', mask)

    hist = cv.calcHist([hsvImg], [0], mask, [180],[0,180])
    
    #normalize histogram:
    sum = np.sum(hist)
    if sum > 0:
        hist = hist / sum

    #plt.plot(hist,color = 'b')
    #plt.xlim([0,256])
    #plt.ylim([0,1])
    #plt.show()

    return hist


def compareColorHistograms(hist1, hist2):
    distance = cv.compareHist(hist1,hist2,cv.HISTCMP_CHISQR)
    return distance
