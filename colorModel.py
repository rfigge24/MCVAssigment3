import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def makeColorHistogram(imgCords,frame):
    """
    This function creates normalized histograms from a list of image coordinates and an image.
    These histograms are based on the full hue range.
    """
    
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

    #This code can be used to show the histogram:
    
    #plt.plot(hist,color = 'b')
    #plt.xlim([0,256])
    #plt.ylim([0,1])
    #plt.show()

    return hist


def compareColorHistograms(hist1, hist2):
    """
    This function gets the Chi-Squared distance of two histograms.
    """

    distance = cv.compareHist(hist1,hist2,cv.HISTCMP_CHISQR)
    return distance
