import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def clusterVoxels(voxels,nrOfClusters):
    """
    This function takes care of the kmeans clustering of the voxels by only looking at x and y and discarding variabilty in height
    """
    #getting rid of the height axis:
    voxels2d = voxels[:,[0,2]]
    # the convergence criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # Set flags (Just to avoid line break in the code)
    flags = cv.KMEANS_PP_CENTERS
    # Apply KMeans
    ret,voxelLabels,centroids = cv.kmeans(voxels2d,nrOfClusters,None,criteria,10,flags)

    return  voxelLabels,centroids


















"""
#This code can be pasted above the return to visualize the clustering!

# visualization:
    A = voxels2d[voxelLabels.ravel()==0]
    B = voxels2d[voxelLabels.ravel()==1]
    C = voxels2d[voxelLabels.ravel()==2]
    D = voxels2d[voxelLabels.ravel()==3]
    

    # Now plot 'A' in red, 'B' in blue, 'centers' in yellow
    plt.scatter(A[:,0],A[:,1])
    plt.scatter(B[:,0],B[:,1],c = 'r')
    plt.scatter(C[:,0],C[:,1],c= 'brown')
    plt.scatter(D[:,0],D[:,1],c = 'g')
    
    plt.scatter(centroids[:,0],centroids[:,1],s = 80,c = 'y', marker = 's')
    plt.xlabel('Height'),plt.ylabel('Weight')
    plt.show()
"""