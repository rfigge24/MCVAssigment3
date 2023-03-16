import numpy as np
from collections import defaultdict
import bgSubstraction as bs
import cv2 as cv
import os
import cluster
import colorModel as cm


prevImg = [None,None,None,None]
FrameNr = 0
MaxFrameNr = None

personCenters = [[],[],[],[]]                                                                       #TODO: save the clustercenters that correspond to each person in this array
personColorModelstwoviews = [[None,None,None,None],[None,None,None,None]]                                                          #TODO: load the offline person colormodels into this list for comparison use
currentColorModelFrame1 = None         
currentColorModelFrame2 = None                                                              



#imgpoint,c to voxelcoord lookup table:
imgp_Cam2VoxelTable = defaultdict(list)                     #cams are 1-4
#Voxpt, c to pixel lookup table:
voxp_Cam2PixelTable = dict()                                #cams are 1-4
# contains for each voxel if it is forground for each cam:
voxelForgroundTable = np.zeros((170,235,100,4))             #cams are 0-3                                             


def buildVoxelLookupTable():
    """
    This function builds the voxel table, a dictionary that maps (imgX,imgY,Cameranr) --> list of (voxX,voxY,voxZ)
    Made such that we dont have to loop over all voxels again when there is a frame update.
    """

    global imgp_Cam2VoxelTable
    global voxp_Cam2PixelTable

    print("please wait while the voxel lookuptable is generated!")

    # get thet voxelCoords of the complete grid:
    voxelGrid = np.zeros((170,235,100))  
    voxelCoords = np.column_stack(np.where(voxelGrid == 0))
    # times 20 because voxels have size 20mm*20mm*20mm, plus 10 to get the voxel center:
    voxelCenterWorldCoords =20 * voxelCoords + np.array((-690,-690,10))

    # reset the voxelgrid:              
    voxelGrid = np.zeros((170,235,100))        

    for c in range(1,5):
        # get the camera parameters from the specific camera:
        path = f'data/cam{c}/config.xml'
        r = cv.FileStorage(path, cv.FileStorage_READ)
        tvecs = r.getNode('CameraTranslationVecs').mat()
        rvecs = r.getNode('CameraRotationVecs').mat()
        mtx = r.getNode('CameraIntrinsicMatrix').mat()
        dist = r.getNode('DistortionCoeffs').mat()
        r.release()
        # project the voxel
        Imgpts, jac = cv.projectPoints(np.float32(voxelCenterWorldCoords), rvecs, tvecs, mtx, dist)
        # reshape Imgpts to shape (x,2) and rounding pixel coords to nearest integer:
        Imgpts = np.int32(np.rint(Imgpts.reshape((-1,2))))
        # add to table:
        for imgCord,voxCord in zip(Imgpts,voxelCoords):
            imgX,imgY =imgCord
            voxX,voxY,voxZ = voxCord
            imgp_Cam2VoxelTable[(imgX,imgY,c)].append(voxCord)
            voxp_Cam2PixelTable[(voxX,voxY,voxZ,c)] = imgCord

    print("generation done!")


def initilizeVoxels(offline = False):                          #in offline mode it returns the color histograms for each cluster
    """
    This function will setup the voxel grid according to the first frame only.
    for each camera the background of the frame gets subtracted and then if a pixel is forground each voxel,
    that corresponds to that pixel will be set to be forground for that camera.

    All coordinates of voxels that are on in all cameras will get added to a list that will be returned.
    """
    global MaxFrameNr
    global FrameNr
    global prevImg
    global voxelForgroundTable
    global currentColorModelFrame1
    global currentColorModelFrame2
    #reset framenr and voxel forground table:
    voxelForgroundTable = np.zeros((170,235,100,4))   

    for c in range(1,5):
        path = os.path.abspath(f'data/cam{c}/video.avi')
        vid = cv.VideoCapture(path)
        MaxFrameNr = vid.get(cv.CAP_PROP_FRAME_COUNT)
        vid.set(cv.CAP_PROP_POS_FRAMES, FrameNr)
        succes, img = vid.read()
#-------------------------------------------------------------------------------------------------------------
        if succes and c == 2:
            currentColorModelFrame1 = img.copy()
        if succes and c == 3:
            currentColorModelFrame2 = img.copy()
#---------------------------------------------------------------------------------------------------------------
        if succes:
            model = None
            if c == 1:
                model = bs.model1
            elif c==2:
                model = bs.model2
            elif c==3:
                model = bs.model3
            elif c==4:
                model = bs.model4

            img = bs.subtractBackground(img, model, 0)
            prevImg[c-1] = img

            for x in range(img.shape[1]):
                for y in range(img.shape[0]):
                    if img[y,x] == 255:
                        for voxCord in imgp_Cam2VoxelTable[(x,y,c)]:
                            Vx, Vy, Vz = voxCord
                            voxelForgroundTable[Vx,Vy,Vz,c-1] = 1
        vid.release()
    # get indices of voxels that are on in all cameras:
    allOn = np.array([1,1,1,1])
    indices = np.where((voxelForgroundTable == allOn).all(axis=3))
    indices = np.column_stack((indices[0], indices[2], -1 * indices[1]))

    #correcting the coordinate offsets:
    indices = indices - np.array((34,0,-34))
#--------------------------------------------------------------------------------------------------------------------------
    #cluster the voxels:
    voxelLabels, centroids = cluster.clusterVoxels(np.float32(indices), 4)

    #create list of voxel coords lists corresponding to the cluster they belong to:
    clusteredVoxelLists = splitClusteredVoxelCoords(indices,voxelLabels, 4)

    #make a colorHistogram for each cluster of voxels looking from camera 2:
    clusterHistogramstwoviews = [[None,None,None,None],[None,None,None,None]]
    for v in range(2):
        for i,clusvoxlist in enumerate(clusteredVoxelLists):
            uniqueImageCords = getUniqueImageCords(clusvoxlist, v + 2, 30)
            if v == 0:
                hist = cm.makeColorHistogram(uniqueImageCords, currentColorModelFrame1)
            if v == 1: 
                hist = cm.makeColorHistogram(uniqueImageCords, currentColorModelFrame2)
            clusterHistogramstwoviews[v][i] = hist
    
    #if offline mode return the histograms:
    if offline:
        return clusterHistogramstwoviews

    #compare the clusterHistograms to the offline ones and assign each cluster to a single person:
    personLabels = assignPersons2Clusters(clusterHistogramstwoviews)

    #add the the cluster center to that persons list in personCenters
    for centroid, pers in zip(centroids, personLabels):
        personCenters[pers].append(centroid)
    
    #order the clusteredvoxellists in order of the persons they are assigned to
    #and make a list of colors for the voxels corresponding to the assigned person:
    voxelLists = [None,None,None,None]
    colorLists = [None,None,None,None]
    colors = [[0,0,0],[255,0,0],[0,255,0],[0,0,255]] 
    for voxelList, persNr in zip(clusteredVoxelLists, personLabels):
        voxelLists[persNr] = np.array(voxelList)
        colorLists[persNr] = np.tile(colors[persNr],(len(voxelList),1))
    
    #concatenate all lists:
    voxelList2 = np.concatenate((np.concatenate((np.concatenate((voxelLists[0],voxelLists[1])),voxelLists[2])),voxelLists[3]))
    colorList2 = np.concatenate((np.concatenate((np.concatenate((colorLists[0],colorLists[1])),colorLists[2])),colorLists[3]))

#------------------------------------------------------------------------------------------------------------------------------
    # update frame nr:
    FrameNr += 5

    return voxelList2, colorList2


def updateVoxels():
    """
    This function updates the voxels according to the previous frame and the next frame.
    It does so by getting the XOR of the two frames forground masks and then only updating 
    the voxels where visibility has been changed.
    This should save time because we do not loop over every voxel.

    All coordinates of voxels that are on in all cameras will get added to a list that will be returned.
    """

    global FrameNr
    global prevImg
    global voxelForgroundTable
    global currentColorModelFrame1
    global currentColorModelFrame2

    for c in range(1,5):
        path = os.path.abspath(f'data/cam{c}/video.avi')
        vid = cv.VideoCapture(path)
        vid.set(cv.CAP_PROP_POS_FRAMES, FrameNr)
        succes, img = vid.read()

        #-------------------------------------------------------------------------------------------------------------
        if succes and c == 2:
            currentColorModelFrame1 = img.copy()
        if succes and c == 3:
            currentColorModelFrame2 = img.copy()
        #---------------------------------------------------------------------------------------------------------------

        if succes:
            model = None
            if c == 1:
                model = bs.model1
            elif c==2:
                model = bs.model2
            elif c==3:
                model = bs.model3
            elif c==4:
                model = bs.model4

            currentImg = bs.subtractBackground(img, model, 0)
            
            # getting the pixels that have changed:
            changes = cv.bitwise_xor(prevImg[c-1],currentImg)
            prevImg[c-1] = currentImg
            
            # getting coords of changed pixels
            changedCoords = np.column_stack(np.where(changes == 255))
            # updating the voxelForgroundTable
            for imgCord in changedCoords:
                Iy,Ix = imgCord
                if currentImg[Iy,Ix] == 255:
                    for voxelCord in imgp_Cam2VoxelTable[Ix,Iy,c]:
                        Vx,Vy,Vz = voxelCord
                        voxelForgroundTable[Vx,Vy,Vz,c-1] = 1
                elif currentImg[Iy,Ix] == 0:
                    for voxelCord in imgp_Cam2VoxelTable[Ix,Iy,c]:
                        Vx,Vy,Vz = voxelCord
                        voxelForgroundTable[Vx,Vy,Vz,c-1] = 0
        vid.release()

    # getting all the voxel coords that are on:
    allOn = np.array([1,1,1,1])
    indices = np.where((voxelForgroundTable == allOn).all(axis=3))
    indices = np.column_stack((indices[0], indices[2], -1 * indices[1]))
    # updating frame nr:
    FrameNr += 5

    #correcting the coordinate offsets:
    indices = indices - np.array((34,0,-34))

    #--------------------------------------------------------------------------------------------------------------------------
    #cluster the voxels:
    voxelLabels, centroids = cluster.clusterVoxels(np.float32(indices), 4)

    #create list of voxel coords lists corresponding to the cluster they belong to:
    clusteredVoxelLists = splitClusteredVoxelCoords(indices,voxelLabels, 4)

    #make a colorHistogram for each cluster of voxels looking from camera 2:
    clusterHistogramstwoviews = [[None,None,None,None],[None,None,None,None]]
    for v in range(2):
        for i,clusvoxlist in enumerate(clusteredVoxelLists):
            uniqueImageCords = getUniqueImageCords(clusvoxlist, v + 2,30)
            if v == 0:
                hist = cm.makeColorHistogram(uniqueImageCords, currentColorModelFrame1)
            if v == 1: 
                hist = cm.makeColorHistogram(uniqueImageCords, currentColorModelFrame2)
            clusterHistogramstwoviews[v][i] = hist

    #compare the clusterHistograms to the offline ones and assign each cluster to a single person:
    personLabels = assignPersons2Clusters(clusterHistogramstwoviews)

    #add the the cluster center to that persons list in personCenters
    for centroid, pers in zip(centroids, personLabels):
        personCenters[pers].append(centroid)
    
    #order the clusteredvoxellists in order of the persons they are assigned to
    #and make a list of colors for the voxels corresponding to the assigned person:
    voxelLists = [None,None,None,None]
    colorLists = [None,None,None,None]
    colors = [[0,0,0],[255,0,0],[0,255,0],[0,0,255]] 
    for voxelList, persNr in zip(clusteredVoxelLists, personLabels):
        voxelLists[persNr] = np.array(voxelList)
        colorLists[persNr] = np.tile(colors[persNr],(len(voxelList),1))
    
    #concatenate all lists:
    voxelList2 = np.concatenate((np.concatenate((np.concatenate((voxelLists[0],voxelLists[1])),voxelLists[2])),voxelLists[3]))
    colorList2 = np.concatenate((np.concatenate((np.concatenate((colorLists[0],colorLists[1])),colorLists[2])),colorLists[3]))

#------------------------------------------------------------------------------------------------------------------------------
    return voxelList2, colorList2

#Newcode:

def assignPersons2Clusters(twoviewHists):
    global personColorModelstwoviews
    personLabels = [None,None,None,None]

    assignmentfromeachview = np.zeros((4,2),np.int64)
    personsalreadychosen = [False, False, False, False]

    for v, personColorModels in enumerate(personColorModelstwoviews):
        dist2Person = [None,None,None,None]
        for i,cHist in enumerate(twoviewHists[v]):
            for j, pmodel in enumerate(personColorModels):
                dist2Person[j] = cm.compareColorHistograms(pmodel, cHist)
            argsortedDist2Cluster = np.argsort(dist2Person)
            assignmentfromeachview[i,v] = argsortedDist2Cluster[0]


    for p in range(4):
        p_predicted_clusters_view1 = np.where(assignmentfromeachview[:,0] == p)
        if p_predicted_clusters_view1[0].shape[0] == 1:
            personLabels[p_predicted_clusters_view1[0][0]] = p
            personsalreadychosen[p] = True
        
        elif p_predicted_clusters_view1[0].shape[0] > 1:
            p_predicted_clusters_bothviews = np.where(assignmentfromeachview[p_predicted_clusters_view1[0],1] == p)
            if p_predicted_clusters_bothviews[0].shape[0] > 0:
                personLabels[p_predicted_clusters_view1[0][p_predicted_clusters_bothviews[0][0]]] = p
                personsalreadychosen[p] = True
            else:
                personLabels[p_predicted_clusters_view1[0][0]] = p
                personsalreadychosen[p] = True
    
    for i in range(4):
        if personLabels[i] == None:
            if personsalreadychosen[assignmentfromeachview[i][1]] == False:
                personLabels[i] = assignmentfromeachview[i][1]
                personsalreadychosen[assignmentfromeachview[i][1]] = True
            else:
                for j in range(4):
                    if personsalreadychosen[j] == False:
                        personLabels[i] = j
                        personsalreadychosen[j] = True
                        break


 
    return personLabels
                

def loadPersonColorModels():
    r = cv.FileStorage('data/PersonColorModels.xml', cv.FILE_STORAGE_READ)
    for v in range(2):    
        for i in range(4):
            personColorModelstwoviews[v][i] = r.getNode(f"ColorModelPerson{v}{i}").mat().ravel()
    r.release()


def splitClusteredVoxelCoords(voxelCords, voxelLabels, nrOfClusters):
    ClusterLists = []
    for i in range(nrOfClusters):
        ClusterLists.append([])
    for voxC, label in zip(voxelCords,voxelLabels.ravel()):
        ClusterLists[label].append(voxC)
    
    return ClusterLists


def getUniqueImageCords(voxels, camNr, threshold):
    global voxp_Cam2PixelTable
    imageCoords = set()
    for voxX,voxY,voxZ in voxels:
        if ((voxX,-1*voxZ,voxY,camNr) in voxp_Cam2PixelTable.keys()) and voxY > threshold:
            imageCoords.add(tuple(voxp_Cam2PixelTable[(voxX,-1*voxZ,voxY,camNr)]))

    return list(imageCoords)





#------------------------------Construction of the Lookup table when script is ran or imported:----------------------------------
buildVoxelLookupTable()
#--------------------------------------------------------------------------------------------------------------------------------
#loadPersonColorModels()
#FrameNr = 1200
#initilizeVoxels()
#while FrameNr < 10000:
    #print(FrameNr)
    #updateVoxels()
    
