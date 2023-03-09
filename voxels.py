import numpy as np
from collections import defaultdict
import bgSubstraction as bs
import cv2 as cv
import os

prevImg = [None,None,None,None]
FrameNr = 0                             #goes in steps of 12 frames each update

#impt,c to voxelcoord lookup table:
imgp_Cam2VoxelTable = defaultdict(list)
# contains for each voxel if it is forground for each cam:
voxelForgroundTable = np.zeros((170,235,100,4))                                                     


def buildVoxelLookupTable():
    """
    This function builds the voxel table, a dictionary that maps (imgX,imgY,Cameranr) --> list of (voxX,voxY,voxZ)
    Made such that we dont have to loop over all voxels again when there is a frame update.
    """

    global imgp_Cam2VoxelTable

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

        # project the voxel
        Imgpts, jac = cv.projectPoints(np.float32(voxelCenterWorldCoords), rvecs, tvecs, mtx, dist)
        # reshape Imgpts to shape (x,2) and rounding pixel coords to nearest integer:
        Imgpts = np.int32(np.rint(Imgpts.reshape((-1,2))))
        # add to table:
        for imgCord,voxCord in zip(Imgpts,voxelCoords):
            x,y =imgCord
            imgp_Cam2VoxelTable[(x,y,c)].append(voxCord)

    print("generation done!")


def initilizeVoxels():
    """
    This function will setup the voxel grid according to the first frame only.
    for each camera the background of the frame gets subtracted and then if a pixel is forground each voxel,
    that corresponds to that pixel will be set to be forground for that camera.

    All coordinates of voxels that are on in all cameras will get added to a list that will be returned.
    """
    global FrameNr
    global prevImg
    global voxelForgroundTable
    #reset framenr and voxel forground table:
    voxelForgroundTable = np.zeros((170,235,100,4))   
    FrameNr = 0

    for c in range(1,5):
        path = os.path.abspath(f'data/cam{c}/video.avi')
        vid = cv.VideoCapture(path)
        vid.set(cv.CAP_PROP_POS_FRAMES, 0)
        succes, img = vid.read()

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

    # get indices of voxels that are on in all cameras:
    allOn = np.array([1,1,1,1])
    indices = np.where((voxelForgroundTable == allOn).all(axis=3))
    indices = np.column_stack((indices[0], indices[2], -1 * indices[1]))

    # update frame nr:
    FrameNr += 12

    return indices - np.array((34,0,-34))


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

    for c in range(1,5):
        path = os.path.abspath(f'data/cam{c}/video.avi')
        vid = cv.VideoCapture(path)
        vid.set(cv.CAP_PROP_POS_FRAMES, FrameNr)
        succes, img = vid.read()

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

    # getting all the voxel coords that are on:
    allOn = np.array([1,1,1,1])
    indices = np.where((voxelForgroundTable == allOn).all(axis=3))
    indices = np.column_stack((indices[0], indices[2], -1 * indices[1]))
    # updating frame nr:
    FrameNr += 12

    return indices - np.array((34,0,-34))



#------------------------------Construction of the Lookup table when script is ran or imported:----------------------------------
buildVoxelLookupTable()
#--------------------------------------------------------------------------------------------------------------------------------
#readLookupTable()
