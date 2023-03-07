import cv2 as cv
import numpy as np
import glob

squaresize = 0
img = None
objp = None

clicks = list()
def click_event(event, x, y, flags, params):
    """
    This function is a mouse event handler.
    The mouslocation gets saved when the user clicks on the image.
    It also draws a circle on the clicked coordinates.
    """
    global img
    if event == cv.EVENT_LBUTTONDOWN:
        drawCircle(x, y, True)
        clicks.append([x,y])


def drawCircle(x, y, show=False):
    """
    This function draws a circle with a cross in the middle on the x and y location that are given.
    There is an option to show the image in the display window.
    """
    global img
    cv.line(img, (x, y-4), (x, y+4), (0,0,255), 1)
    cv.line(img, (x-4, y), (x+4, y), (0,0,255), 1)
    cv.circle(img, (x, y), 8, (0,0,255), 1)
    if show:
        cv.imshow('img', img)

def drawAxis(img, corners, axisImgpts):
    """
    This function draws the axis lines and the cube onto a given image.
    The transformed coordinates of the outer corners of the axis lines are used here.
    """
    # drawing the axis:
    corner = tuple(corners[0].ravel().astype(int))
    img = cv.line(img, corner, tuple(axisImgpts[0].ravel().astype(int)), (255,0,0), 2)
    img = cv.line(img, corner, tuple(axisImgpts[1].ravel().astype(int)), (0,255,0), 2)
    img = cv.line(img, corner, tuple(axisImgpts[2].ravel().astype(int)), (0,0,255), 2)

    return img


def addAxis2frame(img, mtx, dist, rvecs, tvecs, corners):
    """
    This function takes care of the calculation part of adding the cube and axis lines to a frame.
    The camera rotation and transformation is calculated using the 3D obj points and 2d img points of the Chessboard.
    Then the 3D points of the axis lines and the cube are transformed using these parameters.
    It then calls the drawAxisAndCube function with these newly obtained 2d points of the cube and axis points.
    """

    # 3D points of the axis line-ends:
    axis = np.float32([[3*squaresize,0,0], [0,3*squaresize,0], [0,0,3*squaresize]]).reshape(-1,3)

    # project 3D points to image plane:
    axisImgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

    img = drawAxis(img, corners, axisImgpts)

    return img

def manualCornerDetection(size): 
    """
    This function is used to manually select the chessboard corners by clicking on the outer corners.
    You need to select the outer corners from left to right, top to bottom.
    The grid is then interpolated by first getting the perspective transform of the outer corner points and
    then perspective transforming the 3d grid points using the obtained perspective transformation matrix.

    !before selecting the 4 corners you need to press a key. press 'r' to reject a image and 
    press any other key to continue selection corners!
    """
    global img
    ret = True
    clicks.clear()
    cv.imshow('img', img)

    # needs to know if the image needs to be rejected or not
    cv.imshow('img',img)
    k = cv.waitKey(-1)
    if(k == ord('r')):
        return False, None

    cv.setMouseCallback('img', click_event, img)
    
    # we need 4 corners, so wait...
    while len(clicks) < 4:
        cv.waitKey(25)

    # 4 outer corners of checkerboard
    checkCorners = [[0, 0], [(size[0]-1)*squaresize, 0], [0, (size[1]-1)*squaresize], [(size[0]-1)*squaresize, (size[1]-1)*squaresize]]

    # do math magic
    persMx = cv.getPerspectiveTransform(np.float32(checkCorners), np.float32(clicks))

    # get (2d!!) checkerboard array in correct shape
    chkPts = objp[:,0:2]
    chkPts = chkPts.reshape(size[0]*size[1], 1, 2)

    # apply math magic
    persCheck = cv.perspectiveTransform(chkPts, persMx)

    # reset mouse callback
    cv.setMouseCallback('img', lambda *args : None)

    return ret, persCheck


def loadchessBoardFacts(filename):
    """
    This function reads the chessboard dimensions and size from an XML file where it was saved in.
    """
    r = cv.FileStorage(filename, cv.FILE_STORAGE_READ)    
    width = r.getNode("CheckerBoardWidth").real()
    height = r.getNode("CheckerBoardHeight").real()
    squaresize = r.getNode("CheckerBoardSquareSize").real()
    r.release()

    return (int(width),int(height)),int(squaresize)

def loadIntrinsics(filename):
    """
    This function reads the intrinsic mtx and distortion from the already existing config file
    and returns them.
    """
    r = cv.FileStorage(filename, cv.FILE_STORAGE_READ)    
    mtx = r.getNode('CameraIntrinsicMatrix').mat()
    dist = r.getNode('DistortionCoeffs').mat()
    r.release()
    
    return mtx,dist



def cameraExtrinsicCalibration(size, imagefnames, mtx, dist):
    """
    This function calculates the camera extrinsics using the mtx and distortion 
    that are already obtained the extrinsics get calibrated.
    the camera extrinsics will be returned.
    """
    global img
    global objp

    # prepare object points, a grid of the dimensions that are given by size
    objp = np.zeros((size[0]*size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2)

    # scale the objp to the size of the chessboard squares in mm:
    objp = objp * squaresize
    
    # read the first image that is in the directory:
    img = cv.imread(imagefnames[0])

    # Find the chess board corners
    ret, corners = manualCornerDetection(size)
        
    # Draw and display the corners
    cv.drawChessboardCorners(img, size, corners, ret)
    cv.imshow('img', img)
    cv.waitKey(250)

    # solve the rotation and translation:
    ret, rvecs, tvecs = cv.solvePnP(objp, corners, mtx, dist)

    return corners, rvecs, tvecs


def getAndSaveParameterConfig(size, camnum):
    """
    This function calibrates both the intrinsics and extrinsics or a camera for which the cam number is provided.
    These then get saved in the camera folder in a XML file for later use.
    """
    camfolder = 'data/cam' + str(camnum) + '/'
    extrinsicImgNames = glob.glob(camfolder + 'extrinsic/*.jpg')

    # getting Intrinsic parameters and Distortion:
    mtx, dist = loadIntrinsics(camfolder+'config.xml')

    # getting The extrinsic parameters:
    corners, rvecs, tvecs = cameraExtrinsicCalibration(size, extrinsicImgNames, mtx, dist)

    img = cv.imread(extrinsicImgNames[0])
    img2 = addAxis2frame(img, mtx, dist, rvecs, tvecs, corners)
    cv.imshow('img', img2)
    cv.waitKey(-1)

    # saving parameters to file:
    s = cv.FileStorage(camfolder+'config.xml', cv.FileStorage_WRITE)
    s.write('CameraIntrinsicMatrix', mtx)
    s.write('DistortionCoeffs', dist)
    s.write('CameraRotationVecs', rvecs)
    s.write('CameraTranslationVecs', tvecs)
    s.release()



if __name__ == "__main__":
    chessboardshape, squaresize = loadchessBoardFacts("data/checkerboard.xml")
    for i in range(1,5):
        getAndSaveParameterConfig(chessboardshape, i)

