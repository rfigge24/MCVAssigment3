import cv2 as cv
import numpy as np
import glob

def drawAxis(img, corners, axisImgpts):
    """
    This function draws the axis lines and the cube onto a given image.
    The transformed coordinates of the outer corners of the axis lines are used here.
    """
    # drawing the axis:
    corner = tuple([0,0])
    print(axisImgpts[4])
    img = cv.line(img, tuple(axisImgpts[0].ravel().astype(int)), tuple(axisImgpts[1].ravel().astype(int)), (255,0,0), 2)
    img = cv.line(img, tuple(axisImgpts[0].ravel().astype(int)), tuple(axisImgpts[2].ravel().astype(int)), (0,255,0), 2)
    img = cv.line(img, tuple(axisImgpts[0].ravel().astype(int)), tuple(axisImgpts[3].ravel().astype(int)), (0,0,255), 2)
    img = cv.line(img, tuple(axisImgpts[4].ravel().astype(int)), tuple(axisImgpts[1].ravel().astype(int)), (0,0,255), 2)
    img = cv.line(img, tuple(axisImgpts[4].ravel().astype(int)), tuple(axisImgpts[2].ravel().astype(int)), (0,0,255), 2)

    return img


def addAxis2frame(img, mtx, dist, rvecs, tvecs, corners):
    """
    This function takes care of the calculation part of adding the cube and axis lines to a frame.
    The camera rotation and transformation is calculated using the 3D obj points and 2d img points of the Chessboard.
    Then the 3D points of the axis lines and the cube are transformed using these parameters.
    It then calls the drawAxisAndCube function with these newly obtained 2d points of the cube and axis points.
    """

    # 3D points of the axis line-ends:
    axis = np.float32([[-700,-700,0],[2710,-700,0], [-700,4010,0], [-700,-700,2010], [2710,4010,0]]).reshape(-1,3)

    # project 3D points to image plane:
    axisImgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

    img = drawAxis(img, corners, axisImgpts)

    return img

def test():
    for c in range(1,5):
        # get the camera parameters from the specific camera:
        path = f'data/cam{c}/config.xml'
        r = cv.FileStorage(path, cv.FileStorage_READ)
        tvecs = r.getNode('CameraTranslationVecs').mat()
        rvecs = r.getNode('CameraRotationVecs').mat()
        mtx = r.getNode('CameraIntrinsicMatrix').mat()
        dist = r.getNode('DistortionCoeffs').mat()
        camfolder = 'data/cam' + str(c) + '/'
        extrinsicImgNames = glob.glob(camfolder + 'test/*.jpg')

        img = cv.imread(extrinsicImgNames[0])

        img = addAxis2frame(img, mtx,dist,rvecs,tvecs, None)
        cv.imshow('img', img)
        cv.waitKey(-1)

test()


