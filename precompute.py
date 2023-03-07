import os
import glob
import cv2 as cv
import random

def captureExtrinsics(camnum):
    """
    This function captures and saves one random frame of the extrinsic video.
    """
    camfolder = 'data/cam' + str(camnum) + '/'
    vidpath = os.path.abspath(camfolder + 'checkerboard.avi')
    exportfolder = os.path.abspath(camfolder + 'extrinsic')
    if not os.path.exists(exportfolder):
        os.mkdir(exportfolder)
    else:
        ims = glob.glob(exportfolder + '/*')
        for im in ims:
            os.remove(im)

    vid = cv.VideoCapture(vidpath)
    frames = vid.get(cv.CAP_PROP_FRAME_COUNT)
    randomFrame = random.randint(0, frames)
    vid.set(cv.CAP_PROP_POS_FRAMES, randomFrame)

    success, image = vid.read()
    if success:
        cv.imshow("Camera " + str(camnum), image)
        cv.waitKey(100)
        cv.imwrite(exportfolder + '/' + "0" + '.jpg', image)
    

if __name__ == "__main__":
    for i in range(1, 5):
        captureExtrinsics(i)