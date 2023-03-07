import cv2 as cv
import os
import numpy as np

def trainBackgroundModel(bgSubtractor, bgVidPath):
    """
    This function trains a BackgroundSubtractorMOG2 model on all frames of a background video.
    It does some preprocessing by applying a gaussian blur on the image to reduce noise.
    """
    
    vid = cv.VideoCapture(bgVidPath)
    nrOfFrames = vid.get(cv.CAP_PROP_FRAME_COUNT)

    for i in range(0,int(nrOfFrames)):
        vid.set(cv.CAP_PROP_POS_FRAMES, i)
        succes, img = vid.read()
        if succes:
            #preprossesing with a gaussian blur:
            img = cv.GaussianBlur(img,(3,3), 0)
            #training the background model:
            bgSubtractor.apply(img, None, -1)


def testBackgroundModel(bgSubtractor, fgVidPath, dilation):
    """
    This function is to test a trained background model on the frames of the forground video.
    For trying different values of thresholds and dilation this was very handy.

    Same as training the backgroundmodel we apply a gaussian blur before subtracting the background.
    after the subtraction we find the contour of the man with findcontours.
    and we draw this contour only on a all black image of the same size as the origional image.
    If needed there are "dilation" iterations of dilation performed on the final image.
    """

    vid = cv.VideoCapture(fgVidPath)
    nrOfFrames = vid.get(cv.CAP_PROP_FRAME_COUNT)

    for i in range(0,int(nrOfFrames)):
        vid.set(cv.CAP_PROP_POS_FRAMES, i)
        succes, img = vid.read()
        if succes:
            #preprossessing with a gaussian blur:
            blur = cv.GaussianBlur(img,(3,3), 0)
            #subtracting the background:
            fgImg = bgSubtractor.apply(blur, None, 0)
            
            #finding the contour:
            contours, hierarchy  = cv.findContours(fgImg, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
            list = [len(x) for x in contours]
            indexes = np.argsort(list,0)

            testimg = np.zeros_like(fgImg)
            fgImg = cv.drawContours(testimg, contours, indexes[-1] , (255,255,255), -1)

            kernel = np.ones((3,3),np.uint8) 
            fgImg = cv.dilate(fgImg,kernel, iterations = dilation)



            cv.imshow('bgImg', fgImg)
            cv.waitKey(10)

def subtractBackground(img, bgSubtractor, dilation):
    """
    This function performs the background subtranction on a given image with the given trained background model.
    It returns the forground mask.
    """

    #preprossessing with a gaussian blur:
    blur = cv.GaussianBlur(img,(3,3), 0)
    #subtracting the background:
    fgImg = bgSubtractor.apply(blur, None, 0)
            
    #finding the contour:
    #contours, hierarchy  = cv.findContours(fgImg, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    #list = [len(x) for x in contours]
    #indexes = np.argsort(list,0)

    #testimg = np.zeros_like(fgImg)
    #fgImg = cv.drawContours(testimg, contours, indexes[-1] , (255,255,255), -1)

    kernel = np.ones((3,3),np.uint8) 
    fgImg = cv.dilate(fgImg,kernel, iterations = dilation)

    return fgImg



vidpath1 = os.path.abspath('data/cam1/background.avi')
model1 = cv.createBackgroundSubtractorMOG2(150, 100, True)
model1.setShadowValue(0)
model1.setShadowThreshold(0.6)

vidpath2 = os.path.abspath('data/cam2/background.avi')
model2 = cv.createBackgroundSubtractorMOG2(150, 100, True)
model2.setShadowValue(0)
model2.setShadowThreshold(0.6)

vidpath3 = os.path.abspath('data/cam3/background.avi')
model3 = cv.createBackgroundSubtractorMOG2(150, 100, True)
model3.setShadowValue(0)
model3.setShadowThreshold(0.6)

vidpath4 = os.path.abspath('data/cam4/background.avi')
model4 = cv.createBackgroundSubtractorMOG2(150, 100, True)
model4.setShadowValue(0)
model4.setShadowThreshold(0.6)


#------------------------------------Training the background models--------------------------------------
modelList = [model1, model2, model3, model4]
vidpathList = [vidpath1,vidpath2,vidpath3,vidpath4]

print("please wait while the 4 background models get trained!")
for  mod,vidpath in zip(modelList,vidpathList):
    trainBackgroundModel(mod,vidpath)
    print("training model done")


#threshold parameters and dilation parameter for all camera's:
    #cam 1 = 100, 0.5, no dilation
    #cam 2 = 100, 0.42, with 2 iterations of dilation
    #cam 3 = 100, 0.5, no dilation
    #cam 4 = 100, 0.5, no dilation

#testBackgroundModel(model1, os.path.abspath('data/cam1/video.avi'), 0)
#g = subtractBackground(img, model2, 2)


