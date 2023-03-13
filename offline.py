import voxels as v
import cv2 as cv



v.FrameNr = 0
personHistograms = v.initilizeVoxels(offline = True)

s = cv.FileStorage('data/PersonColorModels.xml', cv.FileStorage_WRITE)
for i,pershist in enumerate(personHistograms):
    s.write(f"ColorModelPerson{i}", pershist)
s.release()
