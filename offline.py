import voxels as vox
import cv2 as cv



vox.FrameNr = 0 # for camera 2
personHistogramstwoviews2 = vox.initilizeVoxels(offline = True)

vox.FrameNr = 1200 # for camera 3
personHistogramstwoviews3 = vox.initilizeVoxels(offline = True)

s = cv.FileStorage('data/PersonColorModels.xml', cv.FileStorage_WRITE)
for i,pershist in enumerate(personHistogramstwoviews2[0]):
    s.write(f"ColorModelPerson{0}{i}", pershist)

#hand selected the indices so persons match in the frames
s.write(f"ColorModelPerson{1}{0}", personHistogramstwoviews3[1][3])
s.write(f"ColorModelPerson{1}{1}", personHistogramstwoviews3[1][1])
s.write(f"ColorModelPerson{1}{2}", personHistogramstwoviews3[1][2])
s.write(f"ColorModelPerson{1}{3}", personHistogramstwoviews3[1][0])




s.release()
