import glm
import random
import numpy as np
import cv2 as cv
import voxels as v

block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    data = v.initilizeVoxels()
    colors = np.full_like(data,255)
    return data, colors

def set_voxel_positions_update():
    data = v.updateVoxels()
    colors = np.full_like(data,255)
    return data,colors



def get_cam_positions():
    out = list()
    
    for i in range(1, 5):
        path = f'data/cam{i}/config.xml'
        r = cv.FileStorage(path, cv.FileStorage_READ)
        tvecs = r.getNode('CameraTranslationVecs').mat()
        rvecs = r.getNode('CameraRotationVecs').mat()
        R = np.mat((4,4), float)
        R, _ = cv.Rodrigues(rvecs, R)
        R = np.append(R, [[0], [0], [0]], axis = 1)
        R = np.append(R, [[0, 0, 0, 1]], axis=0)

        tvecs = np.append(tvecs, [[1]]).transpose() # make tvecs a 4x1 matrix
        tvecs = -R.T @ tvecs
        tvecs = tvecs.ravel()[:3]

        tvecs[1], tvecs[2] = tvecs[2],-tvecs[1] # changing y and z, also setting z to -y for the mirroring
        #print(f'Final Cam{i} position: {tvecs / 20}')
        out.append(tvecs / 20)

    return out,[[1,0,0],[0,1,0],[0,0,1],[1,1,0]]


def get_cam_rotation_matrices():
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]

    for i in range(1, 5):
        path = f'data/cam{i}/config.xml'
        r = cv.FileStorage(path, cv.FileStorage_READ)
        rvecs = r.getNode('CameraRotationVecs').mat()
        rvecs[1], rvecs[2] = rvecs[2], rvecs[1] # swap y and z axes
        R = np.mat((4,4), float)
        R, _ = cv.Rodrigues(rvecs, R) # use Rodrigues to get 3x3 rotation matrix
        R = np.append(R, [[0], [0], [0]], axis = 1)
        R = np.append(R, [[0, 0, 0, 1]], axis=0) #transform into 4x4 rotation matrix
        zCorr = np.mat([[0, 1, 0, 0], # -90 degrees along Z
                        [-1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        yCorr = np.mat([[0, 0, 1, 0], # +90 degrees along Y
                        [0, 1, 0, 0],
                        [-1, 0, 0, 0],
                        [0, 0, 0, 1]])
        R = np.matmul(yCorr, R)
        R = np.matmul(zCorr, R)
        #print(f'Cam{i} rotation matrix after Y-axis correction: \n{R}')
        R = glm.mat4(R)
        cam_rotations[i-1] = R

    return cam_rotations
