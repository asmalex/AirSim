import setup_path
import airsim

import numpy as np
import pprint
import cv2
import math
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt



def func(X, a1, b1, a2, b2, c):
    x,y = X
    return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * y) + c

# Camera details should match settings.json
IMAGE_HEIGHT = 144
IMAGE_WIDTH = 256
FOV = 90
# TODO: Vertical FOV rounds down for generating random integers. Some pictures will not be created
VERT_FOV = FOV * IMAGE_HEIGHT // IMAGE_WIDTH

OBS_LEN = 2.2 # Obstacle diameter in meters
DIST_MIN = 2 # Distance from obstacle in meters
DIST_MAX = 20 # Distance from obstacle in meters
STEP = 500

client = airsim.VehicleClient()
client.confirmConnection()

if client.simSetSegmentationObjectID("Obstacle[\w]*", 0, True):
    print("Segmentation color set to black")
else:
    print("Segmentation color specification failed")

airsim.wait_key('Press any key to calculate the effective distacnce of the image from the drone')

ydata = np.linspace(DIST_MIN, DIST_MAX, STEP)
xdata = []

for dist in ydata:
    # Move the obstacle in front of our drone
    object_pose = airsim.Pose(airsim.Vector3r(dist, 0, 0), airsim.to_quaternion(0, 0, 0)) 
    if not client.simSetObjectPose("Obstacle_3", object_pose, True):
            print("Object pose setting failed")

    responses = client.simGetImages([
            #airsim.ImageRequest("0", airsim.ImageType.DepthVis),
            #airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True),
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
            #airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            #airsim.ImageRequest("0", airsim.ImageType.DisparityNormalized),
            #airsim.ImageRequest("0", airsim.ImageType.SurfaceNormals)
            ])

    response = responses[0]

    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
    img_bgr = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
    img_bgr = np.flipud(img_bgr) #original image is fliped vertically
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Change from Airsim's BGR to an RGB image

    u = -1 # Highest point of the obstacle
    b = -1 # Lowest point of the obstacle

    for i in range(len(image)):
        for j in range(len(image[i])):
            if np.array_equal(image[i][j], [0,0,0]):
                if u == -1:
                    u = i
                b = i

    obs_len = b-u # Obstacle diameter in pixels

    l = -1
    r = -1

    for i in range(len(image[0])):
        for j in range(len(image)):
            if np.array_equal(image[j][i], [0,0,0]):
                if l == -1:
                    l = i
                r = i

    print(b-u,r-l)

    xdata.append([b-u, r-l])

xdata = ([x[0] for x in xdata], [x[1] for x in xdata])
print(np.array(xdata).shape)
print(np.array(ydata).shape)

parameters, covariance = curve_fit(func, xdata, ydata)
print("parameters: ", parameters)