import setup_path
import airsim

import numpy as np
import pprint
import cv2
import math
import random
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

#RESULTS: 8.89922094  0.0304452  13.7360959   0.02951979  0.10829337



def func(X, a1, b1, a2, b2, c):
    x,y = X
    return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * y) + c

def polarToCartesian(r, theta, phi):
    return [
         r * math.sin(theta) * math.cos(phi),
         r * math.sin(theta) * math.sin(phi),
         r * math.cos(theta)]

OBS_LEN = 2.2 # Obstacle diameter in meters
DIST_MIN = 2 # Distance from obstacle in meters
DIST_MAX = 20 # Distance from obstacle in meters
STEP = 500

PRECISION_ANGLE = 4 # Fractions of a degree used in generating random pitch, roll, and yaw values
PRECISION_METER = 100 # Fractions of a meter used in generating random distance values

# Camera details should match settings.json
IMAGE_HEIGHT = 144
IMAGE_WIDTH = 256
FOV = 90
# TODO: Vertical FOV rounds down for generating random integers. Some pictures will not be created
VERT_FOV = FOV * IMAGE_HEIGHT // IMAGE_WIDTH

client = airsim.VehicleClient()
client.confirmConnection()

if client.simSetSegmentationObjectID("Obstacle[\w]*", 0, True):
    print("Segmentation color set to black")
else:
    print("Segmentation color specification failed")

airsim.wait_key('Press any key to calculate the effective distacnce of the image from the drone')

ydata = np.linspace(DIST_MIN, DIST_MAX, STEP)
ydata = np.ndarray.flatten(np.array([[y, y, y, y] for y in ydata]))
xdata = []

for dist in ydata:
    # generate a random position for our obstacle
    r = dist
    phi = random.randint(0, 360 * PRECISION_ANGLE) / PRECISION_ANGLE
    theta = random.randint(0, 180 * PRECISION_ANGLE) / PRECISION_ANGLE
    # Convert polar coordinates to cartesian for AirSim
    pos = polarToCartesian(r, math.radians(theta), math.radians(phi))

    # Generate a random angular position for the obstacle
    pitch = random.randint(0, 180 * PRECISION_ANGLE) / PRECISION_ANGLE - 90.0
    roll = random.randint(0, 360 * PRECISION_ANGLE) / PRECISION_ANGLE
    yaw = random.randint(0, 360 * PRECISION_ANGLE) / PRECISION_ANGLE - 180

    # Move the obstacle to our calculated position
    object_pose = airsim.Pose(airsim.Vector3r(pos[0], pos[1], pos[2]), airsim.to_quaternion(math.radians(pitch), math.radians(roll), math.radians(yaw))) #radians
    if not client.simSetObjectPose("Obstacle_3", object_pose, True):
        print("Object pose setting failed")
        break

    # Generate a random offset for the camera angle
    cam_pitch = random.randint(0, VERT_FOV * PRECISION_ANGLE) / PRECISION_ANGLE - VERT_FOV / 2
    # TODO: Rotating the drone causes the obstacle to be removed from the image because the camera is not square
    #cam_roll = random.randint(0, 360 * PRECISION_ANGLE) / PRECISION_ANGLE
    cam_roll = 0
    cam_yaw = random.randint(0, FOV * PRECISION_ANGLE) / PRECISION_ANGLE - FOV/2

    # Set the camera to face the object
    camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(theta - 90 + cam_pitch), math.radians(cam_roll), math.radians(phi + cam_yaw))) #radians
    client.simSetVehiclePose(camera_pose, True)

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