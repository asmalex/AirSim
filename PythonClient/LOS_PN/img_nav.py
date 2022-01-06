import setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2
import math
import time

# Camera details should match settings.json
IMAGE_HEIGHT = 144
IMAGE_WIDTH = 256
CENTER = (IMAGE_HEIGHT //2, IMAGE_WIDTH//2)
FOV = 90
# TODO: Vertical FOV rounds down for generating random integers. Some pictures will not be created
VERT_FOV = FOV * IMAGE_HEIGHT // IMAGE_WIDTH

OBS_LEN = 2.2 # Obstacle diameter in meters
NUM_OBS = 6 # The number of obstacles in the course

VEL = 5 # Target velocity along the LOS vector

def getDepth(h,w):
    # Constants calibrated from image_calibrator.py
    #  2.76588914e+01 6.76894450e-02  7.40320895e+00  8.81630020e-03 -2.69137285e-01
    return 27.6588914 * np.exp(-0.0676894450 * h) + 7.40320895 * np.exp(-0.00881630020 * w) + 0.269137285

def cartesianToPolar(x,y,z):
    return [
        np.sqrt(x**2 + y**2 + z**2),
        np.arctan2(y, x),
        np.arctan2(np.sqrt(x**2 + y**2), z)]

def polarToCartesian(r, theta, phi):
    return [
         r * math.sin(theta) * math.cos(phi),
         r * math.sin(theta) * math.sin(phi),
         r * math.cos(theta)]

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def getBoundBox():
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
    #img_bgr = np.flipud(img_bgr) #original image is fliped vertically
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Change from Airsim's BGR to an RGB image

    u = -1 # Highest point of the obstacle
    b = -1 # Lowest point of the obstacle

    for i in range(len(image)):
        for j in range(len(image[i])):
            if np.array_equal(image[i][j], [0,0,0]):
                if u == -1:
                    u = i
                b = i

    l = -1 # Leftmost point of the obstacle
    r = -1 # Rightmost point of the obstacle

    for j in range(len(image[0])):
        for i in range(len(image)):
            if np.array_equal(image[i][j], [0,0,0]):
                if l == -1:
                    l = j
                r = j

    i = u + (b-u)//2
    j = l + (r-l)//2
    return ([i,j] , [u,b,l,r])

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

if client.simSetSegmentationObjectID("Obstacle[\w]*", 0, True):
    print("Segmentation color set to black")
else:
    print("Segmentation color specification failed")

print("Setting up the drone for flight...")

received = client.simGetObjectPose("Obstacle1")
last_obs_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the obstacle

client.armDisarm(True)
client.takeoffAsync().join()

# Move the drone to a starting position nearby the first obstacle
client.moveToPositionAsync(last_obs_pos[0]-20, last_obs_pos[1]-5, last_obs_pos[2]-1, 1).join()

client.rotateToYawAsync(0, timeout_sec=3e+38, margin=2).join() # Rotate yaw to face forward

while True:
    center, bounds = getBoundBox() # The coordinates for the bounding box of the obstacle
    print(center, bounds)
    depth = getDepth(bounds[1] - bounds[0], bounds[3] - bounds[2]) # Estimated distance to the obstacle in meters
    pixel_size = 2.2 / max(bounds[1] - bounds[0], bounds[3] - bounds[2]) # number of meters per pixel in the surface of the sphere of radius 'depth'. Obtained by comparing the known size of the obstacle to the number of pixels it includes

    yaw_angle = (CENTER[0] - center[0]) * pixel_size / depth # yaw angle from the camera center to the center of the obstacle, calculated using the arc length formula
    pitch_angle = (CENTER[1] - center[1]) * pixel_size / depth # pitch angle from the camera center to the center of the obstacle, calculated using the arc length formula

    vector = polarToCartesian(1, pitch_angle + 90, yaw_angle) # Unit LOS Vector, defined in the Cartesian axis relative to the drone

    received = client.simGetVehiclePose() # TODO: Simulation specific API. Replace with Kinematics orientation estimation and/or GPS position
    # drone_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the drone
    drone_or = np.array([received.orientation.w_val, received.orientation.x_val, received.orientation.y_val, received.orientation.z_val]) # Global quaternion on the drone
    # q^{-1} = [q0/||q||, -q1/||q||, -q2/||q||, -q3/||q||]
    drone_or_inv = [drone_or[0]/(drone_or[0]**2 + drone_or[1]**2 + drone_or[2]**2 + drone_or[3]**2), -drone_or[1]/(drone_or[0]**2 + drone_or[1]**2 + drone_or[2]**2 + drone_or[3]**2), -drone_or[2]/(drone_or[0]**2 + drone_or[1]**2 + drone_or[2]**2 + drone_or[3]**2), -drone_or[3]/(drone_or[0]**2 + drone_or[1]**2 + drone_or[2]**2 + drone_or[3]**2)] # Inverse quaternion of drone's orientation used to convert from Bodyframe to Worldframe
    # v' = v + 2 * r x (s * v + r x v) / m
    LOS = np.array(vector) + np.cross(2 * np.array(drone_or_inv[1:]), drone_or_inv[0]*np.array(vector) + np.cross(np.array(drone_or_inv[1:]), np.array(vector))) / (drone_or_inv[0]**2 + drone_or_inv[1]**2 + drone_or_inv[2]**2 + drone_or_inv[3]**2) # Image of LOS vector under inverse quaternion

    client.moveByVelocityBodyFrameAsync(VEL * vector[0], VEL * vector[1], VEL * vector[2] * -1, 2).join()

client.reset()
client.armDisarm(False)
client.enableApiControl(False)


'''

'''