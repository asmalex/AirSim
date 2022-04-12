import setup_path
import airsim

import torch
import numpy as np
import os
import tempfile
import pprint
import cv2
import math
import time
import threading
from PIL import Image
import random

# Camera details should match settings.json
IMAGE_HEIGHT = 144
IMAGE_WIDTH = 256
CENTER = (IMAGE_HEIGHT //2, IMAGE_WIDTH//2)
FOV = 90
# TODO: Vertical FOV rounds down
VERT_FOV = FOV * IMAGE_HEIGHT // IMAGE_WIDTH

OBS_LEN = 2.2 # Obstacle diameter in meters
NUM_OBS = 6 # The number of obstacles in the course

VEL = 0.1 # Target velocity along the LOS vector

# Load our custom PyTorch YOLOv3 model
model = torch.hub.load('ultralytics/yolov3', 'custom', 'D:\\Redei\\yolov3\\runs\\train\\exp12\\weights\\best.pt')

'''
# Ground truth depth function
def getDepth(h,w):
    received = client.simGetObjectPose("Obstacle1")
    obs_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the obstacle
    received = client.simGetVehiclePose()
    drone_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the drone

    return np.linalg.norm(obs_pos - drone_pos)
'''


def getDepth(h,w):
    # Constants calibrated from image_calibrator.py
    #  8.89922094  0.0304452  13.7360959   0.02951979  0.10829337
    return 8.89922094 * np.exp(-0.0304452 * h) + 13.7360959 * np.exp(-0.02951979 * w) + 0.10829337

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

def printImage(image):
    mask =  np.full_like(image, 255)

    for i in range(len(image)):
        for j in range(len(image[i])):
            if np.array_equal(image[i][j], [0,0,0]):
                mask[i][j] = [0,0,0]

    im = Image.fromarray(mask)
    im.show()

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
    im = Image.fromarray(image)

    #x = threading.Thread(target=printImage, args=(image,), daemon=True)
    #x.start()

    results = model(im).xyxy[0].cpu().numpy()

    # if no box is detected, report back the center of the camera to continue moving in the same direction
    if (results.size == 0):
        result = [CENTER[0], CENTER[1], CENTER[0], CENTER[1]]
    # If any object is detected, report back the box info of the largest box
    else:
        max_size = 0
        i = 0
        for j, result in enumerate(results):
            l = result[0]
            u = result[1]
            r = result[2]
            b = result[3]
            if np.max([b-u, r-l]) > max_size:
                i = j
        result = results[i]

    # Return the dimensions and midpoint of the box
    l = result[0]
    u = result[1]
    r = result[2]
    b = result[3]
    i = u + (b-u)//2
    j = l + (r-l)//2
    return ([i,j] , [u,b,l,r])

#TODO: Drone navigates to the right side of the obstacle regardless of starting position. Corrections to tragectory occur but happen to late





# Generate offsets for the drone's starting position
x = random.randint(-50,-40)
y = random.randint(-10,10)
z = random.randint(-10,10)

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
client.moveToPositionAsync(last_obs_pos[0]+x, last_obs_pos[1]+y, last_obs_pos[2]+z, 1).join()

# client.rotateToYawAsync(0, timeout_sec=3e+38, margin=2).join() # Rotate yaw to face forward

tm = time.time()
while True:
    center, bounds = getBoundBox() # The coordinates for the bounding box of the obstacle
    print("center: ", center)
    print("size: ", [bounds[1] - bounds[0], bounds[3] - bounds[2]])
    depth = getDepth(bounds[1] - bounds[0], bounds[3] - bounds[2]) # Estimated distance to the obstacle in meters
    print("depth: ", depth)
    if bounds[0] == bounds[1] and bounds[2] == bounds[3]:
        pixel_size = 2.2 # If we are reporting back the trivial box, set an arbitrary pixel_size to aviod dividing by zero
    else:
        pixel_size = 2.2 / max(bounds[1] - bounds[0], bounds[3] - bounds[2]) # number of meters per pixel in the surface of the sphere of radius 'depth'. Obtained by comparing the known size of the obstacle to the number of pixels it includes

    yaw_angle = (center[1] - CENTER[1]) * pixel_size / depth # yaw angle from the camera center to the center of the obstacle, calculated using the arc length formula
    pitch_angle = (center[0] - CENTER[0]) * pixel_size / depth # pitch angle from the camera center to the center of the obstacle, calculated using the arc length formula

    print("angles: ", yaw_angle,pitch_angle)

    vector = polarToCartesian(1, pitch_angle + 0.5 * math.pi, -1 * yaw_angle) # Unit LOS Vector, defined in the Cartesian axis relative to the drone

    '''
    # TODO(optional): Test quaternion math (BodyFrame works)

    received = client.simGetVehiclePose() # TODO: Simulation specific API. Replace with Kinematics orientation estimation and/or GPS position
    # drone_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the drone
    drone_or = np.array([received.orientation.w_val, received.orientation.x_val, received.orientation.y_val, received.orientation.z_val]) # Global quaternion on the drone
    # q^{-1} = [q0/||q||, -q1/||q||, -q2/||q||, -q3/||q||]
    drone_or_inv = [drone_or[0]/(drone_or[0]**2 + drone_or[1]**2 + drone_or[2]**2 + drone_or[3]**2), -drone_or[1]/(drone_or[0]**2 + drone_or[1]**2 + drone_or[2]**2 + drone_or[3]**2), -drone_or[2]/(drone_or[0]**2 + drone_or[1]**2 + drone_or[2]**2 + drone_or[3]**2), -drone_or[3]/(drone_or[0]**2 + drone_or[1]**2 + drone_or[2]**2 + drone_or[3]**2)] # Inverse quaternion of drone's orientation used to convert from Bodyframe to Worldframe
    # v' = v + 2 * r x (s * v + r x v) / m
    LOS = np.array(vector) + np.cross(2 * np.array(drone_or_inv[1:]), drone_or_inv[0]*np.array(vector) + np.cross(np.array(drone_or_inv[1:]), np.array(vector))) / (drone_or_inv[0]**2 + drone_or_inv[1]**2 + drone_or_inv[2]**2 + drone_or_inv[3]**2) # Image of LOS vector under inverse quaternion
    print(LOS)
    '''

    # velocity is proportional to the estimated distance from the object
    velocity = VEL*max(depth, 1)

    print("Velocity: ", velocity)

    print("Processing time: ", time.time() - tm)

    client.moveByVelocityBodyFrameAsync(velocity * vector[0], -1 * velocity * vector[1], -1 * velocity * vector[2], 1)

    tm = time.time()

client.reset()
client.armDisarm(False)
client.enableApiControl(False)


'''

'''