# In settings.json first activate computer vision mode and set OriginGeoPoint to (0,0,2000):
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import setup_path
import airsim

import pprint
import os
import time
import math
import random
from datetime import datetime
import csv
import cv2
import numpy as np
from PIL import Image

PRECISION_ANGLE = 4 # Fractions of a degree used in generating random pitch, roll, and yaw values
PRECISION_METER = 100 # Fractions of a meter used in generating random distance values
RADIUS_MAX = 20 # Maximum distance from the obstacle to be expected
#TODO: Replace minimum distace with a test for detecting if the camera is inside the obstacle
RADIUS_MIN = 3 # Minimum distance from the obstacle to be expected. Set this value large enough so that the camera will not spawn inside the object

# Camera details should match settings.json
IMAGE_HEIGHT = 144
IMAGE_WIDTH = 256
FOV = 90
# TODO: Vertical FOV rounds down for generating random integers. Some pictures will not be created
VERT_FOV = FOV * IMAGE_HEIGHT // IMAGE_WIDTH

def polarToCartesian(r, theta, phi):
    return [
         r * math.sin(theta) * math.cos(phi),
         r * math.sin(theta) * math.sin(phi),
         r * math.cos(theta)]

pp = pprint.PrettyPrinter(indent=4)

client = airsim.VehicleClient()
client.confirmConnection()

if client.simSetSegmentationObjectID("Obstacle[\w]*", 0, True):
    print("Segmentation color set to black")
else:
    print("Segmentation color specification failed")

airsim.wait_key('Press any key to get camera parameters')
camera_info = client.simGetCameraInfo(str(0))
print("CameraInfo %d:" % 0)
pp.pprint(camera_info)

tmp_dir = os.path.join(os.getcwd(), "airsim_cv_mode")
tmp_dir = os.path.join(tmp_dir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

airsim.wait_key('Press any key to start generating images')
e = 0
while 1:
    # generate a random position for our obstacle
    r = random.randint(RADIUS_MIN * PRECISION_METER, RADIUS_MAX * PRECISION_METER) / PRECISION_METER
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
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        #airsim.ImageRequest("0", airsim.ImageType.DisparityNormalized),
        #airsim.ImageRequest("0", airsim.ImageType.SurfaceNormals)
        ])

    pose = client.simGetVehiclePose()

    # Print image responses to files
    for i, response in enumerate(responses):
        filename = os.path.join(tmp_dir, str(e) + "_" + str(i))
        if response.pixels_as_float:
            print("Pixels as float not implemented")
            break
        else:
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
            img_bgr = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
            img_bgr = np.flipud(img_bgr) #original image is fliped vertically
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Change from Airsim's BGR to an RGB image
            print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
            np.save(filename, img_rgb)
            im = Image.fromarray(img_rgb)
            im.save(filename + ".jpeg")
        # Create coordinate label file
        with open(filename + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([str(object_pose.orientation.w_val), str(object_pose.orientation.x_val), str(object_pose.orientation.y_val), str(object_pose.orientation.z_val), str(object_pose.position.x_val), str(object_pose.position.y_val), str(object_pose.position.z_val)])
            writer.writerow([str(pose.orientation.w_val), str(pose.orientation.x_val), str(pose.orientation.y_val), str(pose.orientation.z_val), str(pose.position.x_val), str(pose.position.y_val), str(pose.position.z_val)])

    pp.pprint(pose)

    e += 1

# currently reset() doesn't work in CV mode. Below is the workaround
client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)