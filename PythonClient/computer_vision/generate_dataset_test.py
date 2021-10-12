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

# generate a random position for our camera
r = random.randint(RADIUS_MIN * PRECISION_METER, RADIUS_MAX * PRECISION_METER) / PRECISION_METER
phi = random.randint(0, 360 * PRECISION_ANGLE) / PRECISION_ANGLE
theta = random.randint(0, 180 * PRECISION_ANGLE) / PRECISION_ANGLE
# Convert polar coordinates to cartesian for AirSim
pos = polarToCartesian(r, math.radians(theta), math.radians(phi))

# Generate a random offset for the camera angle
pitch = random.randint(0, VERT_FOV * PRECISION_ANGLE) / PRECISION_ANGLE - VERT_FOV / 2
# TODO: Rotating the drone causes the obstacle to be removed from the image because the camera is not square
#roll = random.randint(0, 360 * PRECISION_ANGLE) / PRECISION_ANGLE
roll = 0
yaw = random.randint(0, FOV * PRECISION_ANGLE) / PRECISION_ANGLE - FOV/2

# Calculate coordinates of the center of the obstacle relative to the drone's new position and orientation
# To use global coordinates, use the recorded global position of the camera where the obstacle is at [0,0,0]
obs_r = r
obs_phi = yaw
obs_theta = 90 - pitch
# Convert polar coordinates to cartesian for AirSim
obs_pos = polarToCartesian(obs_r, math.radians(obs_theta), math.radians(obs_phi))

# Record rotational transformation on obstacle for calculating coordinates of key locations relative to the center.
# These can be used to recover any point on the object by translating the object to the origin, applying rotation matrices, then translating back to its position
obs_phi_offset = -phi
obs_theta_offset = -theta

# Move the camera to our calculated position
camera_pose = airsim.Pose(airsim.Vector3r(pos[0], pos[1], pos[2]), airsim.to_quaternion(math.radians(90 - theta + pitch), math.radians(roll), math.radians(phi + 180 + yaw))) #radians
client.simSetVehiclePose(camera_pose, True)


pose = client.simGetVehiclePose()
print("Pose from Airsim")
pp.pprint(pose)
print("Pose from calculations")
pp.pprint(pos)
print("Pose of the obstacle")
pp.pprint(obs_pos)
print("Rotation on the obstacle")
pp.pprint([obs_phi_offset, obs_theta_offset])

client.simPlotArrows([airsim.Vector3r(pos[0], pos[1], pos[2])], [airsim.Vector3r(0, 0, 0)], is_persistent=True)