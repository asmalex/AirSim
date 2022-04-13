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

#Number of obstacles available in the scene for imaging
NUM_OBS = 1

# Default position for our obstacles in Airsim format
OBS_REST = airsim.Pose(airsim.Vector3r(0, 0, -10000), airsim.to_quaternion(math.radians(0), math.radians(0), math.radians(0))) #radians

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
date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
tmp_dir = os.path.join(tmp_dir, date)
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

airsim.wait_key('Press any key to start generating images')
e = 0
while 1:
    # Set all obstacle's positions to default
    for k in range(NUM_OBS):
        if not client.simSetObjectPose("Obstacle"+ str(k), OBS_REST, True):
            print("Object pose setting failed at Obstacle" + str(k))
            exit()

    # Generate a random direction for the camera to face
    phi = random.randint(0, 180 * PRECISION_ANGLE) / PRECISION_ANGLE
    theta = random.randint(0, 360 * PRECISION_ANGLE) / PRECISION_ANGLE

    # Set the camera to face the direction
    camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(phi), math.radians(0), math.radians(theta))) #radians
    client.simSetVehiclePose(camera_pose, True)

    # generate a random number of obstacles to appear in the image
    obs = random.randint(0, NUM_OBS)

    # Generate random offsets to the camera position to place the obstacles at
    positions = []
    for k in range(obs):
        # Generate a random distance for the obstacle
        r = random.randint(RADIUS_MIN * PRECISION_METER, RADIUS_MAX * PRECISION_METER) / PRECISION_METER
        # Generate a random offset for the obstacle position
        obs_pitch = random.randint(0, VERT_FOV * PRECISION_ANGLE) / PRECISION_ANGLE - VERT_FOV / 2
        obs_yaw = random.randint(0, FOV * PRECISION_ANGLE) / PRECISION_ANGLE - FOV/2
        # Calculate the obstacle position in Cartesian coordinates    TODO: Obstacles not in frame


        # For testing, set each offset to 0
        obs_pitch = 0
        obs_yaw = 0
        pos = polarToCartesian(r, math.radians(theta + obs_yaw), math.radians(phi + obs_pitch))

        # Generate a random angular position for the obstacle
        pitch = random.randint(0, 180 * PRECISION_ANGLE) / PRECISION_ANGLE - 90.0
        roll = random.randint(0, 360 * PRECISION_ANGLE) / PRECISION_ANGLE
        yaw = random.randint(0, 360 * PRECISION_ANGLE) / PRECISION_ANGLE - 180

        # Encode obstacle position into Airsim format and store in positions
        object_pose = airsim.Pose(airsim.Vector3r(pos[0], pos[1], pos[2]), airsim.to_quaternion(math.radians(pitch), math.radians(roll), math.radians(yaw))) #radians
        positions.append(object_pose)

    # Find our bounding boxes and write a coordinate label file
    if e % 5 == 0:
        filename = tmp_dir + "\\labels\\validation\\" + date + "_" + str(e) + '.txt'
    else:
        filename = tmp_dir + "\\labels\\train\\" + date + "_" + str(e) + '.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as txtwriter:
        for k in range(len(positions)):
            # Set the obstacle to its calculated position
            if not client.simSetObjectPose("Obstacle"+ str(k), positions[k], True):
                print("Object pose setting failed at Obstacle" + str(k))
                exit()

            # Take a Segmentation picture
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
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Change from Airsim's BGR to an RGB image


            image = img_rgb
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

            y_center = (u + (b-u)/2)/IMAGE_HEIGHT
            x_center = (l + (r-l)/2)/IMAGE_WIDTH
            height = (b-u)/IMAGE_HEIGHT
            width = (r-l)/IMAGE_WIDTH
            txtwriter.write('0 ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n')

            # Return the obstacle to its resting position
            if not client.simSetObjectPose("Obstacle"+ str(k), OBS_REST, True):
                print("Object pose setting failed at Obstacle" + str(k))
                exit()

    # Move all the obstacle back into the frame and take a Scene picture
    for k in range(len(positions)):
        if not client.simSetObjectPose("Obstacle"+ str(k), positions[k], True):
                print("Object pose setting failed at Obstacle" + str(k))
                exit()

    # Take a Scene picture
    responses = client.simGetImages([
    #airsim.ImageRequest("0", airsim.ImageType.DepthVis),
    #airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True),
    #airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
    #airsim.ImageRequest("0", airsim.ImageType.DisparityNormalized),
    #airsim.ImageRequest("0", airsim.ImageType.SurfaceNormals)
    ])
    response = responses[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
    img_bgr = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
    img_bgr = np.flipud(img_bgr) #original image is fliped vertically
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Change from Airsim's BGR to an RGB image
    im = Image.fromarray(img_rgb) # Convert numpy array to an image
    print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
    
    if e % 5 == 0:
        filename = tmp_dir + "\\images\\validation\\" + date + "_" + str(e) + ".jpeg"
    else:
        filename = tmp_dir + "\\images\\train\\" + date + "_" + str(e) + ".jpeg"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    im.save(filename)

    e += 1

# currently reset() doesn't work in CV mode. Below is the workaround
client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)







# Old method for taking pictures of one obstacle
'''
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
    if not client.simSetObjectPose("Obstacle1", object_pose, True):
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
        if response.pixels_as_float:
            print("Pixels as float not implemented")
            break
        else:
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
            img_bgr = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
            img_bgr = np.flipud(img_bgr) #original image is fliped vertically
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Change from Airsim's BGR to an RGB image
            print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
        # Create coordinate label file
        if (i==0):  
            if e % 5 == 0:
                filename = tmp_dir + "\\labels\\validation\\" + date + "_" + str(e) + '.txt'
            else:
                filename = tmp_dir + "\\labels\\train\\" + date + "_" + str(e) + '.txt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as txtwriter:
                image = img_rgb
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

                y_center = (u + (b-u)/2)/IMAGE_HEIGHT
                x_center = (l + (r-l)/2)/IMAGE_WIDTH
                height = (b-u)/IMAGE_HEIGHT
                width = (r-l)/IMAGE_WIDTH
                txtwriter.write('0 ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height))
        # Save the raw image
        else:
            #np.save(filename, img_rgb)
            im = Image.fromarray(img_rgb)
            if e % 5 == 0:
                filename = tmp_dir + "\\images\\validation\\" + date + "_" + str(e) + ".jpeg"
            else:
                filename = tmp_dir + "\\images\\train\\" + date + "_" + str(e) + ".jpeg"
            print(tmp_dir,filename)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            im.save(filename)

    pp.pprint(pose)

    e += 1
'''