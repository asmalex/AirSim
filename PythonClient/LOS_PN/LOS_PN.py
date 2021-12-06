import setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2
import math
import time

def cartesianToPolar(x,y,z):
    return [
        np.sqrt(x**2 + y**2 + z**2),
        np.arctan2(y, x),
        np.arctan2(np.sqrt(x**2 + y**2), z)]


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

NUM_OBS = 6 # The number of obstacles in the course
PG = 3 # Proportional Gain for the control algorithm (Should be between 2 and 6)

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

print("Setting up the drone for flight...")

received = client.simGetObjectPose("Obstacle1")
last_obs_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the obstacle

client.armDisarm(True)
client.takeoffAsync().join()

# Move the drone to a starting position nearby the first obstacle
client.moveToPositionAsync(last_obs_pos[0]-50, last_obs_pos[1]-15, last_obs_pos[2]-1, 1).join()

client.rotateToYawAsync(0, timeout_sec=3e+38, margin=2).join() # Rotate yaw to face forward

received = client.simGetVehiclePose()
last_drone_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the drone
last_drone_or = [received.orientation.w_val, received.orientation.x_val, received.orientation.y_val, received.orientation.z_val] # Quaternion rotation put on the drone

last_LOS = last_obs_pos - last_drone_pos # vector from the drone to the obstacle

print("Beginning Proportional Navigation")
i = 1
t1 = time.time()
client.moveByVelocityAsync(2000, 0, 0, 1, drivetrain = 1).join() # Move slightly in the direction of the first obstacle to ensure that the initial closing velocity is positive
#time.sleep(1)

while i < NUM_OBS + 1:
    print("Moving towrds obstacle " + str(i) + "...\n")
    # Find the positions of the drone and the obstacle
    received = client.simGetObjectPose("Obstacle"+str(i))
    obs_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the obstacle
    received = client.simGetVehiclePose()
    drone_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the drone
    LOS = obs_pos - drone_pos # vector from the drone to the obstacle

    t2 = time.time()

    # Calculate the closing velocity
    drone_vel = (drone_pos - last_drone_pos) / (t2 - t1)
    obs_vel = (obs_pos - last_obs_pos) / (t2 - t1)
    relative_vel = obs_vel - drone_vel

    LOS_rot = np.cross(LOS, relative_vel) / np.dot(LOS, LOS)

    accel = np.cross(-1 * PG * np.linalg.norm(relative_vel) * LOS / np.linalg.norm(LOS), LOS_rot)
    print(accel)

    current_vel = client.getMultirotorState().kinematics_estimated.linear_velocity

    client.moveByVelocityAsync(current_vel.x_val + accel[0], current_vel.y_val + accel[1], current_vel.z_val + accel[1], 1, drivetrain = 1).join()

    last_LOS = LOS
    t1 = t2
    last_drone_pos = drone_pos
    last_obs_pos = obs_pos

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)