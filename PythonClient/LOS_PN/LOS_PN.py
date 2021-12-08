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
PG = 50 # Proportional Gain for the control algorithm (Should be between 2 and 6)

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

last_LOS = last_obs_pos - last_drone_pos # vector from the drone to the obstacle

print("Beginning Proportional Navigation")
i = 1
t1 = time.time()
client.moveByVelocityAsync(2000, 0, 0, 1, drivetrain = 1).join() # Move slightly in the direction of the first obstacle to ensure that the initial closing velocity is positive
#time.sleep(1)

while True:
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

    current_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
    current_vel = np.array([current_vel.x_val, current_vel.y_val, current_vel.z_val])

    accel = np.cross(-1 * PG * np.linalg.norm(relative_vel) * LOS / np.linalg.norm(LOS), LOS_rot)

    
    print("Current Velocity Length: ", np.linalg.norm(current_vel))
    print("Acceleration Length: ", np.linalg.norm(accel))
    print("Distance to obstacle: ", np.linalg.norm(LOS))

    client.moveByVelocityAsync(current_vel[0] + accel[0], current_vel[1] + accel[1], current_vel[2] + accel[1], 5, drivetrain = 1).join()

    if np.linalg.norm(LOS) < 1.5:
        i = i + 1
        if i > NUM_OBS:
            break

        received = client.simGetObjectPose("Obstacle" + str(i))
        last_obs_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the obstacle

        received = client.simGetVehiclePose()
        last_drone_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the drone

        last_LOS = last_obs_pos - last_drone_pos # vector from the drone to the obstacle

        client.moveByVelocityAsync(current_vel[0] + accel[0], current_vel[1] + accel[1], current_vel[2] + accel[1], 5, drivetrain = 1).join()
    else:
        last_LOS = LOS
        last_drone_pos = drone_pos
        last_obs_pos = obs_pos

    t1 = t2

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)