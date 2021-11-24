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
PG = 5 # Proportional Gain for the control algorithm

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

print("Setting up the drone for flight...")

received = client.simGetObjectPose("Obstacle1")
last_obs_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the obstacle

client.takeoffAsync().join()

# Move the drone to a starting position nearby the first obstacle
client.moveToPositionAsync(obs_pos[0]-10, obs_pos[1]-20, obs_pos[2]+1, 1).join()

client.rotateToYawAsync(0, timeout_sec=3e+38, margin=2).join() # Rotate yaw to face forward

received = client.simGetVehiclePose()
last_drone_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the drone
last_drone_or = [received.orientation.w_val, received.orientation.x_val, received.orientation.y_val, received.orientation.z_val] # Quaternion rotation put on the drone

last_unit_LOS = normalize(last_obs_pos - last_drone_pos) # unit vector from the drone to the obstacle

print("Beginning Proportional Navigation")
print("Moving towrds obstacle 1...")
i = 1
t1 = time.time()
time.sleep(0.01)
while i < NUM_OBS + 1:
    # Find the positions of the drone and the obstacle
    received = client.simGetObjectPose("Obstacle"+str(i))
    obs_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the obstacle
    received = client.simGetVehiclePose()
    drone_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the drone
    drone_or = [received.orientation.w_val, received.orientation.x_val, received.orientation.y_val, received.orientation.z_val] # Quaternion rotation put on the drone
    unit_LOS = normalize(obs_pos - drone_pos) # unit vector from the drone to the obstacle

    t2 = time.time()

    # Calculate the closing velocity
    drone_vel = (drone_pos - last_drone_pos) / (t2 - t1)
    obs_vel = (obs_pos - last_obs_pos) / (t2 - t1)
    closing_velocity = np.dot( (obs_vel - drone_vel), np.cross((obs_pos - drone_pos), (obs_pos - drone_pos) / (np.linalg.norm(obs_pos - drone_pos)**2)) )

    # Calculate the magnitude of the rotational change in the LOS vector
    ang = np.arccos(np.inner(unit_LOS, last_unit_LOS) / (np.linalg.norm(unit_LOS) * np.linalg.norm(last_unit_LOS)))
    ang_t = ang / (t2 - t1)

    # Calculate a unit vector in the direction of the desired LOS rotation
    LOS_rot = unit_LOS - np.cross(np.inner(unit_LOS, last_unit_LOS / np.linalg.norm(unit_LOS)), last_unit_LOS / np.linalg.norm(unit_LOS))
    unit_LOS_rot = LOS_rot / np.linalg.norm(LOS_rot)

    # Calculate the acceleration vector to be passed into the drone
    LOS_accel = PG * np.cross(np.cross(closing_velocity, ang_t), unit_LOS_rot)


    #TODO: Accelerate the drone by the calculated vector
    print(LOS_accel)
    current_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
    client.moveByVelocityAsync(current_vel.x_val + LOS_accel[0], current_vel.y_val + LOS_accel[1], current_vel.z_val + LOS_accel[1], 1, drivetrain = 1).join()

    # record the variables for the next iteration
    last_obs_pos = obs_pos
    last_drone_pos = drone_pos
    last_unit_LOS = unit_LOS
    t1 = t2

    











client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
