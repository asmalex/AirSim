import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

#from websocket import create_connection
#import paramiko


class AirSimDroneEnvironmentTwo(gym.Env):
    #ws = create_connection("ws://127.0.0.1:30020/obstacle")

    def __init__(self, ip_address, step_length):
        self.observation_space = spaces.Box(-50, 50, shape=(2,7), dtype=np.float32)
        self.viewer = None
        self.step_length = step_length

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)

        # MoveByManual -vx * -vy * -z * -duration 5 -yaw_rate *
        self.action_space = spaces.Box(np.array([-50.0, -50.0, -50.0, 0.0], dtype=np.float32), np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float32))  
        self._setup_flight()
        
        # self.ssh = paramiko.SSHClient()
        # TODO: Link commands to DroneShell
        # self.ssh.connect(server, username=username, password=password)

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        #self.drone.reset()
        self.send_to_drone("RequestControl")
        self.send_to_drone("Arm")
        self.send_to_drone("Takeoff")
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity (normalized vector in the direction of the first obstacle)
        self.send_to_drone("MoveToPosition -x 35.64 -y 20.49 -z -25.8")
        self.drone.moveToPositionAsync(35.64, 20.49, -25.8, 5).join()
        #TODO: Orient the drone so it can see the first obstacle

        received = self.drone.simGetObjectPose("Obstacle1")
        obs_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the obstacle

        received = self.drone.simGetVehiclePose()
        drone_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the drone

        #print ("Receiving...")
        #received =  self.ws.recv() # Assumes input is a single string representing the csv containing required data
        #print ("Received")
        #received =  received.split(',')
        #received = [float(received[i]) for i in range(len(received))]
            
        #obs_pos = np.array([received[0],received[1],received[2]]) # Global coordinates of the obstacle
        #drone_pos = np.array([received[7],received[8],received[9]]) # Global coordinates of the drone

        self.last_unit_LOS = AirSimDroneEnvironmentTwo.normalize(drone_pos - obs_pos) # unit vector from the drone to the obstacle

    def _get_obs(self):
        received_obs = self.drone.simGetObjectPose("Obstacle1")
        received_drone = self.drone.simGetVehiclePose()
        obs = np.array([[received_obs.position.x_val, received_obs.position.y_val, received_obs.position.z_val, received_obs.orientation.w_val, received_obs.orientation.x_val, received_obs.orientation.y_val, received_obs.orientation.z_val], [received_drone.position.x_val, received_drone.position.y_val, received_drone.position.z_val, received_drone.orientation.w_val, received_drone.orientation.x_val, received_drone.orientation.y_val, received_drone.orientation.z_val]])
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return obs

    def _do_action(self, action):
        command = self.interpret_action(action)
        self.send_to_drone(command)
        self.drone.moveByAngleRatesThrottleAsync(float(action[0]), float(action[1]), float(action[2]), float(action[3]), 0.25).join()

    #TODO: Send signal through websocket to punish the algorithm if no obstacle can be seen
    def _compute_reward(self):
        if self.state["collision"]:
            reward = -100
        else:
            received = self.drone.simGetObjectPose("Obstacle1")
            obs_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the obstacle
            obs_or = [received.orientation.w_val, received.orientation.x_val, received.orientation.y_val, received.orientation.z_val] # Quaternion rotation put on the obstacle
            # v' = v + 2 * r x (s * v + r x v) / m
            obs_face = np.array([1,0,0]) + np.cross(2 * np.array(obs_or[1:]), obs_or[0]*np.array([1,0,0]) + np.cross(np.array(obs_or[1:]), np.array([1,0,0]))) / (obs_or[0]**2 + obs_or[1]**2 + obs_or[2]**2 + obs_or[3]**2)

            received = self.drone.simGetVehiclePose()
            drone_pos = np.array([received.position.x_val, received.position.y_val, received.position.z_val]) # Global coordinates of the drone

            LOS = obs_pos - drone_pos # Vector from the drone to the obstacle
            unit_LOS = AirSimDroneEnvironmentTwo.normalize(LOS) # unit vector from the drone to the obstacle
            LOS_change = math.sqrt((self.last_unit_LOS[0] - unit_LOS[0])**2 + (self.last_unit_LOS[1] - unit_LOS[1])**2 + (self.last_unit_LOS[2] - unit_LOS[2])**2) # Change in LOS since the last step
            self.last_unit_LOS = unit_LOS
            offset = math.sqrt((obs_face[0] - unit_LOS[0])**2 + (obs_face[1] - unit_LOS[1])**2 + (obs_face[2] - unit_LOS[2])**2) # difference between current heading and the heading corresponding to going straight through the obstacle
            
            reward_dir = (2 - offset) * 0.5

            punish_dir = LOS_change

            reward_speed = (
                np.linalg.norm(
                    [
                        self.state["velocity"].x_val,
                        self.state["velocity"].y_val,
                        self.state["velocity"].z_val,
                    ]
                )
                - 0.5
                )
            reward = np.clip(reward_speed, 0, 10) + reward_dir - punish_dir

        done = 0
        if reward <= -10:
            done = 1

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        #Temporarily report reward for testing
        print(reward)

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        # moveByAngleRatesThrottleAsync -roll_rate * -pitch_rate * -yaw_rate * -throttle * -duration 5
        command = "moveByAngleRatesThrottleAsync -roll_rate "
        command += str(action[0])
        command += " -pitch_rate "
        command += str(action[1])
        command += " -yaw_rate "
        command += str(action[2])
        command += " -throttle "
        command += str(action[3])
        command += " -duration 5"

        return command

    # Sends a command string to DroneShell
    def send_to_drone(self, command):
        return # Temporarily disable websocket sending
        #print(command)
        # ssh_stdin, ssh_stdout, ssh_stderr = self.ssh.exec_command(command)

    # Closes the Websocket ClientSide
    #def close_socket(self):
        #self.ws.close()

    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    
    def polarToCartesian(r, theta, phi):
        return [
             r * math.sin(theta) * math.cos(phi),
             r * math.sin(theta) * math.sin(phi),
             r * math.cos(theta)]

    def cartesianToPolar(x,y,z):
        return [
            np.sqrt(x**2 + y**2 + z**2),
            np.arctan2(y, x),
            np.arctan2(np.sqrt(x**2 + y**2), z)]