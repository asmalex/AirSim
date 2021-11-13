import setup_path
import gym
import airgym
import airsim
import time

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

airsim.wait_key('Press any key to takeoff and begin learning')
print("Taking off...")

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-img-v0", # Name of the Gym environment to use
                punish_act_coef = 1, # punish the drone for rapid changes in the action space
                reward_time_coef = 1, # reward the drone for every step it survives# reward the drone for every step it survives
                reward_dir_coef = 0.5, # reward the similarity between the drone's forward heading and the obstacle's
                punish_dir_coef = 5, # punish the change in the unit LOS vector
                punish_dist_coef = 0.1, # reward the closeness of the drone and the obstacle
                action_magnitude = 10, # Maximum value for the vehicle's movement function
                ip_address="127.0.0.1", # ip address of the airsim simulation
                step_length=0.25, # Length of a training step in seconds
                image_shape=(84, 84, 1), # Size of images to learn from
                action_type = 1 # Movement command used to control the drone
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = SAC(
    MlpPolicy, 
    env, 
    learning_rate=0.00025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=10000,
    learning_starts=10000,
    buffer_size=500000,
    tensorboard_log="./tb_logs/",)




# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=5e5,
    tb_log_name="sac_airsim_drone_run_" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("sac_airsim_drone_policy")
