from gym.envs.registration import register

# This package is used for our simulation
register(
    id="airsim-drone-sample-v0", entry_point="airgym.envs:AirSimDroneEnv",
)

register(
    id="airsim-drone-sample-v1", entry_point="airgym.envs:AirSimDroneEnvironment",
)

register(
    id="airsim-car-sample-v0", entry_point="airgym.envs:AirSimCarEnv",
)