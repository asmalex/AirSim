from gym.envs.registration import register

# This package is used for our simulation
register(
    id="airsim-drone-sample-v0", entry_point="airgym.envs:AirSimDroneEnv",
)

register(
    id="airsim-drone-img-v0", entry_point="airgym.envs:AirSimDroneEnvironment",
)

register(
    id="airsim-drone-pos-v0", entry_point="airgym.envs:AirSimDroneEnvironmentTwo",
)

register(
    id="airsim-car-sample-v0", entry_point="airgym.envs:AirSimCarEnv",
)