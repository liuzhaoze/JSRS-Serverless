from gymnasium.envs.registration import register

register(
    id="cloudcomputing_env/CloudComputing-v0",
    entry_point="cloudcomputing_env.envs:CloudComputingEnv",
)
