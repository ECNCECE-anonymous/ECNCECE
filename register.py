from gymnasium.envs.registration import register

register(
    id="EnvJax-v5",  # Unique ID for your environment
    entry_point="env_jax_v5:MyEnv",  # Path to the environment class
)

register(
    id="Env-v4",  # Unique ID for your environment
    entry_point="env_v4:MyEnv",  # Path to the environment class
)

register(
    id="Env-energy-v1",  # Unique ID for your environment
    entry_point="env_energy:MyEnv",  # Path to the environment class
)

register(
    id="Env-energy-v2",  # Unique ID for your environment
    entry_point="env_energy_v2:MyEnv",  # Path to the environment class
)
