from gymnasium.envs.registration import register

register(
    id="PathBuilder-v0",
    entry_point="path_builder.envs:PathBuilderEnv",
)