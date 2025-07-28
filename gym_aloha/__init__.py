from gymnasium.envs.registration import register

register(
    id="gym_aloha/AlohaInsertion-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels", "task": "insertion"},
)

register(
    id="gym_aloha/AlohaTransferCube-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels", "task": "transfer_cube"},
)

register(
    id="gym_aloha/TrossenAIStationaryTransferCube-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels", "task": "trossen_ai_stationary_transfer_cube"},
)

register(
    id="gym_aloha/TrossenAIStationaryTransferCubeEE-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels", "task": "trossen_ai_stationary_transfer_cube_ee"},
)
