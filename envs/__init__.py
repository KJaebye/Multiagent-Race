from gymnasium.envs.registration import register

# cannot be trained by using mappo

register(
    id="MultiAnt-v0",
    entry_point="envs.race.tasks.multi_ant_v0:MultiAntEnv",
)

register(
    id="MultiHopper-v0",
    entry_point="envs.race.tasks.multi_hopper_v0:MultiHopperEnv",
)

register(
    id="MultiWalker2d-v0",
    entry_point="envs.race.tasks.multi_walker2d_v0:MultiWalker2dEnv",
)

register(
    id="MultiHalfCheetah-v0",
    entry_point="envs.race.tasks.multi_half_cheetah_v0:MultiHalfCheetahEnv",
)

register(
    id="MultiSwimmer-v0",
    entry_point="envs.race.tasks.multi_swimmer_v0:MultiSwimmerEnv",
)

register(
    id="MultiHumanoid-v0",
    entry_point="envs.race.tasks.multi_humanoid_v0:MultiHumanoidEnv",
)


CUSTOM_ENVS = ["MultiAnt-v0", 
               "MultiHopper-v0", 
               "MultiWalker2d-v0", 
               "MultiHalfCheetah-v0", 
               "MultiSwimmer-v0",
               "MultiHumanoid-v0",
               ]
