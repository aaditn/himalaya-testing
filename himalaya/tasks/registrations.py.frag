

##
# Himalaya: G1 trekking on continuous rough terrain.
# Registered here (rather than an external package) because train.py's
# @hydra_task_config decorator resolves the task id at import time, before
# any external module gets a chance to register.
##

gym.register(
    id="Himalaya-G1-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.himalaya_env_cfg:HimalayaTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Himalaya-G1-Student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.himalaya_env_cfg:HimalayaStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Himalaya-G1-Teacher-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.himalaya_env_cfg:HimalayaTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)


