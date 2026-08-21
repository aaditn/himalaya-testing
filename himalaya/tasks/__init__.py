"""Himalaya task registration."""

import gymnasium as gym

from . import himalaya_env_cfg  # noqa: F401

gym.register(
    id="Himalaya-G1-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.himalaya_env_cfg:HimalayaTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents."
            "rsl_rl_ppo_cfg:G1RoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Himalaya-G1-Student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.himalaya_env_cfg:HimalayaStudentEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents."
            "rsl_rl_ppo_cfg:G1RoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Himalaya-G1-Teacher-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.himalaya_env_cfg:HimalayaTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents."
            "rsl_rl_ppo_cfg:G1RoughPPORunnerCfg"
        ),
    },
)
