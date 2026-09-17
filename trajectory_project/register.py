import gymnasium as gym

from trajectory_project.franka_trajectory_env_cfg import FrankaTrajectoryReachEnvCfg
from isaaclab_tasks.manager_based.manipulation.reach.config.franka.agents.rsl_rl_ppo_cfg import (
    FrankaReachPPORunnerCfg,
)

gym.register(
    id="Isaac-Franka-Circle-Trajectory-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "trajectory_project.franka_trajectory_env_cfg:FrankaTrajectoryReachEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": FrankaReachPPORunnerCfg,
    },
)
