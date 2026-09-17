from __future__ import annotations

import torch

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg


class TrajectoryPoseCommand(UniformPoseCommand):
    """Continuous circular end-effector trajectory command."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # One trajectory clock for each parallel environment.
        self.trajectory_time = torch.zeros(
            self.num_envs, device=self.device
        )

        # Desired Cartesian velocity in the environment frame.
        # Shape: [num_envs, 3] -> [vx, vy, vz]
        self.velocity_command_b = torch.zeros(
            self.num_envs, 3, device=self.device
        )

    def _resample_command(self, env_ids):
        """Initialize/reset the trajectory for selected environments."""
        self.trajectory_time[env_ids] = 0.0

        # Start from the center of the circular trajectory.
        self.pose_command_b[env_ids, 0] = 0.50
        self.pose_command_b[env_ids, 1] = 0.00
        self.pose_command_b[env_ids, 2] = 0.30

        # Desired velocity at t = 0:
        # vx = 0
        # vy = radius * omega
        # vz = 0
        self.velocity_command_b[env_ids, 0] = 0.0
        self.velocity_command_b[env_ids, 1] = 0.05
        self.velocity_command_b[env_ids, 2] = 0.0

        # Keep the original orientation.
        self.pose_command_b[env_ids, 3:] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            device=self.device,
        )

    def _update_command(self):
        """Update desired position and velocity along the circular trajectory."""
        self.trajectory_time += self._env.step_dt

        radius = 0.10
        omega = 0.5

        theta = omega * self.trajectory_time

        # Desired Cartesian position.
        self.pose_command_b[:, 0] = 0.50 + radius * torch.cos(theta)
        self.pose_command_b[:, 1] = 0.00 + radius * torch.sin(theta)
        self.pose_command_b[:, 2] = 0.30

        # Desired Cartesian velocity:
        # vx = -r * omega * sin(theta)
        # vy =  r * omega * cos(theta)
        # vz = 0
        self.velocity_command_b[:, 0] = -radius * omega * torch.sin(theta)
        self.velocity_command_b[:, 1] = radius * omega * torch.cos(theta)
        self.velocity_command_b[:, 2] = 0.0


class TrajectoryPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for continuous trajectory pose command."""

    class_type = TrajectoryPoseCommand
