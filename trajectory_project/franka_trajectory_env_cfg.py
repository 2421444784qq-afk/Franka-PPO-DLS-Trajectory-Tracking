import isaaclab.controllers as controllers
import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp

from isaaclab.managers import ObservationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import (
    FrankaReachEnvCfg,
)

from trajectory_project.trajectory_command import TrajectoryPoseCommand


@configclass
class FrankaTrajectoryReachEnvCfg(FrankaReachEnvCfg):
    """Franka reaching with continuous circular end-effector trajectory."""

    def __post_init__(self):
        super().__post_init__()

        # Use our continuous circular trajectory command.
        self.commands.ee_pose.class_type = TrajectoryPoseCommand

        # PPO outputs a 3D Cartesian position increment.
        # Differential IK converts this task-space command into
        # joint-space targets using damped least-squares (DLS).
        self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=controllers.DifferentialIKControllerCfg(
                command_type="position",
                use_relative_mode=True,
                ik_method="dls",
                ik_params={"lambda_val": 0.01},
            ),
            scale=0.05,
        )

        # Add desired Cartesian velocity of the circular trajectory
        # to the policy observation.
        self.observations.policy.desired_velocity = ObservationTermCfg(
            func=reach_mdp.desired_velocity,
            params={
                "command_name": "ee_pose",
            },
        )
