import argparse
import contextlib
import csv
import sys
import os

import gymnasium as gym
import numpy as np
import torch
import yaml

import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


TASK = "Isaac-Franka-Circle-Trajectory-v0"
NUM_STEPS = 1000

CHECKPOINT = (
    "logs/rsl_rl/franka_reach/"
    "2026-09-16_22-28-51/model_2999.pt"
)


parser = argparse.ArgumentParser(
    description="Evaluate Franka PPO circular trajectory tracking."
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
)

add_launcher_args(parser)
parser.set_defaults(visualizer=["newton"])

args_cli, hydra_args = setup_preset_cli(parser)

sys.argv = [sys.argv[0]] + hydra_args


def main():

    torch.manual_seed(42)

    env_cfg, _ = resolve_task_config(TASK, "")

    agent_yaml = (
        "logs/rsl_rl/franka_reach/"
        "2026-09-16_22-28-51/params/agent.yaml"
    )
    with open(agent_yaml, "r", encoding="utf-8") as f:
        agent_cfg = yaml.safe_load(f)

    with launch_simulation(env_cfg, args_cli):

        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device

        env = gym.make(TASK, cfg=env_cfg).unwrapped

        obs, _ = env.reset()

        robot = env.scene["robot"]

        body_ids, body_names = robot.find_bodies("panda_hand")
        ee_body_idx = body_ids[0]

        # Wrap environment for RSL-RL.
        env_wrapped = RslRlVecEnvWrapper(
            env,
            clip_actions=agent_cfg.get("clip_actions"),
        )

        print()
        print("=" * 60)
        print("PPO trajectory evaluation")
        print("=" * 60)
        print("Task:", TASK)
        print("Checkpoint:", CHECKPOINT)
        print("EE body:", body_names[0])
        print("EE body index:", ee_body_idx)
        print("Number of environments:", env.num_envs)
        print("Simulation dt:", env.step_dt)
        print("=" * 60)

        # Create PPO runner.
        runner = OnPolicyRunner(
            env_wrapped,
            agent_cfg,
            log_dir=None,
            device=agent_cfg["device"],
        )

        print("[INFO] Loading PPO checkpoint...")
        runner.load(CHECKPOINT)

        policy = runner.get_inference_policy(
            device=env.unwrapped.device
        )

        desired_list = []
        actual_list = []
        error_list = []

        for step in range(NUM_STEPS):

            with torch.inference_mode():

                # PPO policy generates the action.
                actions = policy(obs)

                obs, reward, dones, info = env_wrapped.step(
                    actions
                )

                # Desired EE position.
                desired_pos = (
                    env.command_manager
                    .get_term("ee_pose")
                    .pose_command_w[:, :3]
                )

                # Actual EE position in world coordinates.
                actual_pos = robot.data.body_pos_w.torch[
                    :, ee_body_idx
                ]

                # Convert to environment-local coordinates.
                actual_pos = actual_pos - env.scene.env_origins

                error = desired_pos - actual_pos

                if step < 5:
                    print(
                        "desired:",
                        desired_pos[0].detach().cpu().numpy(),
                    )
                    print(
                        "actual :",
                        actual_pos[0].detach().cpu().numpy(),
                    )
                    print(
                        "error  :",
                        error[0].detach().cpu().numpy(),
                    )

                desired_list.append(
                    desired_pos[0].detach().cpu().numpy()
                )

                actual_list.append(
                    actual_pos[0].detach().cpu().numpy()
                )

                error_list.append(
                    error[0].detach().cpu().numpy()
                )

        env.close()

    desired = np.asarray(desired_list)
    actual = np.asarray(actual_list)
    error = np.asarray(error_list)

    position_error = np.linalg.norm(error, axis=1)

    rmse = np.sqrt(np.mean(position_error ** 2))
    mae = np.mean(position_error)
    max_error = np.max(position_error)

    print()
    print("=" * 60)
    print("PPO Trajectory Tracking Results")
    print("=" * 60)
    print(f"Steps       : {NUM_STEPS}")
    print(f"Duration    : {NUM_STEPS * env.step_dt:.3f} s")
    print(f"RMSE        : {rmse:.6f} m")
    print(f"MAE         : {mae:.6f} m")
    print(f"Max Error   : {max_error:.6f} m")
    print("=" * 60)

    with open(
        "ppo_original_trajectory_results.csv",
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "time",
                "desired_x",
                "desired_y",
                "desired_z",
                "actual_x",
                "actual_y",
                "actual_z",
                "error_x",
                "error_y",
                "error_z",
                "position_error",
            ]
        )

        dt = env.step_dt

        for i in range(len(desired)):

            writer.writerow(
                [
                    i * dt,
                    *desired[i],
                    *actual[i],
                    *error[i],
                    position_error[i],
                ]
            )


if __name__ == "__main__":
    main()
