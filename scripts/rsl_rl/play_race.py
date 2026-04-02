# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import sys
import os
local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)
    print(f"[INFO] Using local rsl_rl from: {local_rsl_path}")
else:
    print(f"[WARNING] Local rsl_rl not found at: {local_rsl_path}")

import argparse
import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2500, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--follow_robot", type=int, default=-1, help="Follow robot index.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — avoids GUI conflicts with Isaac Sim
import matplotlib.pyplot as plt

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# Import extensions to set up environment tasks
import src.isaac_quad_sim2real.tasks   # noqa: F401


def _find_strategy(env):
    """Traverse wrapper layers to find the env strategy attribute."""
    e = env
    for _ in range(10):
        if hasattr(e, 'strategy'):
            return e.strategy
        if hasattr(e, 'env'):
            e = e.env
        else:
            break
    return None


def _plot_reward_debug(reward_log, out_dir):
    """Save a 5-panel reward debug figure as PNG."""
    steps = [r["timestep"] for r in reward_log]

    fig, axes = plt.subplots(5, 1, figsize=(18, 14), sharex=True)
    fig.suptitle("Reward Debug — env 0", fontsize=12)

    # --- Panel 1: sparse events (gate crossed, wrong, lap, apex) ---
    ax = axes[0]
    for key, color, label in [
        ("gate_crossed",           "C0", "gate_crossed"),
        ("wrong_crossing",         "C1", "wrong_crossing"),
        ("lap_completed",          "C2", "lap_completed"),
        ("loop_apex_just_reached", "C3", "loop_apex"),
    ]:
        fired = [s for s, r in zip(steps, reward_log) if r[key]]
        ax.vlines(fired, 0, 1, colors=color, linewidth=1.5, label=label)
    ax.set_ylim(-0.1, 1.3)
    ax.set_ylabel("Events")
    ax.legend(fontsize=8, ncol=4, loc="upper right")

    # --- Panel 2: raw progress (distance delta to current gate) ---
    ax = axes[1]
    ax.plot(steps, [r["progress_raw"] for r in reward_log], color="steelblue", linewidth=0.7)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_ylabel("progress_raw (m)")

    # --- Panel 3: drone z-height ---
    ax = axes[2]
    ax.plot(steps, [r["drone_z"] for r in reward_log], color="darkorange", linewidth=0.7)
    ax.set_ylabel("drone_z (m)")

    # --- Panel 4: power-loop state flags ---
    ax = axes[3]
    ax.plot(steps, [r["crossed_gate2_flag"] for r in reward_log], linewidth=0.8, label="crossed_gate2")
    ax.plot(steps, [r["above_gate2_flag"]   for r in reward_log], linewidth=0.8, label="above_gate2")
    ax.plot(steps, [r["loop_incomplete"]     for r in reward_log], linewidth=0.8, linestyle="--", label="loop_incomplete")
    ax.set_ylabel("Loop flags")
    ax.legend(fontsize=8, ncol=3, loc="upper right")

    # --- Panel 5: gate index + crash overlay ---
    ax = axes[4]
    ax.plot(steps, [r["gate_idx"] for r in reward_log], color="purple", linewidth=0.8, label="gate_idx")
    ax.set_ylabel("gate_idx", color="purple")
    ax2 = ax.twinx()
    ax2.fill_between(steps, [r["crashed"] for r in reward_log], alpha=0.35, color="red", label="crashed")
    ax2.set_ylabel("crashed", color="red")
    ax.set_xlabel("Timestep")
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "reward_debug.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[INFO] Reward debug plot saved to: {out_path}")


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # All debug outputs go here — download this one folder
    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = os.path.join(os.path.dirname(log_root_path), "debug_run_" + run_tag)
    os.makedirs(debug_dir, exist_ok=True)
    print(f"[INFO] Debug outputs will be saved to: {debug_dir}")

    if args_cli.follow_robot == -1:
        env_cfg.viewer.resolution = (1920, 1080)
        env_cfg.viewer.eye = (10.7, 0.4, 7.2)
        env_cfg.viewer.lookat = (-2.7, 0.5, -0.3)
    elif args_cli.follow_robot >= 0:
        env_cfg.viewer.eye = (-0.8, 0.8, 0.8)
        env_cfg.viewer.resolution = (1920, 1080)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.env_index = args_cli.follow_robot
        env_cfg.viewer.asset_name = "robot"

    env_cfg.is_train = False
    env_cfg.domain_randomization = False
    env_cfg.max_motor_noise_std = 0.0
    env_cfg.seed = args_cli.seed

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(debug_dir, "video"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # reset environment
    obs = env.get_observations()
    # Extract tensor from TensorDict for policy
    if hasattr(obs, "get"):  # Check if it's a TensorDict
        obs = obs["policy"]  # Extract the policy observation

    # Debug reward logging setup (env 0 only)
    strategy = _find_strategy(env)
    reward_log = []
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
            # Extract tensor from TensorDict for policy
            if hasattr(obs, "get"):  # Check if it's a TensorDict
                obs = obs["policy"]  # Extract the policy observation

        timestep += 1

        # Collect per-step reward breakdown from strategy debug snapshot
        if strategy is not None and strategy._debug_rewards:
            record = {"timestep": timestep}
            record.update(strategy._debug_rewards)
            reward_log.append(record)

        if args_cli.video and timestep == args_cli.video_length:
            break

    # close the simulator
    env.close()

    # Write reward debug CSV and plot into debug_dir
    if reward_log:
        csv_path = os.path.join(debug_dir, "reward_debug.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=reward_log[0].keys())
            writer.writeheader()
            writer.writerows(reward_log)
        print(f"[INFO] Reward debug CSV saved to: {csv_path}")
        _plot_reward_debug(reward_log, debug_dir)
    print(f"[INFO] All outputs in: {debug_dir}")
    print(f"[INFO] To download:  scp -r <user>@<vm>:{debug_dir} .")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
