# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    # Spawn / reset configuration
    random_gate_spawn: bool  = True
    spawn_dist_min:    float = 1.0
    spawn_dist_max:    float = 3.0
    spawn_lateral:     float = 0.3
    spawn_vertical:    float = 0.2
    spawn_yaw_noise:   float = 0.3
    spawn_vel_max:     float = 0.0

    # Domain randomization
    domain_randomization: bool = False
    dr_start_iter:        int  = 0
    dr_full_iter:         int  = 500

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key.endswith("_reward_scale")]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Gate crossing state — owned by strategy, not env
        n_gates = self.env._waypoints.shape[0]
        self._target_gate_crossed     = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lap_completed_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gates_since_spawn       = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._prev_x_drone_wrt_all_gates = torch.zeros(self.num_envs, n_gates, device=self.device)

        # Set nominal parameters for all envs at init
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value
        self.env._tau_m[:] = self.env._tau_m_value
        self.env._thrust_to_weight[:] = self.env._twr_value

    def _compute_gate_crossings(self, drone_pose: torch.Tensor) -> torch.Tensor:
        """Check all gates for crossings. Returns [num_envs, n_gates] int tensor:
        +1 = forward crossing, -1 = backward crossing, 0 = no crossing.

        A crossing occurs when:
          1. x in gate frame flips sign between prev and now
          2. Drone is within the gate opening (|y| < half, |z| < half)
        """
        n_gates   = self.env._waypoints.shape[0]
        gate_half = self.cfg.gate_model.gate_side / 2.0

        crossings = torch.zeros(self.num_envs, n_gates, dtype=torch.int, device=self.device)

        for g in range(n_gates):
            pos_wrt_g, _ = subtract_frame_transforms(
                self.env._waypoints[g, :3].unsqueeze(0).expand(self.num_envs, -1),
                self.env._waypoints_quat[g, :].unsqueeze(0).expand(self.num_envs, -1),
                drone_pose
            )
            x_now = pos_wrt_g[:, 0]
            within = (torch.abs(pos_wrt_g[:, 1]) < gate_half) & (torch.abs(pos_wrt_g[:, 2]) < gate_half)

            prev_x = self._prev_x_drone_wrt_all_gates[:, g]
            fwd = (prev_x > 0) & (x_now <= 0) & within   # correct direction
            bwd = (prev_x <= 0) & (x_now > 0) & within    # wrong direction

            crossings[:, g] = fwd.int() - bwd.int()
            self._prev_x_drone_wrt_all_gates[:, g] = x_now

        return crossings

    def _update_gate_state(self, drone_pose: torch.Tensor):
        """Detect gate crossings and advance _idx_wp for the target gate.

        Called at the top of get_rewards() each training step.
        """
        n_gates = self.env._waypoints.shape[0]

        self._gates_crossed_this_step = self._compute_gate_crossings(drone_pose)

        # Advance waypoint only for forward crossing of the target gate
        target_crossed = self._gates_crossed_this_step[
            torch.arange(self.num_envs, device=self.device), self.env._idx_wp
        ] > 0

        # Save before _idx_wp advances — rewards read this after advancement
        self._target_gate_crossed = target_crossed

        # Incorrect crossing: forward through non-target gate, or backward through any gate.
        # Only counts as wrong if the correct gate wasn't also crossed this step.
        target_mask = torch.zeros(self.num_envs, n_gates, dtype=torch.bool, device=self.device)
        target_mask.scatter_(1, self.env._idx_wp.unsqueeze(1).long(), True)
        wrong_fwd = ((self._gates_crossed_this_step > 0) & ~target_mask).any(dim=1)
        any_bwd   = (self._gates_crossed_this_step < 0).any(dim=1)
        self._wrong_gate_crossed = (wrong_fwd | any_bwd) & ~target_crossed

        ids_crossed = torch.where(target_crossed)[0]
        if ids_crossed.numel() > 0:
            self.env._idx_wp[ids_crossed]             = (self.env._idx_wp[ids_crossed] + 1) % n_gates
            self.env._n_gates_passed[ids_crossed]     += 1
            self._gates_since_spawn[ids_crossed]  += 1

        # Use _gates_since_spawn (always starts at 0) for lap detection,
        # NOT _n_gates_passed (which starts at waypoint_indices for obs encoding).
        self._lap_completed_this_step = (
            target_crossed
            & (self._gates_since_spawn % n_gates == 0)
            & (self._gates_since_spawn > 0)
        )


    def get_rewards(self) -> torch.Tensor:

        if not self.cfg.is_train:
            return torch.zeros(self.num_envs, device=self.device)

        # Gate state (_target_gate_crossed, _lap_completed_this_step, _wrong_gate_crossed)
        # and _pose_drone_wrt_gate are already updated by _get_dones before this is called.
        crossed       = self._target_gate_crossed
        lap_completed = self._lap_completed_this_step

        # Progress reward: delta distance to current gate center.
        # Positive when closing distance, negative when retreating.
        # Retreat penalized retreat_mult× harder so any oscillation is net negative.
        dist_now  = torch.linalg.norm(self.env._pose_drone_wrt_gate, dim=1)
        progress  = self.env._last_distance_to_goal - dist_now  # +ve = approaching
        retreat_mult = self.env.rew['progress_retreat_multiplier']
        progress  = torch.where(progress >= 0, progress, progress * retreat_mult)
        # Zero out progress on gate-crossing steps: dist_now is relative to the new
        # target gate while _last_distance_to_goal was relative to the old one.
        progress[crossed] = 0.0
        self.env._last_distance_to_goal = dist_now.clone()

        # Crash: sustained contact force after a 100-step grace period
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask    = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # Kill envs that crossed wrong gate (triggers cond_crashed > 100 next step)
        self.env._crashed[self._wrong_gate_crossed] = 101

        if self.cfg.is_train:
            rewards = {
                # Sparse: +reward each time drone correctly traverses a gate
                "gate_cross":   crossed.float()       * self.env.rew['gate_cross_reward_scale'],
                # Sparse: large +reward on completing a full lap
                "lap_complete": lap_completed.float() * self.env.rew['lap_complete_reward_scale'],
                # Dense negative: penalise contact each step (after grace period)
                "crash":        crashed.float()       * self.env.rew['crash_reward_scale'],
                # Dense: delta distance to gate — inactive (coeff=0), enable for lap time opt
                "progress":     progress              * self.env.rew['progress_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                 torch.ones_like(reward) * self.env.rew['death_cost'], reward)
            reward[self._wrong_gate_crossed] = 0.0

            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Observation vector (27 dims):
            pos_w             (3)  — world position
            lin_vel_b         (3)  — linear velocity in body frame
            ang_vel_b         (3)  — body rates (roll/pitch/yaw rate)
            quat_w            (4)  — orientation quaternion in world frame
            prev_gate_rel_w   (3)  — vector from drone to previous gate in world frame
            gate_rel_w        (3)  — vector from drone to current gate in world frame
            next_gate_rel_w   (3)  — vector from drone to next gate in world frame
            gate_normal_w     (3)  — current gate approach normal in world frame
            target_gate_phase (2)  — (sin, cos) of target gate index, periodic across boundary
        """
        n_gates   = self.env._waypoints.shape[0]
        quat_w    = self.env._robot.data.root_quat_w          # [N, 4]

        # --- Core drone state ---
        pos_w     = self.env._robot.data.root_link_pos_w       # [N, 3]
        lin_vel_b = self.env._robot.data.root_com_lin_vel_b    # [N, 3]
        ang_vel_b = self.env._robot.data.root_ang_vel_b        # [N, 3]

        # --- Gate positions relative to drone in world frame ---
        prev_idx        = (self.env._idx_wp - 1) % n_gates
        prev_gate_pos_w = self.env._waypoints[prev_idx, :3]            # [N, 3]
        prev_gate_rel_w = prev_gate_pos_w - pos_w                      # [N, 3]

        curr_gate_pos_w = self.env._waypoints[self.env._idx_wp, :3]    # [N, 3]
        gate_rel_w      = curr_gate_pos_w - pos_w                      # [N, 3]

        next_idx        = (self.env._idx_wp + 1) % n_gates
        next_gate_pos_w = self.env._waypoints[next_idx, :3]            # [N, 3]
        next_gate_rel_w = next_gate_pos_w - pos_w                      # [N, 3]

        # --- Gate approach normal in world frame ---
        gate_normal_w = self.env._normal_vectors[self.env._idx_wp]      # [N, 3]

        # --- Target gate phase: sin/cos encoding of current target gate index ---
        angle     = 2.0 * np.pi * self.env._idx_wp.float() / n_gates
        gate_sin  = torch.sin(angle).unsqueeze(1)                       # [N, 1]
        gate_cos  = torch.cos(angle).unsqueeze(1)                       # [N, 1]

        obs = torch.cat([
            pos_w,              # (3) world position
            lin_vel_b,          # (3)
            ang_vel_b,          # (3)
            quat_w,             # (4)
            prev_gate_rel_w,    # (3) vector from drone to prev gate (world)
            gate_rel_w,         # (3) vector from drone to current gate (world)
            next_gate_rel_w,    # (3) vector from drone to next gate (world)
            gate_normal_w,      # (3) gate approach direction in world frame
            gate_sin,           # (1)
            gate_cos,           # (1)
        ], dim=-1)

        return {"policy": obs}

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        n_gates = self.env._waypoints.shape[0]
        cfg = self.cfg

        random_gate_spawn = getattr(cfg, "random_gate_spawn", self.random_gate_spawn)
        spawn_dist_min = getattr(cfg, "spawn_dist_min", self.spawn_dist_min)
        spawn_dist_max = getattr(cfg, "spawn_dist_max", self.spawn_dist_max)
        spawn_lateral = getattr(cfg, "spawn_lateral", self.spawn_lateral)
        spawn_vertical = getattr(cfg, "spawn_vertical", self.spawn_vertical)
        spawn_yaw_noise = getattr(cfg, "spawn_yaw_noise", self.spawn_yaw_noise)
        spawn_vel_max = getattr(cfg, "spawn_vel_max", self.spawn_vel_max)

        # Gate selection with a lightweight curriculum (config-overridable).
        # Stage 0: gate 0 only; Stage 1: gates 0/1/3; Stage 2: all gates.
        it = int(getattr(self.env, "iteration", 0))
        use_reset_curriculum = bool(getattr(cfg, "use_reset_curriculum", True))
        curriculum_iter_1 = int(getattr(cfg, "curriculum_iter_1", 1000))
        curriculum_iter_2 = int(getattr(cfg, "curriculum_iter_2", 3000))

        if use_reset_curriculum:
            if it < curriculum_iter_1:
                gate_pool = [0]
            elif it < curriculum_iter_2:
                gate_pool = [0, 1, 3]
            else:
                gate_pool = list(range(n_gates))
        else:
            gate_pool = list(range(n_gates))

        pool_tensor = torch.tensor(gate_pool, device=self.device, dtype=self.env._idx_wp.dtype)
        if random_gate_spawn:
            waypoint_indices = pool_tensor[torch.randint(0, len(gate_pool), (n_reset,), device=self.device)]
        else:
            waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)

        # Optionally respawn failed envs at the gate where they failed.
        if bool(getattr(cfg, "reset_on_fail_same_gate", True)) and self.cfg.is_train:
            crashed_mask = self.env.reset_terminated[env_ids]
            timeout_mask = self.env.reset_time_outs[env_ids]
            if crashed_mask.any():
                waypoint_indices = torch.where(crashed_mask, self.env._idx_wp[env_ids], waypoint_indices)
            if timeout_mask.any():
                waypoint_indices = torch.where(timeout_mask, self.env._idx_wp[env_ids], waypoint_indices)

        # Gate world pose
        x0_wp = self.env._waypoints[waypoint_indices, 0]
        y0_wp = self.env._waypoints[waypoint_indices, 1]
        z_wp = self.env._waypoints[waypoint_indices, 2]
        theta = self.env._waypoints[waypoint_indices, -1]

        # Spawn behind selected gate in gate-local frame
        x_local = torch.empty(n_reset, device=self.device).uniform_(-spawn_dist_max, -spawn_dist_min)
        y_local = torch.empty(n_reset, device=self.device).uniform_(-spawn_lateral, spawn_lateral)
        z_local = torch.empty(n_reset, device=self.device).uniform_(-spawn_vertical, spawn_vertical)

        # Rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_wp + z_local

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # Point drone toward selected gate with yaw noise
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-spawn_yaw_noise, spawn_yaw_noise)
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw + yaw_noise,
        )
        default_root_state[:, 3:7] = quat

        # Optional initial forward velocity toward gate
        if spawn_vel_max > 0.0:
            fwd_speed = torch.empty(n_reset, device=self.device).uniform_(0.0, spawn_vel_max)
            gate_normal = self.env._normal_vectors[waypoint_indices]
            default_root_state[:, 7:10] = gate_normal * fwd_speed.unsqueeze(1)
        else:
            default_root_state[:, 7:10] = 0.0
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            x_local = torch.empty(n_reset, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(n_reset, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards gate 0
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0
            default_root_state[:, 7:] = 0.0

            quat = quat_from_euler_xyz(
                torch.zeros(n_reset, device=self.device),
                torch.zeros(n_reset, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = torch.full(
                (n_reset,), self.env._initial_wp, device=self.device, dtype=self.env._idx_wp.dtype
            )

        # Set waypoint indices
        self.env._idx_wp[env_ids] = waypoint_indices

        # _n_gates_passed set to spawn gate index → sin/cos lap progress obs correct at spawn.
        # _gates_since_spawn always starts at 0 → used for lap completion detection.
        self.env._n_gates_passed[env_ids] = waypoint_indices
        self._gates_since_spawn[env_ids] = 0

        # Sync debug visualizer to spawn gate so the red sphere is correct from frame 0
        self.env._desired_pos_w[env_ids, :3] = self.env._waypoints[waypoint_indices, :3]

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Compute drone position in target gate frame — used for obs and reward init
        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        # Init prev_x for all gates from actual drone position
        spawn_pos = self.env._robot.data.root_link_state_w[env_ids, :3]
        n_gates = self.env._waypoints.shape[0]
        for g in range(n_gates):
            pos_wrt_g, _ = subtract_frame_transforms(
                self.env._waypoints[g, :3].unsqueeze(0).expand(len(env_ids), -1),
                self.env._waypoints_quat[g, :].unsqueeze(0).expand(len(env_ids), -1),
                spawn_pos
            )
            self._prev_x_drone_wrt_all_gates[env_ids, g] = pos_wrt_g[:, 0]
        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._pose_drone_wrt_gate[env_ids], dim=1
        )

        self.env._crashed[env_ids] = 0

        # Domain randomization: linear curriculum
        dr_enabled = bool(getattr(self.cfg, "domain_randomization", self.domain_randomization))
        if dr_enabled:
            n = len(env_ids)
            alpha = max(0.0, min(1.0, (self.env.iteration - self.dr_start_iter)
                                      / max(1, self.dr_full_iter - self.dr_start_iter)))

            def _rand(nominal, lo_mult, hi_mult):
                lo = nominal * (1.0 + alpha * (lo_mult - 1.0))
                hi = nominal * (1.0 + alpha * (hi_mult - 1.0))
                return torch.empty(n, device=self.device).uniform_(lo, hi)

            self.env._thrust_to_weight[env_ids] = _rand(self.env._twr_value, 0.95, 1.05)
            self.env._K_aero[env_ids, 0] = _rand(self.env._k_aero_xy_value, 0.5, 2.0)
            self.env._K_aero[env_ids, 1] = _rand(self.env._k_aero_xy_value, 0.5, 2.0)
            self.env._K_aero[env_ids, 2] = _rand(self.env._k_aero_z_value,  0.5, 2.0)
            self.env._kp_omega[env_ids, 0] = _rand(self.env._kp_omega_rp_value, 0.85, 1.15)
            self.env._kp_omega[env_ids, 1] = _rand(self.env._kp_omega_rp_value, 0.85, 1.15)
            self.env._ki_omega[env_ids, 0] = _rand(self.env._ki_omega_rp_value, 0.85, 1.15)
            self.env._ki_omega[env_ids, 1] = _rand(self.env._ki_omega_rp_value, 0.85, 1.15)
            self.env._kd_omega[env_ids, 0] = _rand(self.env._kd_omega_rp_value, 0.7, 1.3)
            self.env._kd_omega[env_ids, 1] = _rand(self.env._kd_omega_rp_value, 0.7, 1.3)
            self.env._kp_omega[env_ids, 2] = _rand(self.env._kp_omega_y_value, 0.85, 1.15)
            self.env._ki_omega[env_ids, 2] = _rand(self.env._ki_omega_y_value, 0.85, 1.15)
            self.env._kd_omega[env_ids, 2] = _rand(self.env._kd_omega_y_value, 0.7, 1.3)