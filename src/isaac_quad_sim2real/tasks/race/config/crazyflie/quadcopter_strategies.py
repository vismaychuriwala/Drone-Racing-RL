# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

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
            keys.append("lap_time")  # lap_time reward has no _reward_scale config key
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

        # Rolling action history buffer: [num_envs, num_prev_action_steps * 4]
        # Oldest actions at index 0, most recent at index -4:
        n = self.cfg.num_prev_action_steps
        self._action_history = torch.zeros(
            self.num_envs, n * 4, dtype=torch.float, device=self.device
        )

        # Power-loop tracking: gate 2 must be crossed AND loop height achieved
        # before gate 3 crossing earns reward / progress counts.
        self._crossed_gate2      = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._above_gate2        = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._loop_apex_rewarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Debug snapshot populated each step during play (is_train=False)
        self._debug_rewards: dict = {}

    def get_rewards(self) -> torch.Tensor:

        if not self.cfg.is_train:
            # --- Play mode: run tracking & store env-0 indicators for CSV/plot debug ---
            crossed        = self.env._gate_crossed_this_step
            lap_completed  = self.env._lap_completed_this_step
            wrong_crossing = self.env._wrong_crossing_this_step
            curr_d  = torch.norm(self.env._pose_drone_wrt_gate, dim=-1)
            drone_z = self.env._robot.data.root_link_pos_w[:, 2]

            # Power-loop flags (same logic as training path)
            gate2_just_crossed = crossed & (self.env._idx_wp == 3)
            self._crossed_gate2 = self._crossed_gate2 | gate2_just_crossed
            prev_above_gate2 = self._above_gate2.clone()
            gate2_z = self.env._waypoints[2, 2]
            loop_height_threshold = 1.0
            # Height-only condition — lateral fired too easily since gate 2 exit is
            # already ~1.25m lateral from gate 3, barely below any reasonable threshold.
            self._above_gate2 = self._above_gate2 | (
                (self.env._idx_wp == 3)
                & self._crossed_gate2
                & (drone_z > gate2_z + loop_height_threshold)
            )
            loop_apex_just_reached = (
                self._above_gate2 & ~prev_above_gate2 & ~self._loop_apex_rewarded
            )
            self._loop_apex_rewarded = self._loop_apex_rewarded | loop_apex_just_reached
            loop_incomplete = (
                (self.env._idx_wp == 3) & self._crossed_gate2 & ~self._above_gate2
            )

            # Reset power-loop flags once gate 3 is crossed (keeps behaviour
            # purely between gate 2 and gate 3; default progress resumes after).
            n_gates = self.env._waypoints.shape[0]
            gate3_just_crossed = crossed & (self.env._idx_wp == (4 % n_gates))
            self._crossed_gate2[gate3_just_crossed]      = False
            self._above_gate2[gate3_just_crossed]        = False
            self._loop_apex_rewarded[gate3_just_crossed] = False

            contact_forces = self.env._contact_sensor.data.net_forces_w
            crashed_vals = (torch.norm(contact_forces, dim=-1) > 1e-8).reshape(self.num_envs)

            self._debug_rewards = {
                "gate_crossed":           int(crossed[0].item()),
                "wrong_crossing":         int(wrong_crossing[0].item()),
                "lap_completed":          int(lap_completed[0].item()),
                "loop_apex_just_reached": int(loop_apex_just_reached[0].item()),
                "loop_incomplete":        int(loop_incomplete[0].item()),
                "gate3_just_crossed":     int(gate3_just_crossed[0].item()),
                "progress_raw":           round(
                    float(self.env._last_distance_to_goal[0].item() - curr_d[0].item()), 5
                ),
                "crashed":                int(crashed_vals[0].item()),
                "gate_idx":               int(self.env._idx_wp[0].item()),
                "drone_z":                round(float(drone_z[0].item()), 4),
                "crossed_gate2_flag":     int(self._crossed_gate2[0].item()),
                "above_gate2_flag":       int(self._above_gate2[0].item()),
            }
            self.env._last_distance_to_goal = curr_d.clone()
            return torch.zeros(self.num_envs, device=self.device)


        # Gate crossing and lap completion are computed in env._update_gate_state()
        # which runs at the top of _get_dones — before this call and before
        # _get_observations, so _idx_wp and _pose_drone_wrt_gate are already
        # correct for the current gate when we get here.
        crossed        = self.env._gate_crossed_this_step
        lap_completed  = self.env._lap_completed_this_step
        wrong_crossing = self.env._wrong_crossing_this_step

        # Progress reward: change in Euclidean distance (norm) to current gate center.
        # Positive when the drone gets closer to the gate (distance decreases).
        # Retreat penalized retreat_mult× harder so oscillation is net negative.
        # current Euclidean distance to gate center (gate-frame coordinates)
        curr_d = torch.norm(self.env._pose_drone_wrt_gate, dim=-1)

        # Fix: when an env crossed a gate this step, _idx_wp was advanced
        # before get_rewards() runs, so _pose_drone_wrt_gate is relative to
        # the NEW gate while _last_distance_to_goal still holds the value
        # measured to the OLD gate. Replace last_d for crossed envs with
        # the current distance to avoid a spurious large progress spike.
        last_d = self.env._last_distance_to_goal.clone()
        try:
            crossed_mask = crossed.bool()
        except Exception:
            crossed_mask = (crossed > 0)
        last_d[crossed_mask] = curr_d[crossed_mask]

        # --- Power-loop tracking ---
        # Stage 1: gate 2 forward crossing (_idx_wp already advanced to 3).
        gate2_just_crossed = crossed & (self.env._idx_wp == 3)
        self._crossed_gate2 = self._crossed_gate2 | gate2_just_crossed

        # Stage 2: loop apex — drone must reach height above gate 2 while targeting gate 3.
        prev_above_gate2 = self._above_gate2.clone()
        gate2_z = self.env._waypoints[2, 2]                        # gate 2 world z
        drone_z = self.env._robot.data.root_link_pos_w[:, 2]
        loop_height_threshold = 1.0                                # metres above gate 2 centre
        # No lateral condition — that fired too easily (drone starts ~1.25m lateral
        # from gate 3 right after crossing gate 2). Height only forces a real vertical arc.
        self._above_gate2 = self._above_gate2 | (
            (self.env._idx_wp == 3)
            & self._crossed_gate2
            & (drone_z > gate2_z + loop_height_threshold)
        )
        # Fire exactly once: the step the apex threshold is first crossed.
        loop_apex_just_reached = self._above_gate2 & ~prev_above_gate2 & ~self._loop_apex_rewarded
        self._loop_apex_rewarded = self._loop_apex_rewarded | loop_apex_just_reached

        # Suppress progress toward gate 3 while the arc is incomplete (play-mode debug only).
        loop_incomplete = (
            (self.env._idx_wp == 3)
            & self._crossed_gate2
            & ~self._above_gate2
        )

        # Tanh-normalised progress: tanh(Δd / scale) ∈ (-1, +1).
        # scale = typical forward distance per step (e.g. 0.05m @ 3m/s, 60Hz).
        # Symmetrically penalise retreat by retreat_mult before normalising so
        # the asymmetry is preserved while still being bounded.
        progress_norm_scale = self.env.rew.get('progress_norm_scale', 0.05)
        delta_d = last_d - curr_d   # +ve when getting closer
        retreat_mult = self.env.rew['progress_retreat_multiplier']
        delta_d = torch.where(delta_d >= 0, delta_d, delta_d * retreat_mult)
        base_progress = torch.tanh(delta_d / progress_norm_scale)
        progress = base_progress

        # Between gate 2 and 3: replace progress with a blend of height climbing
        # and gate-3 approach.  Two regimes:
        #   - Before apex: 80% height, 20% gate-3 → primarily incentivise climbing
        #   - After apex:  25% height, 75% gate-3 → dense pull toward gate 3
        # Outside the powerloop segment (any other gate pair) progress is unchanged.
        in_powerloop_segment = (self.env._idx_wp == 3) & self._crossed_gate2

        # Height term: Gaussian centered on target apex height.
        # Peaks at 1.0 when drone is right at the apex, drops off above or below.
        target_loop_height   = gate2_z + loop_height_threshold
        height_error         = torch.abs(drone_z - target_loop_height)
        loop_height_progress = torch.exp(-height_error / 0.3)

        # Gate-3 approach: reuse base_progress which already includes retreat multiplier.
        dense_loop_progress = torch.where(
            self._above_gate2,
            0.25 * loop_height_progress + 0.75 * base_progress,
            0.8  * loop_height_progress + 0.2  * base_progress,
        )

        progress = torch.where(in_powerloop_segment, dense_loop_progress, progress)

        # Keep real curr distance in _last_distance_to_goal so future checks use true pose
        self.env._last_distance_to_goal = curr_d.clone()

        # Lap time bonus: exp((target - lap_elapsed) / constant)
        #   faster than target → exp(+) > 1 → extra reward
        #   at target          → exp(0) = 1 → full bonus
        #   slower than target → exp(-) < 1 → decaying toward zero
        # Uses _lap_elapsed snapshot taken in _update_gate_state BEFORE the
        # lap timer is reset, so we see the actual lap duration.
        lap_time_bonus = self.env.rew['lap_time_bonus'] * torch.exp(
            (self.env.rew['lap_target_time'] - self.env._lap_elapsed) / self.env.rew['lap_time_constant']
        )

        # Crash: sustained contact force after a 100-step grace period
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask    = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        if self.cfg.is_train:
            rewards = {
                # Sparse: gate crossing reward with 3 explicit power-loop checkpoints:
                #   Stage 1 — gate 2 cross:   +scale
                #   Stage 2 — loop apex:      +scale*0.5 (one-shot milestone for climbing)
                #   Stage 3 — gate 3 cross:   +scale if apex done, else 0
                "gate_cross":      self._powerloop_gate_cross_reward(crossed, loop_apex_just_reached),
                # Sparse: flat +reward on completing a full lap
                "lap_complete":    lap_completed.float()   * self.env.rew['lap_complete_reward_scale'],
                # Sparse: extra lap reward — higher for faster laps
                "lap_time":        lap_completed.float()   * lap_time_bonus,
                # Sparse: penalty for wrong gate or wrong direction crossing
                "wrong_crossing":  wrong_crossing.float()  * self.env.rew['wrong_crossing_reward_scale'],
                # Dense negative: penalise contact each step (after grace period)
                "crash":           crashed.float()         * self.env.rew['crash_reward_scale'],
                
                # Dense: delta distance to gate
                "progress":        progress                * self.env.rew['progress_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                 torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def _powerloop_gate_cross_reward(
        self, crossed: torch.Tensor, loop_apex_just_reached: torch.Tensor
    ) -> torch.Tensor:
        """Gate crossing reward encoding three explicit power-loop checkpoints.

        Stage 1 — gate 2 forward crossing:  +scale        (all envs, no condition)
        Stage 2 — loop apex first reached:  +scale * 0.5  (one-shot, incentivises climbing)
        Stage 3 — gate 3 forward crossing:  +scale if apex done, else 0 (no reward, no penalty)

        Total for full loop: +11.5  vs shortcut: +5.0  (with scale=5.0)
        """
        scale    = self.env.rew['gate_cross_reward_scale']
        base_rew = crossed.float() * scale

        # Stage 2: one-shot bonus the step the loop apex is first crossed.
        base_rew = base_rew + loop_apex_just_reached.float() * scale * 0.5

        # Stage 3: gate 3 earns reward only if the arc was completed.
        # After gate 3 is crossed, _idx_wp was advanced to 4.
        # Shortcut → zero the gate_cross component (not negative).
        gate3_just_crossed = crossed & (self.env._idx_wp == 4)
        gate3_shortcut     = gate3_just_crossed & self._crossed_gate2 & ~self._above_gate2
        base_rew = torch.where(gate3_shortcut, base_rew - crossed.float() * scale, base_rew)

        # Reset all power-loop flags once gate 3 is crossed (valid or shortcut).
        self._crossed_gate2[gate3_just_crossed]      = False
        self._above_gate2[gate3_just_crossed]        = False
        self._loop_apex_rewarded[gate3_just_crossed] = False

        return base_rew

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Observation vector (25 dims):
            lin_vel_b         (3)  — linear velocity in body frame
            ang_vel_b         (3)  — body rates (roll/pitch/yaw rate)
            quat_w            (4)  — orientation quaternion in world frame
            pos_wrt_gate      (3)  — drone position in current gate frame
            next_gate_in_gate (3)  — next gate position in current gate frame (lookahead)
            gate_normal_b     (3)  — current gate approach normal in body frame
            prev_actions      (num_prev_action_steps * 4)  — action history, oldest first
            lap_progress      (2)  — (sin, cos) of gate index in lap, periodic across boundary
        """
        n_gates   = self.env._waypoints.shape[0]
        quat_w    = self.env._robot.data.root_quat_w          # [N, 4]

        # Rotation matrices
        rot_w2b = matrix_from_quat(quat_w).transpose(-1, -2)  # world → body [N,3,3]

        # --- Core drone state ---
        lin_vel_b = self.env._robot.data.root_com_lin_vel_b    # [N, 3]
        ang_vel_b = self.env._robot.data.root_ang_vel_b        # [N, 3]

        # --- Current gate relative position (already in gate frame) ---
        pos_wrt_gate = self.env._pose_drone_wrt_gate           # [N, 3]

        # --- Next gate position in current gate frame ---
        next_idx          = (self.env._idx_wp + 1) % n_gates
        curr_gate_pos_w   = self.env._waypoints[self.env._idx_wp, :3]   # [N, 3]
        curr_gate_quat_w  = self.env._waypoints_quat[self.env._idx_wp]  # [N, 4]
        next_gate_pos_w   = self.env._waypoints[next_idx, :3]           # [N, 3]
        next_gate_in_gate, _ = subtract_frame_transforms(
            curr_gate_pos_w, curr_gate_quat_w, next_gate_pos_w
        )                                                                # [N, 3]

        # --- Gate approach normal rotated into body frame ---
        gate_normal_w = self.env._normal_vectors[self.env._idx_wp]      # [N, 3]
        gate_normal_b = torch.bmm(
            rot_w2b, gate_normal_w.unsqueeze(-1)
        ).squeeze(-1)                                                    # [N, 3]

        # --- Action history: shift buffer left, append latest action ---
        self._action_history = torch.roll(self._action_history, -4, dims=1)
        self._action_history[:, -4:] = self.env._previous_actions       # [N, steps*4]

        # --- Lap progress: sin/cos of gate index, periodic across lap boundary ---
        gate_idx  = (self.env._n_gates_passed % n_gates).float()        # [N]
        angle     = 2.0 * np.pi * gate_idx / n_gates
        lap_sin   = torch.sin(angle).unsqueeze(1)                       # [N, 1]
        lap_cos   = torch.cos(angle).unsqueeze(1)                       # [N, 1]

        obs = torch.cat([
            lin_vel_b,          # (3)
            ang_vel_b,          # (3)
            quat_w,             # (4)
            pos_wrt_gate,       # (3) drone position in current gate frame
            next_gate_in_gate,  # (3) next gate center in current gate frame
            gate_normal_b,      # (3) gate approach direction in body frame
            self._action_history,  # (num_prev_action_steps * 4)
            lap_sin,            # (1)
            lap_cos,            # (1)
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
        self._action_history[env_ids] = 0.0
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

        # TODO ----- START -----
        n_gates = self.env._waypoints.shape[0]
        cfg     = self.cfg

        # --- Gate selection ---
        if cfg.random_gate_spawn:
            waypoint_indices = torch.randint(
                0, n_gates, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype
            )
        else:
            waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)

        # --- Gate world pose ---
        x0_wp = self.env._waypoints[waypoint_indices, 0]
        y0_wp = self.env._waypoints[waypoint_indices, 1]
        z_wp  = self.env._waypoints[waypoint_indices, 2]
        theta = self.env._waypoints[waypoint_indices, -1]   # gate yaw in world frame

        # --- Per-env spawn offsets in gate-local frame ---
        # x_local is negative (behind the gate along its approach axis)
        x_local = torch.empty(n_reset, device=self.device).uniform_(-cfg.spawn_dist_max, -cfg.spawn_dist_min)
        y_local = torch.empty(n_reset, device=self.device).uniform_(-cfg.spawn_lateral,   cfg.spawn_lateral)
        z_local = torch.empty(n_reset, device=self.device).uniform_(-cfg.spawn_vertical,  cfg.spawn_vertical)

        # --- Rotate gate-local offset into world frame ---
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local

        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_wp  + z_local

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # --- Per-env yaw: point toward gate + noise ---
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        yaw_noise   = torch.empty(n_reset, device=self.device).uniform_(-cfg.spawn_yaw_noise, cfg.spawn_yaw_noise)
        zeros       = torch.zeros(n_reset, device=self.device)
        quat = quat_from_euler_xyz(zeros, zeros, initial_yaw + yaw_noise)
        default_root_state[:, 3:7] = quat

        # --- Optional: initial forward velocity toward gate ---
        if cfg.spawn_vel_max > 0.0:
            fwd_speed    = torch.empty(n_reset, device=self.device).uniform_(0.0, cfg.spawn_vel_max)
            gate_normal  = self.env._normal_vectors[waypoint_indices]          # [n_reset, 3]
            default_root_state[:, 7:10] = gate_normal * fwd_speed.unsqueeze(1)
        else:
            default_root_state[:, 7:10] = 0.0
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

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

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        # _n_gates_passed set to spawn gate index → sin/cos lap progress obs correct at spawn.
        # _gates_since_spawn always starts at 0 → used for lap completion detection.
        self.env._n_gates_passed[env_ids] = waypoint_indices
        self.env._gates_since_spawn[env_ids] = 0
        self.env._lap_start_step[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        # Init _last_distance_to_goal from Euclidean distance to current gate
        # so the progress term reflects change in norm (getting closer/farther).
        self.env._last_distance_to_goal[env_ids] = torch.norm(
            self.env._pose_drone_wrt_gate[env_ids], dim=-1
        ).clone()

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0
        # Initialize _prev_x_all_gates to actual x-positions relative to all gates
        # so that the first step doesn't see a false +ve→-ve transition.
        if self.env._prev_x_all_gates is not None:
            n_gates = self.env._waypoints.shape[0]
            drone_pos = self.env._robot.data.root_link_state_w[env_ids, :3]  # [n_reset, 3]
            n_reset_ids = drone_pos.shape[0]
            drone_exp    = drone_pos.unsqueeze(1).expand(-1, n_gates, -1).reshape(-1, 3)
            gate_pos_exp = self.env._waypoints[:, :3].unsqueeze(0).expand(n_reset_ids, -1, -1).reshape(-1, 3)
            gate_qat_exp = self.env._waypoints_quat.unsqueeze(0).expand(n_reset_ids, -1, -1).reshape(-1, 4)
            pos_all, _ = subtract_frame_transforms(gate_pos_exp, gate_qat_exp, drone_exp)
            pos_all = pos_all.reshape(n_reset_ids, n_gates, 3)
            self.env._prev_x_all_gates[env_ids] = pos_all[:, :, 0]

        self.env._crashed[env_ids] = 0
        self._crossed_gate2[env_ids]      = False
        self._above_gate2[env_ids]        = False
        self._loop_apex_rewarded[env_ids] = False

        # --- Domain randomization: sample dynamics per episode from eval ranges ---
        if self.cfg.domain_randomization:
            n = len(env_ids)
            cfg = self.cfg
            def _u(lo, hi):
                return torch.empty(n, device=self.device).uniform_(lo, hi)

            # Aerodynamics: ±2× nominal
            self.env._K_aero[env_ids, :2] = _u(cfg.k_aero_xy * 0.5, cfg.k_aero_xy * 2.0).unsqueeze(1)
            self.env._K_aero[env_ids, 2]  = _u(cfg.k_aero_z * 0.5,  cfg.k_aero_z * 2.0)

            # TWR: ±5%
            self.env._thrust_to_weight[env_ids] = _u(cfg.thrust_to_weight * 0.95, cfg.thrust_to_weight * 1.05)

            # PID roll/pitch: kp ±15%, ki ±15%, kd ±30%
            self.env._kp_omega[env_ids, :2] = _u(cfg.kp_omega_rp * 0.85, cfg.kp_omega_rp * 1.15).unsqueeze(1)
            self.env._ki_omega[env_ids, :2] = _u(cfg.ki_omega_rp * 0.85, cfg.ki_omega_rp * 1.15).unsqueeze(1)
            self.env._kd_omega[env_ids, :2] = _u(cfg.kd_omega_rp * 0.70, cfg.kd_omega_rp * 1.30).unsqueeze(1)

            # PID yaw: kp ±15%, ki ±15%, kd ±30%
            self.env._kp_omega[env_ids, 2] = _u(cfg.kp_omega_y * 0.85, cfg.kp_omega_y * 1.15)
            self.env._ki_omega[env_ids, 2] = _u(cfg.ki_omega_y * 0.85, cfg.ki_omega_y * 1.15)
            self.env._kd_omega[env_ids, 2] = _u(cfg.kd_omega_y * 0.70, cfg.kd_omega_y * 1.30)
