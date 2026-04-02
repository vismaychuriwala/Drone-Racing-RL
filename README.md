# Drone Racing RL — ESE 6510

Powerloop track, 7 gates. Train a Crazyflie racing policy with PPO in Isaac Lab.
Policy at 50 Hz. Sim at 500 Hz (10× decimation). PID inner loop at 500 Hz.

---

## Setup

```bash
# Isaac Lab and project repo must be at the same directory level
git clone git@github.com:Jirl-upenn/ese651_project.git
conda activate env_isaaclab
```

---

## Training

```bash
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 8192 \              # reduce if OOM
    --max_iterations 1000 \
    --headless \
    --logger wandb                 # remove if no wandb
    --seed 42 \                    # optional
    --device cuda:0 \              # optional
    --video \                      # record videos during training
    --video_length 200 \           # steps per video
    --video_interval 2000          # iterations between videos
```

Logs saved to `logs/rsl_rl/quadcopter_direct/YYYY-MM-DD_HH-MM-SS/`.

---

## Evaluation

```bash
python scripts/rsl_rl/play_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 1 \
    --load_run YYYY-MM-DD_HH-MM-SS \   # dir under logs/rsl_rl/quadcopter_direct/
    --checkpoint best_model.pt \
    --headless \
    --video \
    --video_length 800
```

---

## Files — what to touch vs what not to

| File | Status | Notes |
|---|---|---|
| `quadcopter_strategies.py` | **OURS** | rewards, obs, reset |
| `train_race.py` reward dict + env_cfg overrides | **OURS** | coefficients, spawn config |
| `ppo.py` `update()` | **OURS** | PPO implementation |
| `quadcopter_env.py` | mostly boilerplate | added `_update_gate_state()`, spawn cfg fields |
| `rollout_storage.py` | boilerplate | GAE lives here |
| `actor_critic.py` | boilerplate | MLP + Gaussian head |
| `on_policy_runner.py` | boilerplate | training loop |

---

## Action Space

4-dim continuous: `[collective_thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]`
- Thrust scaled: `((a[0]+1)/2) * robot_weight * thrust_to_weight`
- Body rates → PID → motor speeds → forces via `set_external_force_and_torque`
- Action smoothing: `a_applied = β·a_new + (1-β)·a_prev`
- Not high-level ("move forward 1m") — motor-level commands through a PID layer

---

## Step Cycle (per policy step = 10 sim steps)

```
_pre_physics_step   clamp+smooth actions, set thrust
_apply_action       10× physics: PID → motor speeds → forces + aero drag
_get_dones          _update_gate_state() → crossing, then termination check
_get_rewards        read crossed/lap flags, compute reward
_get_observations   build obs vector
```

Gate crossing and `_idx_wp` advancement in `_update_gate_state` (called from `_get_dones`) —
before rewards and obs. No 1-step lag.

---

## Gate Crossing Detection

Direction-enforced. x-axis of gate local frame = approach direction (gate normal).
All 7 gates checked in one batched call per step. Classifies as good (correct gate + direction)
or wrong (wrong gate or wrong direction).

```python
# Drone position relative to ALL gates computed each step
fwd_cross = (prev_x > 0) & (x_now <= 0) & within_bounds   # per gate
bwd_cross = (prev_x <= 0) & (x_now > 0) & within_bounds   # per gate
good_cross  = (fwd_cross & target_mask).any(dim=1)
wrong_cross = ((fwd_cross & ~target_mask) | bwd_cross).any(dim=1) & ~good_cross
```

`_prev_x_all_gates` initialized to actual x-positions on reset (not blanket 1.0) to avoid
false crossings on step 1. `good_cross` suppresses `wrong_cross` on the same step — needed
because co-located opposite-direction gates (e.g. powerloop gates 3 and 6) would otherwise
fire a spurious backward crossing whenever the correct gate is traversed.

---

## Rewards

Configured in `train_race.py`:

```python
rewards = {
    'gate_cross_reward_scale':     5.0,    # sparse: per correct gate traversal
    'lap_complete_reward_scale':  25.0,    # sparse: flat reward per full lap
    'lap_time_bonus':              1.0,    # sparse: extra lap reward for fast laps
    'lap_target_time':             7.0,    # seconds — target lap time
    'lap_time_constant':           3.0,    # seconds — decay steepness
    'crash_reward_scale':       -0.005,    # dense: per-step contact, after 100-step grace
    'death_cost':                 -1.0,    # terminal: on episode death (crash/altitude/wrong crossing)
    'timeout_cost':               -0.5,    # terminal: on episode timeout (discourages idling)
    'progress_reward_scale':      0.02,    # dense: delta distance to gate center
    'progress_retreat_multiplier': 1.005,  # retreat penalized slightly harder than approach
}
```

**Terminal conditions:**
- **Death (-1.0):** crash (sustained contact > 100 steps), altitude violation, or wrong gate crossing.
  Wrong crossing is now purely terminal — no separate reward param. Episode ends immediately.
- **Timeout (-0.5):** milder than death so the policy still prefers surviving, but idling out the
  clock is penalized. Without this, the policy learns to hover and avoid risk.

**Crash fires on gate-frame contact too** — kept very small (-0.005) to avoid penalizing gate brushes.

**Progress reward:**
```python
progress = last_dist - curr_dist                            # +ve = closing distance
progress = where(progress >= 0, progress, progress * 1.005) # retreat penalized slightly more
```
Euclidean distance delta to gate center. Scale=0.02, retreat multiplier nearly symmetric at 1.005.
Progress just nudges toward the gate — gate crossing reward does the heavy lifting.

**Lap time bonus:** `bonus * exp((target_time - lap_elapsed) / constant)`:
- Lap in 4s: `1 * exp(1.0)` = 2.7 bonus (on top of flat 25)
- Lap in 7s (target): `1 * exp(0)` = 1.0 bonus
- Lap in 10s: `1 * exp(-1)` = 0.37 bonus
- Uses per-lap elapsed time (not episode time) — 2nd/3rd laps get fair timing.
- 30s episode fits ~3 laps at target pace. Constant=3s gives moderate decay.

**Wrong crossing** is detected but not penalized via reward — instead triggers episode termination
(death cost applies). `good_cross` suppresses `wrong_cross` on the same step to prevent spurious
penalties on co-located opposite-direction gates (powerloop gates 3/6).

---

## Observations (25 dims, `num_prev_action_steps=1`)

| Feature | Dims | Frame | Notes |
|---|---|---|---|
| `lin_vel_b` | 3 | body | linear velocity |
| `ang_vel_b` | 3 | body | body rates — critical for rotational dynamics |
| `quat_w` | 4 | world | orientation |
| `pos_wrt_gate` | 3 | gate | drone pos in current gate frame. x=along normal, y/z=centering |
| `next_gate_in_gate` | 3 | gate | next gate center in current gate frame — lookahead |
| `gate_normal_b` | 3 | body | gate approach direction in drone body frame |
| `prev_actions` | 4×N | — | action history oldest first. N=`num_prev_action_steps` |
| `lap_sin`, `lap_cos` | 2 | — | sin/cos of gate index — periodic, no boundary discontinuity |

**Removed:** `drone_pose_w` — leaks absolute track position, hurts generalization.

**`pos_wrt_gate` is position only** — name is misleading. It's `[x,y,z]` from `subtract_frame_transforms`,
orientation output discarded. Does not encode which way the drone faces.

**Lap progress obs:** `gate_idx = n_gates_passed % n_gates`. On reset, `n_gates_passed` initialized
to `waypoint_indices` (not 0) → sin/cos correct at any random spawn gate.

**Lap detection uses `_gates_since_spawn`** (separate counter, always starts at 0) — NOT
`_n_gates_passed`. Without this, spawning at gate 3 fires lap after only 4 crossings because
`(3+4) % 7 == 0`. `_gates_since_spawn` fires correctly after 7 crossings regardless of spawn gate.

**Why sin+cos not scalar:** `cos(θ)` alone is symmetric — gate 1 and gate 6 give same value.
`[0,π]` range doesn't fix it — same ambiguity. Need both for unique circular encoding.

---

## Reset Strategy

All params in `QuadcopterEnvCfg`, override in `train_race.py` via `env_cfg.X = Y`:

```python
random_gate_spawn: bool  = True   # uniform random gate 0-6; False = always gate 0
spawn_dist_min:    float = 1.0    # m behind gate along approach axis
spawn_dist_max:    float = 3.0
spawn_lateral:     float = 0.3    # ±m lateral offset from gate center
spawn_vertical:    float = 0.2    # ±m vertical offset
spawn_yaw_noise:   float = 0.3    # ±rad yaw perturbation, per-env
spawn_vel_max:     float = 0.0    # m/s forward velocity toward gate; 0=disabled
```

**Why random gate:** without it, policy never trains gates 4-6 unless it first clears 0-3.
Powerloop (gates 1-2) only trained when policy already good. Random = equal coverage.

**Spawn sign convention:** `x_local` is negative (behind gate). Rotation gives negative world offset.
`x0_wp - x_rot` double-negates → places drone on approach side (positive x in gate frame).
Using `+` instead of `-` would place drone on exit side and trigger false gate crossing on step 1.

**Yaw fix from original:** original had `torch.empty(1).uniform_()` → same yaw broadcast to all
n_reset envs. Fixed to `torch.empty(n_reset)` for per-env perturbation.

**Velocity:** `default_root_state[:, 7:10] = gate_normal * fwd_speed`. Disabled by default.
Enable phase 2. Simulates carry-over speed from previous gate exit.

---

## Domain Randomization

**Enabled** (`env_cfg.domain_randomization = True` in `train_race.py`). Per-episode
sampling in `reset_idx` from the eval ranges specified in the project handout:

```
Aerodynamics:  k_aero_xy × [0.5, 2.0],  k_aero_z × [0.5, 2.0]
TWR:           thrust_to_weight × [0.95, 1.05]
PID roll/pitch: kp × [0.85, 1.15],  ki × [0.85, 1.15],  kd × [0.70, 1.30]
PID yaw:        kp × [0.85, 1.15],  ki × [0.85, 1.15],  kd × [0.70, 1.30]
```

Each reset env gets independently sampled dynamics. `is_train`, `domain_randomization`, and
`rewards` must be explicitly set in train/play scripts (default `None` in cfg — raises error if
unset). When `False`, dynamics are fixed at nominal values set in `__init__`. The eval environment
samples from these exact ranges — training with DR forces the policy to be robust to dynamics
mismatch.

---

## PPO (`ppo.py update()`)

```python
ratio         = exp(log_prob_new - log_prob_old)
surr_loss     = -min(ratio * A, clamp(ratio, 1-ε, 1+ε) * A).mean()
value_clipped = V_target + clamp(V - V_target, -ε, ε)
value_loss    = max((V - R)², (V_clipped - R)²).mean()
loss          = surr_loss + value_loss_coef * value_loss - entropy_coef * entropy
```

Adaptive KL LR: KL > 2×desired → lr /= 1.5. KL < 0.5×desired → lr *= 1.5. Bounds [1e-5, 1e-2].

**Bug fixed (commit a6da75d):** missing `.squeeze(-1)` on `actions_log_prob`, `prev_log_probs`,
`advantage_estimates`. Without it `[B,1] * [B,1]` broadcasts to `[B,B]` — wrong gradients + OOM.

Key hyperparams (`dummy_config.yaml`):
- `num_steps_per_env: 24` — rollout window (0.48s at 50Hz). Short → limited GAE horizon for sparse rewards.
- `num_mini_batches: 4` → mini-batch ≈ 49k transitions (8192×24/4)
- `num_learning_epochs: 5`, `gamma: 0.998`, `lam: 0.95`

OOM history: tried `num_mini_batches=16, num_steps=16` (commit 105ceb7) → reverted (commit a6da75d).
Real fix was the `.squeeze(-1)` bug, not batch size.

---

## Planned Extensions

### Phase 2 — after base policy completes laps
- Gate-centering reward: penalize `y² + z²` at crossing moment
- Smoothness penalty: `(a_t - a_{t-1})².sum()`, scale ~0.01
- Alive bonus: +0.01/step
- Tune `lap_time_bonus` / `lap_target_time` / `lap_time_constant` once laps are reliable

### Phase 3 — robustness
- Observation noise: Gaussian noise on obs at train time
- Increase `num_steps_per_env` 24→48 or 96. Reduce `num_envs` proportionally.
- Entropy annealing: `entropy_coef` 0.005 → 0 over training

### Architecture experiments
- **Asymmetric actor-critic:** critic gets privileged dynamics params. Actor normal obs only. Pass different obs to `evaluate()`. Easy, free improvement.
- **RMA:** phase 1 with privileged dynamics in obs. Phase 2 encoder(obs-action history) → latent dynamics. High effort, best eval robustness.
- **GRU:** `is_recurrent=True` already supported. Hypothesis: won't beat MLP+action_history. Worth 1 ablation.
- **Second lookahead gate (N+2):** powerloop needs vertical loop plan 2 gates ahead.
- **Ensemble:** train 3-5 seeds, average actions at eval. Simple robustness.

---

## Rejected / Deferred — concerns tracked

| Idea | Decision | Concern |
|---|---|---|
| Single scalar lap progress (e.g. 2.7) | Rejected | Euclidean inter-gate dist wrong for powerloop (gates 1-2 are 1.25m apart but require vertical loop). Boundary discontinuity 6.9→0. |
| Single cos with [0,π] range | Rejected | cos(0)=cos(2π) regardless of range. Periodicity survives rescaling. |
| Heading/yaw error as gate obs | Rejected | Heading loses y/z centering. Normalized direction loses distance. Full 3D pos_wrt_gate strictly better. |
| Crash reward as primary deterrent | Kept tiny (-0.005) | Fires on gate-frame contact. Death cost is real deterrent. |
| Separate wrong crossing reward | Replaced by terminal death | Double-penalizing (reward + death) was redundant. Now wrong crossing = episode death, death_cost applies. Cleaner signal. |
| Velocity reward (dot with gate normal) | Replaced by progress | Fixed gate normal is wrong when drone is past gate. Using drone-to-gate direction collapses to delta-distance (progress). Progress is simpler, cheaper, correct everywhere. |
| LSTM/Transformer for longer rollouts | Deferred | Recurrence doesn't enable longer rollouts — BPTT still truncated at rollout boundary. Memory bottleneck same. Overkill given good obs design. |
| KL-penalty PPO instead of clipped | Rejected | Needs extra β tuning. Adaptive KL LR schedule already gives most of benefit. |
| Time-decaying lap bonus as primary speed signal | Active (small scale) | Scale=1.0, target=7s, constant=3s. Uses per-lap timer, not episode time. Increase scale in phase 2 once laps reliable. |
| Initial velocity at spawn in phase 1 | Deferred to phase 2 | Policy crashes immediately if it never learned to handle entry velocity. Add after base policy works. |
