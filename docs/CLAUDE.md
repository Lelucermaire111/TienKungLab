# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TienKung-Lab is a reinforcement learning locomotion framework for the TienKung full-sized humanoid robot. It trains walking/running policies using **AMP-PPO** (Adversarial Motion Priors + Proximal Policy Optimization) inside NVIDIA Isaac Sim, then validates via MuJoCo Sim2Sim and deploys on real hardware.

## Installation

```bash
# Install main package
pip install -e .

# Install embedded RSL-RL fork
cd rsl_rl && pip install -e .
```

Requires: Python 3.10, Isaac Sim 4.5.0 / Isaac Lab 2.1.0, MuJoCo 3.3.2.

## Common Commands

**Training:**
```bash
python legged_lab/scripts/train.py --task=lite_walk --headless --logger=tensorboard --num_envs=4096
python legged_lab/scripts/train.py --task=lite_run  --headless --logger=tensorboard --num_envs=4096
```

**Playback (requires display):**
```bash
python legged_lab/scripts/play.py --task=lite_walk --num_envs=1
python legged_lab/scripts/play_amp_animation.py --task=lite_walk --num_envs=1
```

**Sim2Sim (MuJoCo):**
```bash
python legged_lab/scripts/sim2sim.py --task walk --policy Exported_policy/walk.pt --duration 100
```

**Linting:**
```bash
pre-commit run --all-files
```

## Architecture

### Package Layout

- **`legged_lab/`** — main application package
- **`rsl_rl/`** — embedded fork of RSL-RL v2.3.1 (RL algorithms; do not confuse with upstream)
- **`Exported_policy/`** — pretrained TorchScript checkpoints (`walk.pt`, `policy.pt`)

### Data Flow

1. `train.py` looks up a task name in `TaskRegistry` → gets `(EnvClass, EnvCfg, AgentCfg)`
2. Instantiates the env (e.g. `TienKungEnv`) in Isaac Sim with robot articulation, contact sensors, ray casters
3. Selects a runner (`OnPolicyRunner` or `AmpOnPolicyRunner`) based on `agent_cfg.runner_class_name`
4. AMP runner trains with PPO rollouts + a discriminator comparing agent motion to reference data in `datasets/motion_amp_expert/`
5. Policy is saved as TorchScript `.pt`; `sim2sim.py` loads it to drive a MuJoCo model via PD control

### Key Modules

| Module | Role |
|---|---|
| `legged_lab/utils/task_registry.py` | Global registry mapping task name → `(EnvClass, EnvCfg, AgentCfg)` |
| `legged_lab/envs/base/base_env.py` | Abstract `BaseEnv` (VecEnv subclass): sim setup, robot, sensors, reward/event managers |
| `legged_lab/envs/base/base_config.py` | All `@configclass` sub-configs: scene, sim, robot, commands, normalization, noise, etc. |
| `legged_lab/envs/tienkung/tienkung_env.py` | Concrete env: AMP integration, cameras, curriculum |
| `legged_lab/envs/tienkung/walk_cfg.py` | Walk task env + agent configs (gait, rewards, events) |
| `legged_lab/mdp/rewards.py` | Custom reward functions (gait periodicity, AMP discriminator score, etc.) |
| `legged_lab/mdp/events.py` | Domain randomization events |
| `rsl_rl/algorithms/amp_ppo.py` | AMP-PPO algorithm (PPO + discriminator) |
| `rsl_rl/runners/amp_on_policy_runner.py` | Training loop for AMP-PPO |

### Robot Variants

- **`lite`** — TienKung Lite humanoid (`legged_lab/assets/tienkung2_lite/`, tasks: `lite_walk`, `lite_run`)
- **`dex`** — EVT2 dexterous hands variant (`legged_lab/assets/EVT2/`, tasks: `dex_walk`, `dex_run`)

Each variant has its own `envs/<variant>/` directory with env class, task configs, and `datasets/` (AMP expert data + visualization data).

### Config System

Configs are composed `@configclass` dataclasses (Isaac Lab pattern). Task configs inherit from `BaseEnvCfg` and `BaseAgentCfg`. To add a new task: create env cfg + agent cfg dataclasses, register them in `legged_lab/envs/__init__.py` via `task_registry.register_task()`.

### AMP Motion Data

Two datasets per task:
- `datasets/motion_amp_expert/` — used during training as discriminator reference
- `datasets/motion_visualization/` — used by `play_amp_animation.py` for visualization

Data format: `.txt` files with per-frame joint positions/velocities. Generated via the motion retargeting pipeline (SMPLX → GMR → `gmr_data_conversion.py`).

## Code Style

Enforced by pre-commit: **black** (line-length 120), **flake8**, **isort** (profile: black), **pyupgrade** (py37+), **codespell**. License headers are auto-inserted.
