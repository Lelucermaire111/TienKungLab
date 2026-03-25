# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import os
import time  # 添加时间模块用于速度统计

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg_, agent_cfg_ = task_registry.get_cfgs(env_class_name)

    env_cfg = env_cfg_()
    agent_cfg = agent_cfg_()
    
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events = None
    env_cfg.scene.max_episode_length_s = 1000.0  # 设置为很大的值以测试长期稳定性
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    # 固定前向速度为 3.0 m/s，用于测试高速稳定奔跑能力
    env_cfg.commands.ranges.lin_vel_x = (3.5, 3.5)  # 固定 3.5 m/s
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)   # 固定 0，直线奔跑
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)   # 固定 0，不转向
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()

    # ===== 速度监测与稳定时间统计初始化 =====
    print("[INFO] 速度监测已启用 - 目标速度: 3.0 m/s")
    print("[INFO] 稳定奔跑判定: 未倒地即可")
    start_time = time.time()
    all_speeds = []
    all_errors = []
    step_counter = 0

    # 稳定时间跟踪 - 仅以未倒地作为判定条件
    target_speed = 3.5
    # 每个环境的稳定步数计数
    stable_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    # 记录每个环境的最大稳定时间
    max_stable_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    # 记录每个环境是否当前处于稳定状态
    is_stable = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    # ==========================================

    while simulation_app.is_running():

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, terminated, _ = env.step(actions)

            # ===== 速度与稳定时间统计 =====
            step_counter += 1
            # 获取实际速度（机身坐标系）
            actual_vel = env.robot.data.root_lin_vel_b
            target_vel = env.command_generator.command

            # 计算前向速度误差（只考虑 x 方向）- 仅用于统计，不用于稳定判定
            forward_speed_error = torch.abs(target_vel[:, 0] - actual_vel[:, 0])
            # 判断当前步是否稳定: 仅以未倒地作为判定条件
            current_stable = ~terminated

            # 更新稳定步数统计
            # 如果当前稳定，增加计数；否则重置为0
            stable_steps = torch.where(current_stable, stable_steps + 1, torch.zeros_like(stable_steps))
            # 更新最大稳定步数
            max_stable_steps = torch.maximum(max_stable_steps, stable_steps)
            # 记录是否处于稳定状态
            is_stable = current_stable

            # 计算统计信息
            speed_2d = actual_vel[:, 0].mean().item()  # 前向速度平均值
            avg_error = forward_speed_error.mean().item()
            num_stable = current_stable.sum().item()

            all_speeds.append(speed_2d)
            all_errors.append(avg_error)

            # 每2秒输出一次统计
            if time.time() - start_time > 2.0:
                avg_speed = sum(all_speeds) / len(all_speeds)
                avg_err = sum(all_errors) / len(all_errors)
                max_forward_speed = max(all_speeds)
                min_forward_speed = min(all_speeds)

                # 计算当前最稳定的环境
                best_env = max_stable_steps.argmax().item()
                best_stable_steps = max_stable_steps[best_env].item()
                best_stable_time = best_stable_steps * env.step_dt
                current_best_time = stable_steps[best_env].item() * env.step_dt

                # 存活环境数量
                num_alive = current_stable.sum().item()

                print(f"\n{'='*60}")
                print(f"[Velocity Stats] 统计周期: {len(all_speeds)} 步 | 总步数: {step_counter}")
                print(f"  目标速度: {target_speed:.1f} m/s")
                print(f"  存活环境: {num_alive}/{env.num_envs}")
                if num_alive > 0:
                    alive_avg_speed = actual_vel[current_stable, 0].mean().item()
                    alive_max_speed = actual_vel[current_stable, 0].max().item()
                    alive_min_speed = actual_vel[current_stable, 0].min().item()
                    print(f"  存活环境速度: 平均={alive_avg_speed:.3f} | 最大={alive_max_speed:.3f} | 最小={alive_min_speed:.3f} m/s")
                print(f"  总体速度: 平均={avg_speed:.3f} | 最大={max_forward_speed:.3f} | 最小={min_forward_speed:.3f} m/s")
                print(f"  跟踪误差: {avg_err:.3f} m/s")
                print(f"\n[Stable Run Stats]")
                print(f"  全局最大稳定时间: {(max_stable_steps.max().item() * env.step_dt):.2f}s")
                print(f"  最佳环境 (Env {best_env}): 当前={current_best_time:.2f}s | 历史最大={best_stable_time:.2f}s")

                # 重置统计
                all_speeds = []
                all_errors = []
                start_time = time.time()
            # ==============================


if __name__ == "__main__":
    play()
    simulation_app.close()
