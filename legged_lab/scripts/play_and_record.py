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

"""Play and record video of trained policy.

IMPORTANT: This script maintains exact environment configuration compatibility with play.py
 to avoid model loading size mismatch errors.

Usage:
    python legged_lab/scripts/play_and_record.py --task=lite_walk --video_path=output.mp4

Examples:
    # Record 10s video with default settings
    python legged_lab/scripts/play_and_record.py --task=lite_walk --load_run=2025-03-24_01-31-26 --load_checkpoint=model_11200.pt

    # Record with custom duration and FPS
    python legged_lab/scripts/play_and_record.py --task=lite_walk \
        --video_path=walk_demo.mp4 --video_duration=15 --video_fps=30
"""

import argparse
import os

import torch
import numpy as np
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Play and record video of trained policy.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video_path", type=str, default="output.mp4", help="Output video path.")
parser.add_argument("--video_fps", type=int, default=30, help="Video FPS.")
parser.add_argument("--video_duration", type=float, default=10.0, help="Video duration in seconds.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Start camera rendering (same condition as play.py)
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def save_frame_as_image(frame_bgr, output_path):
    """Save a BGR frame as PNG image."""
    try:
        import cv2
        cv2.imwrite(output_path, frame_bgr)
        return True
    except Exception as e:
        print(f"[WARN] Failed to save frame: {e}")
        return False


def capture_viewport_frame():
    """Capture current viewport frame.

    Returns:
        np.ndarray: BGR image or None if capture failed
    """
    try:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is None:
            print("[DEBUG] get_active_viewport() returned None")
            return None

        print(f"[DEBUG] Viewport obtained: {viewport}")

        # Try different API methods for Isaac Sim 4.5.0
        frame = None

        # Method 1: capture_frame()
        if hasattr(viewport, 'capture_frame'):
            print("[DEBUG] Trying capture_frame()...")
            frame = viewport.capture_frame()
            if frame is not None:
                print(f"[DEBUG] capture_frame() success: {type(frame)}")
            else:
                print("[DEBUG] capture_frame() returned None")
        # Method 2: get_drawable().capture_frame()
        elif hasattr(viewport, 'get_drawable'):
            print("[DEBUG] Trying get_drawable().capture_frame()...")
            drawable = viewport.get_drawable()
            if drawable and hasattr(drawable, 'capture_frame'):
                frame = drawable.capture_frame()

        if frame is None:
            print("[DEBUG] Frame is None after all methods")
            return None

        if not hasattr(frame, 'data'):
            print(f"[DEBUG] Frame has no 'data' attribute: {type(frame)}, attrs={dir(frame)}")
            return None

        if frame.data is None:
            print("[DEBUG] frame.data is None")
            return None

        # Convert to numpy
        import cv2
        print(f"[DEBUG] Converting frame: shape=({frame.height}, {frame.width}), data_len={len(frame.data)}")
        image = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 4)
        # RGBA -> BGR
        image_bgr = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)
        print(f"[DEBUG] Frame converted successfully: {image_bgr.shape}")
        return image_bgr

    except Exception as e:
        print(f"[DEBUG] capture_viewport_frame exception: {e}")
        import traceback
        traceback.print_exc()
        return None


def play_and_record():
    """Main play and record function.

    NOTE: Environment configuration must match play.py exactly to avoid
    size mismatch errors during model loading.
    """
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg_, agent_cfg_ = task_registry.get_cfgs(env_class_name)

    env_cfg = env_cfg_()
    agent_cfg = agent_cfg_()

    # EXACT same environment configuration as play.py
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (0.0, 1.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
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

    # Create environment
    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    # Load model (same as play.py)
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    # Export model (same as play.py)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # Keyboard control (same as play.py)
    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard
        keyboard = Keyboard(env)  # noqa:F841

    # ==================== VIDEO RECORDING SETUP ====================
    video_path = args_cli.video_path
    video_fps = args_cli.video_fps
    video_duration = args_cli.video_duration
    max_frames = int(video_duration * video_fps)

    # Create frames directory
    frames_dir = video_path.replace('.mp4', '_frames')
    if os.path.exists(frames_dir):
        import shutil
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    print(f"[INFO] Recording {max_frames} frames ({video_duration}s) at {video_fps} FPS")
    print(f"[INFO] Frames will be saved to: {frames_dir}/")

    # Determine render interval based on simulation dt
    render_interval = max(1, int(1.0 / (env.step_dt * video_fps)))

    frame_count = 0
    step_counter = 0

    # Get initial frame to determine resolution
    print("[INFO] Waiting for viewport to initialize...")
    first_frame = None
    max_wait_iterations = 60  # Wait up to ~3 seconds
    for i in range(max_wait_iterations):
        simulation_app.update()
        if i % 10 == 0:
            print(f"[INFO] Waiting for viewport... ({i}/{max_wait_iterations})")
        first_frame = capture_viewport_frame()
        if first_frame is not None:
            break

    if first_frame is not None:
        height, width = first_frame.shape[:2]
        print(f"[INFO] Capture resolution: {width}x{height}")
    else:
        width, height = 1920, 1080
        print(f"[WARN] Could not detect resolution, using default {width}x{height}")
        print("[WARN] Will continue anyway and try to capture during simulation")
    # =============================================================

    # Main simulation loop
    obs, _ = env.get_observations()

    print("[INFO] Starting recording... Press Ctrl+C to stop early")

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

            # ==================== FRAME CAPTURE ====================
            step_counter += 1
            if step_counter >= render_interval and frame_count < max_frames:
                print(f"[DEBUG] Capturing frame {frame_count} (step {step_counter})...")
                frame = capture_viewport_frame()
                if frame is not None:
                    print(f"[DEBUG] Got frame: {frame.shape}")
                    # Resize if needed to maintain consistent resolution
                    if frame.shape[0] != height or frame.shape[1] != width:
                        import cv2
                        frame = cv2.resize(frame, (width, height))

                    # Save frame
                    frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.png")
                    save_success = save_frame_as_image(frame, frame_path)
                    if save_success:
                        frame_count += 1
                        print(f"[DEBUG] Saved frame {frame_count}")
                    else:
                        print(f"[WARN] Failed to save frame to {frame_path}")

                    if frame_count % 30 == 0:
                        progress = 100 * frame_count / max_frames
                        print(f"Progress: {frame_count}/{max_frames} frames ({progress:.1f}%)")
                else:
                    print(f"[DEBUG] Frame capture returned None")

                step_counter = 0

            # Stop after duration
            if frame_count >= max_frames:
                print(f"[INFO] Reached target frame count: {max_frames}")
                break
            # ======================================================

    except KeyboardInterrupt:
        print("\n[INFO] Recording interrupted by user")
    finally:
        # ==================== VIDEO FINALIZATION ====================
        print(f"\n[INFO] Recording complete. Total frames: {frame_count}")

        if frame_count > 0:
            # Try to convert frames to video using imageio
            try:
                import imageio

                frames = []
                for i in range(frame_count):
                    frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
                    frames.append(imageio.imread(frame_path))

                # Save as video
                imageio.mimsave(video_path, frames, fps=video_fps, quality=8)
                print(f"[INFO] Video saved to: {os.path.abspath(video_path)}")

                # Clean up frames
                import shutil
                shutil.rmtree(frames_dir)
                print(f"[INFO] Temporary frames cleaned up")

            except ImportError:
                print(f"[INFO] imageio not available, frames kept in: {frames_dir}/")
                print(f"[INFO] Convert to video manually with:")
                print(f"  ffmpeg -framerate {video_fps} -i {frames_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {video_path}")
            except Exception as e:
                print(f"[WARN] Video conversion failed: {e}")
                print(f"[INFO] Frames kept in: {frames_dir}/")
                print(f"[INFO] Convert manually with FFmpeg command above")
        # ==========================================================


if __name__ == "__main__":
    play_and_record()
    simulation_app.close()
