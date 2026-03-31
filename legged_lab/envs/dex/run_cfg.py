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

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.tiangong_dex_urdf_EVT2 import DEX_V3_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401


@configclass
class GaitCfg:
    """Optimized gait parameters for Sim2Real - more conservative for stability"""
    gait_air_ratio_l: float = 0.55  # Reduced from 0.6 for more contact time
    gait_air_ratio_r: float = 0.55
    gait_phase_offset_l: float = 0.55  # Adjusted for running
    gait_phase_offset_r: float = 0.05
    gait_cycle: float = 0.5  # Faster cycle for running (was 0.64)
    # Add bounds for curriculum
    gait_cycle_lower: float = 0.4
    gait_cycle_upper: float = 0.7


@configclass
class DexEventCfg(EventCfg):
    """TienKung environment event configuration with enhanced domain randomization for Sim2Real"""

    # Inherited attributes
    randomize_pd_gains: EventTerm = None
    randomize_apply_external_force_torque: EventTerm = None
    randomize_rigid_body_com: EventTerm = None
    randomize_joint_parameters: EventTerm = None  # NEW: Joint friction/armature randomization

@configclass
class DexRewardCfg:
    """Optimized rewards for Sim2Real - prioritize stability over speed"""

    # === Task Rewards (reduced weights for more conservative behavior) ===
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.0,  # Reduced from 5.0 - don't over-prioritize speed
        params={"std": 0.6}  # Increased std for more tolerance
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.5,  # Reduced from 2.0
        params={"std": 0.6}
    )

    # === Stability Rewards (INCREASED for Sim2Real robustness) ===
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)  # Increased from -0.5
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)  # Increased from -0.05
    torso_ang_vel_xy_l2 = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link")},
        weight=-1.0  # Increased from -0.5
    )
    torso_ang_acc_xy_l2 = RewTerm(
        func=mdp.body_ang_acc_xy_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link")},
        weight=-2e-4  # Increased from -1e-4
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)  # Increased from -0.5
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
        weight=-0.5  # Increased from -0.25
    )
    waist_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link")},
        weight=-1.5  # Increased from -1.0
    )

    # === Contact & Safety (INCREASED for Sim2Real) ===
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,  # Increased from -0.5
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"]
            ),
            "threshold": 3.0,  # Lower threshold for earlier detection
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,  # Increased from -0.25
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-5e-3,  # Increased from -3e-3
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "threshold": 400,  # Lower threshold
            "max_reward": 300,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,  # Increased from -0.5
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"])},
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-1.0,  # Increased from -0.5
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ankle_roll.*"]), "threshold": 0.18},
    )

    # === Energy & Smoothness (maintained) ===
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)  # Increased from -0.01 for smoother actions

    # === Termination & Survival ===
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)  # Increased penalty for falling
    alive_reward = RewTerm(func=mdp.alive_reward, weight=1.0)  # Increased from 0.5

    # === Joint Regularization ===
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,  # Increased from -0.15
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_yaw_.*_joint",
                    "hip_roll_.*_joint",
                    "shoulder_pitch_.*_joint",
                    "elbow_pitch_.*_joint",
                ],
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,  # Increased from -0.2
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_roll_.*_joint", "shoulder_yaw_.*_joint"])},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,  # Increased from -0.1
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.08,  # Increased from -0.05
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_pitch_.*_joint",
                    "knee_pitch_.*_joint",
                    "ankle_pitch_.*_joint",
                    "ankle_roll_.*_joint",
                ],
            )
        },
    )

    # === Gait Rewards (adjusted for running) ===
    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.5, params={"delta_t": 0.02})  # Reduced from 2.0
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.5, params={"delta_t": 0.02})  # Reduced from 2.0
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=1.0, params={"delta_t": 0.02})  # Reduced from 1.5

    # === Action Regularization ===
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.001)  # Increased from -0.0005
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.002)  # Increased from -0.001
    hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-0.5)  # Increased from -0.3
    hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-0.5)  # Increased from -0.3
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-0.8)  # Increased from -0.5

    # ============================================
    # Running-specific rewards (adjusted for stability)
    # ============================================
    feet_contact_alternation = RewTerm(
        func=mdp.feet_contact_alternation,
        weight=0.8,  # Reduced from 1.0
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"])}
    )

    feet_air_time_reward = RewTerm(
        func=mdp.feet_air_time_reward,
        weight=0.8,  # Reduced from 1.0
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"]),
            "target_time": 0.25  # Reduced from 0.35 for more contact time
        }
    )

    feet_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=0.8,  # Reduced from 1.0
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["ankle_roll.*"]),
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"]),
            "min_height": 0.06  # Reduced from 0.08
        }
    )

    step_frequency_reward = RewTerm(
        func=mdp.step_frequency_reward,
        weight=0.8,  # Reduced from 1.0
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"]),
            "target_freq": 3.5  # Increased from 3.0 for faster running
        }
    )

    forward_velocity_reward = RewTerm(
        func=mdp.forward_velocity_reward,
        weight=1.0  # Reduced from 1.5
    )

    feet_contact_forces_balanced = RewTerm(
        func=mdp.feet_contact_forces_balanced,
        weight=-0.5,  # Increased from -0.3
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"])}
    )


@configclass
class DexRunFlatEnvCfg:
    """Optimized environment configuration for Sim2Real migration"""
    amp_motion_files_display = ["legged_lab/envs/dex/datasets/motion_visualization/run.txt"]
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=DEX_V3_CFG,
        # Use flat terrain for initial Sim2Real training
        terrain_type="plane",
        terrain_generator=None,
        # terrain_type="generator",
        # terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,  # Start from flat terrain
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,  # Disable for Sim2Real
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
        feet_body_names=["ankle_roll.*"],
    )
    reward = DexRewardCfg()
    gait = GaitCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(8.0, 12.0),  # Slightly variable
        rel_standing_envs=0.2,  # Reduced from 0.4 - more training time for running
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # Start with lower speeds, gradually increase with curriculum
        ranges=CommandRangesCfg(
            lin_vel_x=(0.0, 2.0),  # Reduced max from 2.5 - focus on stable running first
            lin_vel_y=(-0.3, 0.3),  # Reduced from (-0.5, 0.5)
            ang_vel_z=(-1.0, 1.0),  # Reduced from (-1.57, 1.57)
            heading=(-math.pi, math.pi)
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,  # ENABLE noise for Sim2Real robustness
        noise_scales=NoiseScalesCfg(
            lin_vel=0.3,  # Increased from 0.2
            ang_vel=0.3,  # Increased from 0.2
            projected_gravity=0.08,  # Increased from 0.05
            joint_pos=0.02,  # Increased from 0.01
            joint_vel=2.0,  # Increased from 1.5
            height_scan=0.1,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=DexEventCfg(
            # === Physics Material (expanded friction range) ===
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.4, 1.2),  # Expanded from (0.6, 1.0)
                    "dynamic_friction_range": (0.3, 1.0),  # Expanded from (0.4, 0.8)
                    "restitution_range": (0.0, 0.01),  # Expanded from (0.0, 0.005)
                    "num_buckets": 64,
                },
            ),
            # === Base Mass (expanded range) ===
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis", "waist_yaw_link"]),
                    "mass_distribution_params": (-8.0, 8.0),  # Expanded from (-5.0, 5.0)
                    "operation": "add",
                },
            ),
            # === Reset Base ===
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            # === Reset Joints ===
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            # === Push Robot (more aggressive pushing for robustness) ===
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(8.0, 12.0),  # More frequent from (10.0, 15.0)
                params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}},  # Stronger pushes
            ),
            # === PD Gains Randomization (expanded range) ===
            randomize_pd_gains=EventTerm(
                func=mdp.randomize_actuator_gains,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "stiffness_distribution_params": (0.6, 1.4),  # Expanded from (0.75, 1.25)
                    "damping_distribution_params": (0.6, 1.4),  # Expanded from (0.75, 1.25)
                    "operation": "scale",
                    "distribution": "uniform"
                },
            ),
            # === External Force/Torque (more aggressive) ===
            randomize_apply_external_force_torque=EventTerm(
                func=mdp.apply_external_force_torque,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "force_range": (-30.0, 30.0),  # Expanded from (-20.0, 20.0)
                    "torque_range": (-8.0, 8.0),  # Expanded from (-5.0, 5.0)
                },
            ),
            # === COM Randomization (expanded) ===
            randomize_rigid_body_com=EventTerm(
                func=mdp.randomize_rigid_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis", "waist_yaw_link", "waist_pitch_link"]),
                    "com_range": {
                        "x": (-0.08, 0.08),  # Expanded from (-0.05, 0.05)
                        "y": (-0.08, 0.08),
                        "z": (-0.03, 0.03),  # Add Z variation
                    },
                },
            ),
            # === NEW: Joint Parameters Randomization (friction, armature) ===
            randomize_joint_parameters=EventTerm(
                func=mdp.randomize_joint_parameters,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "friction_distribution_params": (0.001, 0.8),  # Joint friction variation
                    "armature_distribution_params": (0.002, 0.08),  # Joint armature variation
                    "operation": "abs",  # Absolute values
                    "distribution": "uniform",
                },
            ),
        ),
        # === ENABLE Action Delay for Sim2Real ===
        action_delay=ActionDelayCfg(
            enable=False,  # WAS False
            params={"max_delay": 3, "min_delay": 0}  # 0-3 steps delay at 50Hz = 0-60ms
        ),
    )
    # === Higher simulation frequency for better accuracy ===
    sim: SimCfg = SimCfg(
        dt=0.002,  # Reduced from 0.005 for 500Hz physics (was 200Hz)
        decimation=10,  # 10 * 0.002 = 0.02s = 50Hz control (matches typical real systems)
        physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    )


@configclass
class DexRunAgentCfg(RslRlOnPolicyRunnerCfg):
    """Optimized training configuration for Sim2Real"""
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Increased from 0.005 for more exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4, 
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,  # Reduced from 0.01 for more conservative updates
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=30.0,  # Reduced from 50.0 - less aggressive symmetry constraint
            data_augmentation_func=mdp.data_augmentation_func_g1,
        ),
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "run_sim2real"
    run_name = "stable_run_v1"
    logger = "tensorboard"
    neptune_project = "run"
    wandb_project = "run"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # AMP parameters - adjusted for stability
    amp_reward_coef = 0.3
    amp_motion_files = ["legged_lab/envs/dex/datasets/motion_amp_expert/new_run.txt"]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.75  # Increased from 0.7 - rely more on task reward
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.08] * 23  # Increased from 0.05 for more exploration
