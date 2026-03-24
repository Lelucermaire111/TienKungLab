# Dex Run 任务速度提升参数调优指南

> 本文档汇总了所有可用于提升 dex_run 任务机器人跑步速度的参数配置。
> 基于代码版本: TienKung-Lab-dev (2025-2026)

---

## 一、速度命令范围（最重要）

**文件**: `legged_lab/envs/dex/run_cfg.py:240`

```python
class DexRunFlatEnvCfg:
    commands: CommandsCfg = CommandsCfg(
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0),      # ← 修改这里: 提升到 (0.0, 2.0) 或更高
            lin_vel_y=(-0.5, 0.5),      # ← 侧向速度，保持或适当提升
            ang_vel_z=(-1.57, 1.57),    # 偏航角速度
            heading=(-math.pi, math.pi)
        ),
    )
```

| 参数 | 当前值 | 建议值 | 说明 |
|------|--------|--------|------|
| `lin_vel_x` | (-0.6, 1.0) | (0.0, 2.5) | 前进速度上限，跑步应 ≥2.0 m/s |
| `lin_vel_y` | (-0.5, 0.5) | (-0.8, 0.8) | 侧向移动范围 |

**注意**: 如果参考 AMP 数据中没有高速运动，单纯提升命令范围可能无效。

---

## 二、奖励函数权重

**文件**: `legged_lab/envs/dex/run_cfg.py:72-182`

### 2.1 速度跟踪奖励（核心）

```python
class DexRewardCfg:
    # 线速度跟踪 - 权重可适当提升
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=4.0,                    # ← 建议: 5.0 ~ 6.0
        params={"std": 0.5}             # ← 建议: 0.8 ~ 1.0（更宽容）
    )

    # 角速度跟踪
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,                    # ← 可保持或略降
        params={"std": 0.5}
    )
```

### 2.2 新跑步专用奖励（推荐启用）

以下奖励函数已在 `legged_lab/mdp/rewards.py` 实现，建议添加到配置中：

```python
class DexRewardCfg:
    # 双脚交替接触奖励（替代旧的 gait_feet_frc_perio）
    feet_contact_alternation = RewTerm(
        func=mdp.feet_contact_alternation,
        weight=2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*")}
    )

    # 悬空时间奖励 - 跑步需要更多空中时间
    feet_air_time_reward = RewTerm(
        func=mdp.feet_air_time_reward,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "target_time": 0.3           # ← 跑步可适当提升到 0.35 ~ 0.4
        }
    )

    # 前向速度额外奖励
    forward_velocity_reward = RewTerm(
        func=mdp.forward_velocity_reward,
        weight=2.0,                    # ← 鼓励突破基础速度
        params={}
    )

    # 抬脚高度奖励 - 防止拖地
    feet_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "min_height": 0.05          # ← 跑步可提升到 0.08 ~ 0.1
        }
    )

    # 步频奖励 - 跑步需要更高步频
    step_frequency_reward = RewTerm(
        func=mdp.step_frequency_reward,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "target_freq": 2.5           # ← 跑步可提升到 3.0 ~ 3.5 Hz
        }
    )

    # 双脚接触力平衡（可选）
    feet_contact_forces_balanced = RewTerm(
        func=mdp.feet_contact_forces_balanced,
        weight=-0.5,                   # 负值表示惩罚
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*")}
    )
```

### 2.3 可放宽的惩罚项

```python
class DexRewardCfg:
    # 身体姿态惩罚 - 跑步时允许更大倾斜
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
        weight=-0.25                   # ← 可放宽到 -0.1
    )

    waist_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link")},
        weight=-1.0                    # ← 可放宽到 -0.5
    )

    # 两脚距离惩罚 - 跑步时步幅更大，距离变化也更大
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-0.5,                   # ← 可降到 -0.3 或更低
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ankle_roll.*"]), "threshold": 0.2}
    )

    # 脚Y方向间距 - 高速时允许更大偏差
    feet_y_distance = RewTerm(
        func=mdp.feet_y_distance,
        weight=-0.5                    # ← 高速跑步时可降低到 -0.2
    )

    # 躯干角速度惩罚 - 跑步需要更多动态平衡
    torso_ang_vel_xy_l2 = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link")},
        weight=-0.5                    # ← 可放宽到 -0.3
    )

    # 终止惩罚 - 如果策略过于保守，可降低
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-20.0                   # ← 可测试 -10.0
    )

    # 髋关节偏航动作惩罚
    hip_yaw_action = RewTerm(
        func=mdp.hip_yaw_action,
        weight=-0.3                    # ← 可降到 -0.1，允许更多转向动作
    )
```

### 2.4 建议禁用的旧版步态奖励

```python
class DexRewardCfg:
    # 以下基于步态时钟的奖励可注释掉，改用上面的新奖励
    # gait_feet_frc_perio = RewTerm(...)      # 可禁用
    # gait_feet_spd_perio = RewTerm(...)      # 可禁用
    # gait_feet_frc_support_perio = RewTerm(...)  # 可禁用
```

---

## 三、步态时钟参数

**文件**: `legged_lab/envs/dex/run_cfg.py:55-60`

```python
@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.6       # ← 可提升到 0.65（更多空中时间）
    gait_air_ratio_r: float = 0.6       # ← 可提升到 0.65
    gait_phase_offset_l: float = 0.6    # 左脚相位偏移
    gait_phase_offset_r: float = 0.1    # 右脚相位偏移（反相）
    gait_cycle: float = 0.64            # ← 可降到 0.5 ~ 0.55（更快步频）
```

| 参数 | 当前值 | 建议值 | 说明 |
|------|--------|--------|------|
| `gait_air_ratio_*` | 0.6 | 0.65 ~ 0.7 | 摆动期占比，跑步需要更多空中时间 |
| `gait_cycle` | 0.64s | 0.5 ~ 0.55s | 步态周期，跑步应比走路快 |

---

## 四、动作空间与控制参数

**文件**: `legged_lab/envs/dex/run_cfg.py:211`

```python
class DexRunFlatEnvCfg:
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,               # ← 可提升到 0.3 ~ 0.4（更大动作幅度）
        terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
        feet_body_names=["ankle_roll.*"],
    )
```

### 4.1 仿真参数

**文件**: `legged_lab/envs/dex/run_cfg.py:340`

```python
class DexRunFlatEnvCfg:
    sim: SimCfg = SimCfg(
        dt=0.005,                        # 物理步长 5ms (200 Hz)
        decimation=4,                    # RL 控制频率 = 200/4 = 50 Hz
        # 对比 Walk 任务: dt=0.0025, decimation=4 → 100 Hz
        # 如需更精细控制，可改为 dt=0.0025
        physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    )
```

**注意**: 当前控制频率 50 Hz 偏低，如 GPU 资源允许，建议改为 100 Hz（与 walk 相同）。

---

## 五、AMP 配置

**文件**: `legged_lab/envs/dex/run_cfg.py:393-399`

```python
class DexRunAgentCfg(RslRlOnPolicyRunnerCfg):
    # AMP 奖励系数
    amp_reward_coef = 0.3               # ← 如参考数据质量高，可提升到 0.4

    # AMP 参考数据文件 - 确保包含高速跑步数据！
    amp_motion_files = ["legged_lab/envs/dex/datasets/motion_amp_expert/new_run.txt"]

    # AMP 与任务奖励混合比例
    amp_task_reward_lerp = 0.7          # 70% 任务奖励 + 30% AMP 奖励
                                        # ← 如需更快速度，可降到 0.6（更多任务驱动）

    amp_num_preload_transitions = 200000
    amp_discr_hidden_dims = [1024, 512, 256]
```

**关键检查**: 确保 `new_run.txt` 包含高速运动数据（速度 ≥2.0 m/s）。

---

## 六、训练算法参数

**文件**: `legged_lab/envs/dex/run_cfg.py:344-400`

```python
class DexRunAgentCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24              # 每轮 rollout 步数
    max_iterations = 50000              # 总训练轮数

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,             # 初始探索噪声
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        clip_param=0.2,                 # PPO clip 参数
        entropy_coef=0.005,             # ← 如需更多探索，可提升到 0.01
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        desired_kl=0.01,                # 目标 KL 散度
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_mirror_loss=True,
            mirror_loss_coeff=50.0,     # ← 可测试降到 30.0（减少对称约束）
            data_augmentation_func=mdp.data_augmentation_func_g1,
        ),
    )

    # 动作最小标准差
    min_normalized_std = [0.05] * 23    # 防止过早收敛
```

---

## 七、域随机化参数

**文件**: `legged_lab/envs/dex/run_cfg.py:254-339`

如需 Sim2Real 迁移，保持现有随机化；如仅在仿真中追求极限速度，可适当减小随机范围。

```python
class DexEventCfg(EventCfg):
    # 推力扰动 - 增加抗干扰能力
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}}
                                        # ← 可增大到 ±1.5 m/s 增强鲁棒性
    )
```

---

## 八、课程学习（推荐添加）

当前配置未显式启用速度课程学习。建议添加 `GridAdaptiveCurriculum` 实现速度渐进：

```python
# 在 CommandsCfg 中确保启用了课程学习
class DexRunFlatEnvCfg:
    commands: CommandsCfg = CommandsCfg(
        # 保持其他参数
        # GridAdaptiveCurriculum 会根据成功率自动调整各速度区域的采样概率
    )
```

---

## 九、参数调优优先级总结

| 优先级 | 参数 | 预期速度提升 | 风险 |
|--------|------|-------------|------|
| 🔴 P0 | `lin_vel_x` 上限 | +50~100% | 可能不稳定 |
| 🔴 P0 | 检查 AMP 数据速度 | +30~50% | 如数据慢则无效 |
| 🔴 P0 | 启用新跑步奖励 | +20~30% | 无 |
| 🟡 P1 | `action_scale` | +10~20% | 动作过大可能不稳定 |
| 🟡 P1 | `gait_cycle` 降低 | +10~15% | 需要更高控制频率 |
| 🟡 P1 | 放宽姿态惩罚 | +10~15% | 可能姿态不自然 |
| 🟢 P2 | `amp_task_reward_lerp` | +5~10% | 步态可能不自然 |
| 🟢 P2 | `track_lin_vel_xy_exp` std | +5~10% | 无 |
| ⚪ P3 | 控制频率提升 | +5~10% | 计算量增加 |

---

## 十、建议的实验步骤

1. **基线测试**: 记录当前配置下的最大稳定速度
2. **解锁速度**: 将 `lin_vel_x` 上限提升到 2.0，观察是否能达到
3. **奖励调优**: 启用新的跑步奖励函数，禁用旧步态奖励
4. **动作幅度**: 逐步提升 `action_scale` 到 0.3
5. **步频优化**: 降低 `gait_cycle` 到 0.55
6. **AMP 调整**: 如需要更高速度，降低 `amp_task_reward_lerp` 到 0.6

---

## 附录：快速修改模板

将以下内容添加到 `run_cfg.py` 的 `DexRewardCfg` 中：

```python
# === 跑步速度提升优化（添加到 DexRewardCfg）===

# 1. 提升速度跟踪权重和宽容度
# track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=5.0, params={"std": 0.8})

# 2. 启用新跑步奖励
# feet_contact_alternation = RewTerm(func=mdp.feet_contact_alternation, weight=2.0, params={...})
# feet_air_time_reward = RewTerm(func=mdp.feet_air_time_reward, weight=2.0, params={...})
# forward_velocity_reward = RewTerm(func=mdp.forward_velocity_reward, weight=2.0)
# feet_clearance = RewTerm(func=mdp.feet_clearance, weight=1.0, params={...})
# step_frequency_reward = RewTerm(func=mdp.step_frequency_reward, weight=1.0, params={...})

# 3. 放宽惩罚项
# body_orientation_l2 = RewTerm(..., weight=-0.1)
# waist_orientation_l2 = RewTerm(..., weight=-0.5)
# feet_too_near = RewTerm(..., weight=-0.3)
```

---

*文档生成时间: 2026-03-24*
*适用代码版本: TienKung-Lab-dev main branch*
