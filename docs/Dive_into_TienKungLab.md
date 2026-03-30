# TienKung-Lab Locomotion 训练算法深度学习文档

---

## 目录

1. [整体架构与数据流](#1-整体架构与数据流)
2. [核心算法：AMP-PPO](#2-核心算法amp-ppo)
   - [PPO 基础原理](#21-ppo-基础原理)
   - [AMP（对抗运动先验）](#22-amp对抗运动先验)
   - [AMP + PPO 联合训练](#23-amp--ppo-联合训练)
3. [神经网络结构](#3-神经网络结构)
   - [Actor-Critic 网络](#31-actor-critic-网络)
   - [判别器网络](#32-判别器网络)
4. [观测空间（Observation Space）](#4-观测空间observation-space)
   - [Policy Observation（Actor 输入）](#41-policy-observationactor-输入)
   - [Critic Observation（Critic 特权输入）](#42-critic-observationcritic-特权输入)
   - [AMP Observation（判别器输入）](#43-amp-observationamp-输入)
   - [历史帧缓存](#44-历史帧缓存)
5. [动作空间（Action Space）](#5-动作空间action-space)
6. [奖励函数详解](#6-奖励函数详解)
   - [运动跟踪奖励](#61-运动跟踪奖励)
   - [稳定性惩罚](#62-稳定性惩罚)
   - [能量与机械惩罚](#63-能量与机械惩罚)
   - [接触与步态奖励](#64-接触与步态奖励)
   - [站立奖励](#65-站立奖励)
   - [AMP 风格奖励](#66-amp-风格奖励)
   - [终止惩罚](#67-终止惩罚)
   - [奖励权重汇总表（Walk 任务）](#68-奖励权重汇总表walk-任务)
7. [步态时钟机制（Gait Clock）](#7-步态时钟机制gait-clock)
8. [域随机化（Domain Randomization）](#8-域随机化domain-randomization)
9. [课程学习（Curriculum Learning）](#9-课程学习curriculum-learning)
10. [对称性损失（Symmetry Loss）](#10-对称性损失symmetry-loss)
11. [AMP 运动数据格式与加载](#11-amp-运动数据格式与加载)
12. [训练超参数详解](#12-训练超参数详解)
13. [Sim2Sim 迁移](#13-sim2sim-迁移)
14. [关键设计决策与 Trade-offs](#14-关键设计决策与-trade-offs)

---

## 1. 整体架构与数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                       训练主循环 (AmpOnPolicyRunner)              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Isaac Sim 仿真环境                      │   │
│  │   4096 个并行环境  ×  step_dt=0.01s (4 × 0.0025s)        │   │
│  │   每步输出: obs, critic_obs, amp_obs, reward, done        │   │
│  └──────────────────────────────────────────────────────────┘   │
│            │                                                     │
│  ┌─────────▼──────────────────────────────────────────────┐    │
│  │              RolloutStorage (24步/env 缓存)              │    │
│  │   存储: obs, critic_obs, actions, values, rewards,      │    │
│  │         log_probs, dones, advantages                    │    │
│  └─────────────────────────────────────────────────────────┘   │
│            │                                                     │
│  ┌─────────▼──────────────────────────────────────────────┐    │
│  │               AMPPPO.update() 每轮收集后执行             │    │
│  │   ┌──────────┐  ┌──────────────┐  ┌────────────────┐   │    │
│  │   │ PPO Loss │  │ AMP Disc Loss│  │ Symmetry Loss  │   │    │
│  │   │ surrogate│  │ expert_loss  │  │ mirror MSE     │   │    │
│  │   │ value    │  │ policy_loss  │  │                │   │    │
│  │   │ entropy  │  │ grad_pen     │  │                │   │    │
│  │   └──────────┘  └──────────────┘  └────────────────┘   │    │
│  │         5 epochs × 4 mini-batches per update            │    │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**时间步关系：**

| 参数 | 值 | 含义 |
|---|---|---|
| `physics_dt` | 0.0025 s | PhysX 物理仿真步长 |
| `decimation` | 4 | 每次 RL step 执行 4 次物理步 |
| `step_dt` | 0.01 s | RL 决策频率 = 100 Hz |
| `num_steps_per_env` | 24 | 每次收集后更新前的 rollout 步数 |
| `num_envs` | 4096 | 并行环境数 |
| 每次更新样本量 | 4096 × 24 = **98304** | 一次 update 的总样本数 |

---

## 2. 核心算法：AMP-PPO

### 2.1 PPO 基础原理

PPO（Proximal Policy Optimization）的目标是在更新策略时，不让新策略与旧策略偏离过大。核心公式：

**Surrogate Loss（Actor）：**

```
ratio_t = exp(log π_θ(a_t|s_t) - log π_θ_old(a_t|s_t))

L_CLIP = E[ min(ratio_t · A_t,  clip(ratio_t, 1-ε, 1+ε) · A_t) ]
```

- `ε = 0.2`（`clip_param`）
- `A_t` 是 GAE 优势估计（见下文）

**Value Function Loss（Critic）：**

```python
# 使用 clipped value loss 防止 value 更新过猛
value_clipped = target_values + (value - target_values).clamp(-clip_param, clip_param)
value_losses = (value - returns).pow(2)
value_losses_clipped = (value_clipped - returns).pow(2)
value_loss = max(value_losses, value_losses_clipped).mean()
```

**GAE（Generalized Advantage Estimation）：**

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

- `γ = 0.99`（折扣因子）
- `λ = 0.95`（GAE λ）

**Timeout Bootstrapping（关键细节）：**

```python
# 如果 episode 因超时结束（不是真正失败），对 reward 做 bootstrap 补偿
rewards += gamma * squeeze(values * time_outs.unsqueeze(1))
```

这防止了因超时终止导致的价值低估。

**自适应学习率：**

```python
# 根据 KL 散度自动调整 lr
if kl_mean > desired_kl * 2.0:
    learning_rate = max(1e-5, learning_rate / 1.5)
elif kl_mean < desired_kl / 2.0:
    learning_rate = min(1e-2, learning_rate * 1.5)
```

- `desired_kl = 0.01`，`schedule = "adaptive"`
- 这是本项目的关键超参数，控制更新步长

---

### 2.2 AMP（对抗运动先验）

AMP（Adversarial Motion Priors，[Peng et al. 2021](https://arxiv.org/abs/2104.02180)）的核心思想：**训练一个判别器来区分 agent 的运动和专家参考动作，将判别器的输出作为 reward 反馈给 agent**。这样 agent 会主动模仿参考动作的运动风格，而不需要手工设计每个细节的奖励。

**判别器目标（LSGAN 损失）：**

```python
# 专家数据 → 判别器输出应趋向 +1
expert_loss = MSE(D(s_expert, s'_expert), +1)

# Agent 数据 → 判别器输出应趋向 -1
policy_loss = MSE(D(s_policy, s'_policy), -1)

amp_loss = 0.5 * (expert_loss + policy_loss)
```

**梯度惩罚（Gradient Penalty）：**

```python
# 在专家数据上计算梯度惩罚，稳定判别器训练
grad = autograd.grad(outputs=disc, inputs=expert_data, ...)[0]
grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
# lambda_ = 10
```

这是 WGAN-GP 风格的梯度惩罚，让判别器梯度范数趋向 0（而不是 1）。

**AMP 奖励计算（推理时）：**

```python
d = Discriminator(concat[s_t, s_{t+1}])  # 判别器输出 scalar
reward = amp_reward_coef * clamp(1 - (1/4) * (d - 1)^2,  min=0)
```

这将判别器输出 `d ∈ [-1, +1]` 映射为奖励：
- `d = +1`（像专家）→ reward = `amp_reward_coef = 0.3`（最大）
- `d = -1`（不像专家）→ reward = 0

**任务奖励与 AMP 奖励的混合：**

```python
# task_reward_lerp = 0.7
reward = (1.0 - task_reward_lerp) * amp_reward + task_reward_lerp * task_reward
       = 0.3 * amp_r + 0.7 * task_r
```

这意味着 70% 来自手工设计的任务奖励，30% 来自运动风格匹配。

---

### 2.3 AMP + PPO 联合训练

在每次 `update()` 中，三个生成器并行采样，逐 mini-batch 同步更新：

```python
for sample, sample_amp_policy, sample_amp_expert in zip(
    ppo_generator,         # 来自 RolloutStorage（agent 轨迹）
    amp_policy_generator,  # 来自 ReplayBuffer（agent AMP obs 历史）
    amp_expert_generator   # 来自 AMPLoader（专家数据）
):
    # 1. 计算 PPO 损失
    loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy

    # 2. 加入对称性损失
    loss += mirror_loss_coeff * symmetry_loss   # weight=100

    # 3. 计算判别器损失并加入总损失
    amp_loss = 0.5 * (expert_loss + policy_loss)
    loss += amploss_coef * amp_loss + amploss_coef * grad_pen_loss
    # amploss_coef = 1.0

    # 4. 反向传播，一次 backward 更新 policy + discriminator
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(policy.parameters(), max_grad_norm=1.0)
    optimizer.step()

    # 5. 更新 AMP normalizer（在线 running mean/std）
    amp_normalizer.update(policy_state)
    amp_normalizer.update(expert_state)
```

**重要细节：policy 和 discriminator 共享同一个 Adam 优化器**（但参数组不同，判别器有额外的 weight decay）：

```python
params = [
    {"params": policy.parameters(),              "name": "policy"},
    {"params": discriminator.trunk.parameters(), "weight_decay": 1e-3, "name": "amp_trunk"},
    {"params": discriminator.amp_linear.parameters(), "weight_decay": 1e-1, "name": "amp_head"},
]
optimizer = Adam(params, lr=1e-3)
```

**AMP ReplayBuffer（agent 侧）：**

- 容量：100,000 个转换
- 存储 agent 自身的 `(s_t, s_{t+1})` AMP 观测对
- 每步 `process_env_step` 时插入，训练时随机采样
- 防止判别器过拟合到最新轨迹

---

## 3. 神经网络结构

### 3.1 Actor-Critic 网络

```
Actor (Policy):
  输入: obs_history (展平后) → shape = [num_envs, actor_obs_dim × history_len]

  MLP: [512, 256, 128]
  激活: ELU
  输出: action_mean (shape: [num_envs, 20])  ← 20 个关节

  动作分布: 各向同性高斯, std = exp(log_std)
  初始 std: 1.0 (init_noise_std)
  最小 std: [0.05] × 20 (min_normalized_std, 在 AMPPPO 中约束)

Critic (Value Network):
  输入: critic_obs_history (包含特权信息)

  MLP: [512, 256, 128]  ← 与 actor 独立的参数
  激活: ELU
  输出: value (scalar per env)
```

- `noise_std_type = "scalar"` 表示所有关节共享同一个 learnable log_std
- `activation = "elu"` 而非 ReLU，对负值有梯度，有助于训练稳定性

### 3.2 判别器网络

```
Discriminator:
  输入: concat[s_t, s_{t+1}]  shape = [batch, 2 × amp_obs_dim]
       amp_obs_dim = 20(关节位置) + 20(关节速度) + 12(末端位置) = 52
       输入维度 = 52 × 2 = 104

  trunk (MLP): [1024, 512, 256]  激活: ReLU
  amp_linear: Linear(256, 1)     激活: 无（输出原始 logit）

  参数总量约: 1024×104 + 512×1024 + 256×512 + 1×256 ≈ 1.1M
```

---

## 4. 观测空间（Observation Space）

### 4.1 Policy Observation（Actor 输入）

每个时间步的 single-frame obs 由以下向量拼接：

| 分量 | 维度 | 描述 | 噪声 σ |
|---|---|---|---|
| `commands` | 3 | (vx, vy, ωz) 命令速度（或 heading） | — |
| `projected_gravity` | 3 | 重力向量在机体坐标系投影 | 0.05 |
| `root_lin_vel_b` | 3 | 机体坐标系线速度 | 0.2 |
| `root_ang_vel_b` | 3 | 机体坐标系角速度 | 0.2 |
| `joint_pos` | 20 | 关节位置 − 默认位置 | 0.01 |
| `joint_vel` | 20 | 关节速度 | 1.5 |
| `last_action` | 20 | 上一步动作 | — |
| **合计** | **72** | | |

注意 `actor_obs_history_length = 10`，所以 actor 实际输入是 **72 × 10 = 720 维**的历史展平向量。

### 4.2 Critic Observation（Critic 特权输入）

Critic 除了 policy obs 外，还包含 agent 在实际部署时看不见的信息：

| 额外分量 | 维度 | 描述 |
|---|---|---|
| `root_lin_vel_w` | 3 | 世界坐标系线速度（更精确） |
| `contact_forces` | 2×3 | 双脚接触力向量 |
| height scan（可选） | N | 地形高度扫描 |
| gait phases | 4 | 步态相位 (当前相位 + 相位比) |

同样有 `critic_obs_history_length = 10` 的历史。

### 4.3 AMP Observation（判别器输入）

AMP obs 是用来让判别器判断运动风格的特征，与策略训练的 obs 解耦：

```python
# 来自 AMPLoader 的数据结构
JOINT_POS_SIZE = 20    # 关节位置
JOINT_VEL_SIZE = 20    # 关节速度
END_EFFECTOR_POS_SIZE = 12  # 末端执行器位置（4个末端 × 3D）

# 单帧 AMP obs = 20 + 20 + 12 = 52 维
# 判别器输入 = concat(s_t, s_{t+1}) = 52 × 2 = 104 维
```

末端执行器包括：双手、双脚，各 3D 位置（共 12 维）。

### 4.4 历史帧缓存

```python
# 使用 CircularBuffer 维护历史
self.actor_history_buf = CircularBuffer(
    max_len=actor_obs_history_length,  # 10
    batch_size=num_envs,
    device=device
)
# 每步 push 当前 obs，取全部历史展平
actor_obs = self.actor_history_buf.buffer  # [num_envs, 10, 72]
actor_obs = actor_obs.reshape(num_envs, -1)  # [num_envs, 720]
```

历史帧提供**时序信息**，让 policy 能感知速度变化趋势、接触序列等动态特征，而不仅仅依赖当前帧。

---

## 5. 动作空间（Action Space）

```python
# 输出: 20 维关节目标位置偏移量
action = actor_net(obs)          # 均值，范围: [-∞, +∞]
action_scaled = action * action_scale  # action_scale = 0.25

# 实际关节目标位置
joint_target = default_joint_pos + action_scaled

# 通过 PD 控制器转换为力矩（由物理引擎执行）
torque = Kp * (joint_target - joint_pos) - Kd * joint_vel
```

`action_scale = 0.25` 意味着网络输出 1 对应 0.25 rad 的关节偏移，**限制了策略的探索范围**，提高训练稳定性。

动作经过 `DelayBuffer` 可模拟控制延迟（`enable=False` 默认关闭，用于 Sim2Real 时开启）。

---

## 6. 奖励函数详解

### 6.1 运动跟踪奖励

**线速度跟踪（最重要的任务奖励）：**

```python
def track_lin_vel_xy_yaw_frame_exp(env, std=0.5):
    # 在偏航坐标系（yaw frame）中计算速度误差
    vel_yaw = quat_apply_inverse(yaw_quat(root_quat_w), root_lin_vel_w)
    error = sum((command[:, :2] - vel_yaw[:, :2])^2, dim=1)
    return exp(-error / std^2)   # weight = 4.0
```

使用偏航坐标系而非机体坐标系，让速度跟踪与机器人侧倾/俯仰解耦。std=0.5 对应：误差 0.5 m/s 时奖励衰减到 ~0.61，误差 1.0 m/s 衰减到 ~0.14。

**角速度跟踪：**

```python
def track_ang_vel_z_world_exp(env, std=0.5):
    error = (command[:, 2] - root_ang_vel_w[:, 2])^2
    return exp(-error / std^2)   # weight = 2.0
```

### 6.2 稳定性惩罚

| 函数 | 计算方式 | 权重 | 作用 |
|---|---|---|---|
| `lin_vel_z_l2` | `root_lin_vel_b[:, 2]^2`，clamp max=25 | −0.5 | 惩罚竖直方向速度（防止弹跳） |
| `ang_vel_xy_l2` | `sum(root_ang_vel_b[:, :2]^2)`，clamp max=400 | −0.05 | 惩罚翻滚/俯仰角速度 |
| `flat_orientation_l2` | `sum(projected_gravity_b[:, :2]^2)` | −0.5 | 惩罚机体倾斜 |
| `body_orientation_l2` | 骨盆姿态的重力向量投影误差 | −1.0 | 惩罚骨盆倾斜（更严格） |
| `torso_ang_vel_xy_l2` | 骨盆的滚转/俯仰角速度（在机体系下） | −0.05 | 减少躯干摇摆 |

### 6.3 能量与机械惩罚

| 函数 | 计算方式 | 权重 | 作用 |
|---|---|---|---|
| `energy` | `norm(|torque × joint_vel|)` | −1e-3 | 惩罚机械功率（效率） |
| `joint_acc_l2` | `sum(joint_acc^2)`，clamp max=1e4 | −2.5e-7 | 惩罚关节加速度（平滑动作） |
| `action_rate_l2` | `sum((a_t - a_{t-1})^2)`，clamp max=4 | −0.01 | 惩罚动作变化率（平滑控制） |
| `ankle_torque` | `sum(ankle_torques^2)` | −0.0005 | 减少踝关节负载 |
| `ankle_action` | `sum(|ankle_actions|)` | −0.001 | 限制踝关节动作幅度 |

### 6.4 接触与步态奖励

**步态周期奖励（核心，基于相位时钟）：**

```python
def gait_clock(phase, air_ratio, delta_t):
    """将相位 [0,1] 映射为摆动期/支撑期指示函数"""
    # I_frc: 摆动期力矩（应为0，即脚在空中，不接触地面）
    # I_spd: 支撑期速度（应为0，即脚接触地面，不滑动）
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    # ... 平滑插值过渡区 ...
    I_frc = swing_flag + transition_terms
    I_spd = 1 - I_frc
    return I_frc, I_spd

def gait_feet_frc_perio(env, delta_t=0.02):
    """摆动期脚接触力应为0"""
    left_swing_mask = gait_clock(gait_phase[:, 0], phase_ratio[:, 0], delta_t)[0]
    left_score = left_swing_mask * exp(-100 * avg_feet_force^2)
    # weight = 2.0
```

步态参数（Walk 任务）：

```python
@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.38   # 左脚摆动相占比 38%
    gait_air_ratio_r: float = 0.38   # 右脚摆动相占比 38%
    gait_phase_offset_l: float = 0.38  # 左脚相位偏移
    gait_phase_offset_r: float = 0.88  # 右脚相位偏移（与左脚相差 0.5，反相）
    gait_cycle: float = 0.85           # 步态周期 0.85s
```

三个步态相关奖励：
- `gait_feet_frc_perio`（weight=2.0）: 摆动期脚应零接触力
- `gait_feet_spd_perio`（weight=2.0）: 支撑期脚应零速度（不滑动）
- `gait_feet_frc_support_perio`（weight=1.5）: 支撑期脚应有接触力

**足部距离限制：**

```python
def feet_too_near_humanoid(env, threshold=0.2):
    """两脚距离小于 0.2m 时惩罚，防止 cross-legged 步态"""
    distance = norm(feet_pos[:, 0] - feet_pos[:, 1])
    return clamp(threshold - distance, min=0)  # weight = -2.0

def feet_y_distance(env):
    """惩罚与目标 y 间距 0.299m 的偏差（仅在无横向速度命令时）"""
    y_distance_error = |left_foot_y_body - right_foot_y_body - 0.299|
    return y_distance_error * (|vy_cmd| < 0.1)  # weight = -4.0
```

**接触相关：**

```python
def undesired_contacts(env, threshold=1.0, ...):
    """膝关节/肩膀/手肘/骨盆接触地面时惩罚"""   # weight = -1.0

def feet_slide(env, ...):
    """足部接触地面时的滑动速度惩罚"""          # weight = -1.0

def feet_stumble(env, ...):
    """横向力 > 5 × 法向力时惩罚（踩地不稳）"""  # weight = -2.0

def body_force(env, threshold=1000, max_reward=400):
    """足部接触力超过 1000N 时惩罚（冲击力过大）""" # weight = -3e-3
```

### 6.5 站立奖励

```python
def stand_still_exp(env, zero_threshold=0.2):
    """命令速度 ≈ 0 时，奖励关节位置接近默认姿态"""
    zero_flag = (|cmd_xy| + |cmd_ωz|) <= 0.2
    angle_error = sum(|joint_pos - default_pos|, dim=1)
    return exp(-1.0 * angle_error) * zero_flag   # weight = 7.0  (最大奖励项之一)

def stand_still_double_support(env, ...):
    """站立时奖励双脚同时着地"""              # weight = +0.5

def stand_still(env, ...):
    """站立时惩罚关节偏离默认（L1）"""        # weight = -0.5
```

### 6.6 AMP 风格奖励

这不是手工设计的奖励函数，而是由 `discriminator.predict_amp_reward()` 在 `tienkung_env.py` 的 `get_observations()` 中计算，并混入总奖励。奖励计算已在 [2.2节](#22-amp对抗运动先验) 描述。

### 6.7 终止惩罚

```python
def is_terminated(env):
    """非超时终止（即真实失败）时给予大惩罚"""
    return reset_buf & ~time_out_buf   # weight = -50.0

def alive_reward(env):
    """每存活一步给 +0.5"""
    return (~reset_buf).float()        # weight = +0.5
```

### 6.8 奖励权重汇总表（Walk 任务）

| 奖励项 | 权重 | 正/负 | 物理含义 |
|---|---|---|---|
| `track_lin_vel_xy` | 4.0 | + | 线速度跟踪（主要任务） |
| `stand_still_exp` | 7.0 | + | 站立时保持姿态（重要！） |
| `track_ang_vel_z` | 2.0 | + | 角速度跟踪 |
| `gait_feet_frc_perio` | 2.0 | + | 摆动期零接触力 |
| `gait_feet_spd_perio` | 2.0 | + | 支撑期零滑动 |
| `gait_feet_frc_support_perio` | 1.5 | + | 支撑期有接触力 |
| `alive_reward` | 0.5 | + | 存活激励 |
| `stand_still_double_support` | 0.5 | + | 站立双脚接触 |
| `termination_penalty` | −50.0 | − | 跌倒大惩罚 |
| `body_orientation_l2` | −1.0 | − | 躯干姿态 |
| `feet_too_near` | −2.0 | − | 两脚过近 |
| `feet_stumble` | −2.0 | − | 脚踩不稳 |
| `dof_pos_limits` | −2.0 | − | 关节限位 |
| `hip_roll_action` | −2.0 | − | 横摆髋关节 |
| `feet_y_distance` | −4.0 | − | 脚间距偏差 |
| `feet_slide` | −1.0 | − | 脚滑动 |
| `flat_orientation_l2` | −0.5 | − | 机体倾斜 |
| `lin_vel_z_l2` | −0.5 | − | 竖直速度 |
| `joint_deviation_hip` | −0.15 | − | 髋/肩关节偏离 |
| `action_rate_l2` | −0.01 | − | 动作平滑性 |
| `energy` | −1e-3 | − | 能量消耗 |

---

## 7. 步态时钟机制（Gait Clock）

步态时钟是一个显式的相位调度器，强制 agent 遵循预设的步态节律。

```
相位 φ ∈ [0, 1)，以 gait_cycle = 0.85s 为周期
左脚: φ_L(t) = (t / gait_cycle + phase_offset_L) mod 1
右脚: φ_R(t) = (t / gait_cycle + phase_offset_R) mod 1

phase_offset_L = 0.38
phase_offset_R = 0.88  →  两脚相差 0.5，完全反相（标准行走步态）

air_ratio = 0.38 意味着:
  摆动期(Swing): φ ∈ [0, 0.38)  → 38% 时间脚在空中
  支撑期(Stance): φ ∈ [0.38, 1) → 62% 时间脚在地面
```

`gait_clock()` 函数返回两个掩码：
- **I_frc**（摆动期掩码）：= 1 表示当前时刻应该摆动（脚应离地，接触力=0）
- **I_spd**（支撑期掩码）：= 1 表示当前时刻应该支撑（脚应接地，速度=0）

在过渡区 `delta_t = 0.02` 内做线性插值，避免奖励信号突变。

**注意：** 代码中步态奖励函数已被标注为 `DEPRECATED`，新的无步态时钟奖励函数（`feet_contact_alternation`、`feet_air_time_reward` 等）已被添加，但 Walk 配置的 `LiteRewardCfg` 仍在使用旧版步态时钟奖励。

---

## 8. 域随机化（Domain Randomization）

域随机化是 Sim2Real 迁移的核心技术，让策略对各种物理参数不确定性鲁棒。

| 随机化类型 | 触发时机 | 随机化范围 | 作用 |
|---|---|---|---|
| **摩擦系数** | `startup`（一次性） | static: [0.6, 1.0]，dynamic: [0.4, 0.8] | 地面摩擦不确定性 |
| **质量添加（骨盆）** | `startup` | [-5.0, +5.0] kg | 携带负载/质量误差 |
| **关节 PD 增益** | `reset`（每次重置） | scale × [0.75, 1.25] | 执行器特性差异 |
| **外力/力矩** | `reset` | 力: ±20N，力矩: ±5Nm | 外部扰动 |
| **关节参数** | `startup` | 摩擦: [0.001, 0.6]，armature: [0.002, 0.060] | 关节内摩擦/惯量 |
| **质心偏移（骨盆）** | `startup` | x/y: ±0.05m | 质心不确定性 |
| **推力扰动** | `interval`（10-15s随机） | 速度: x/y ±1.0 m/s | 模拟推撞 |
| **初始姿态** | `reset` | 位置 ±0.5m，速度 ±0.5 m/s | 多样初始状态 |
| **初始关节** | `reset` | scale: [0.5, 1.5] × 默认 | 多样起始姿态 |

**分级随机化策略：**
- `startup` 事件：场景创建时执行一次（如摩擦系数、质心偏移），每次训练固定
- `reset` 事件：每次 episode 重置时执行（如 PD 增益），增加跨 episode 多样性
- `interval` 事件：按时间间隔随机触发（如推力），模拟运动中的突发扰动

---

## 9. 课程学习（Curriculum Learning）

### 地形课程（TerrainCurriculum）

```python
terrain_type = "generator"
terrain_generator = ROUGH_TERRAINS_CFG  # 随机粗糙地形
max_init_terrain_level = 5              # 初始最大难度等级

# 机器人从低难度地形开始，随着训练进展，成功率高时提升地形难度
```

地形包括：`gravel_terrains`（砾石地形）和 `rough_terrains`（粗糙地形）。

### 速度命令课程（GridAdaptiveCurriculum）

```python
# Walk 任务速度范围
lin_vel_x: (-0.6, 1.2) m/s   # 前后速度
lin_vel_y: (-0.5, 0.5) m/s   # 横向速度
ang_vel_z: (-1.57, 1.57) rad/s  # 偏航速度

rel_standing_envs = 0.4  # 40% 的环境始终使用零命令（站立）
rel_heading_envs = 1.0   # 100% 的环境使用 heading 控制模式
```

`GridAdaptiveCurriculum` 将速度空间离散化为格，根据机器人在各格上的成功率（速度跟踪误差），自适应地增大或减小该格的采样概率。

---

## 10. 对称性损失（Symmetry Loss）

由于双足人形机器人天然左右对称，可以利用这一先验知识加速训练：

```python
symmetry_cfg = RslRlSymmetryCfg(
    use_data_augmentation=False,  # 不用数据增广（因为可能与 AMP 冲突）
    use_mirror_loss=True,         # 使用镜像损失
    mirror_loss_coeff=100,        # 权重 100（非常大！）
    data_augmentation_func=mdp.data_augmentation_func_g1,
)
```

**镜像损失计算：**

```python
# 1. 对 obs 做镜像翻转（左右对调）→ mirrored_obs
mirrored_obs, _ = augmentation_func(obs=obs_batch, actions=None)

# 2. 用原始 obs 预测均值动作
action_mean_orig = policy.act_inference(obs_batch)

# 3. 对原始动作做镜像变换 → expected_symmetric_action
_, action_mean_symm = augmentation_func(obs=None, actions=action_mean_orig)

# 4. 要求对镜像 obs 的预测 = 镜像后的原始动作
symmetry_loss = MSE(policy(mirrored_obs), action_mean_symm)
loss += 100 * symmetry_loss
```

权重 100 非常大，意味着对称性约束是训练中的**强约束**，会显著影响策略的左右对称性。

`data_augmentation_func_g1` 定义了 TienKung 机器人的关节镜像映射（左右关节交换，并翻转符号）。

---

## 11. AMP 运动数据格式与加载

### 数据文件格式

```json
{
  "MotionWeight": 1.0,
  "FrameDuration": 0.0333,
  "Frames": [
    [joint_pos×20, joint_vel×20, end_effector_pos×12],
    ...
  ]
}
```

- `FrameDuration = 0.0333s` ≈ 30 FPS 捕获
- `MotionWeight` 用于多轨迹加权采样
- Walk 任务：`walk.txt`（约 4 秒）
- Run 任务：`run.txt`

### 帧插值

```python
def slerp(frame1, frame2, blend):
    return (1 - blend) * frame1 + blend * frame2
```

由于参考数据帧率（30 FPS）与 RL 决策频率（100 Hz）不同，在采样时使用线性插值在帧之间插值。

### 预加载（Preload Transitions）

```python
# 预加载 200,000 个 (s_t, s_{t+1}) 对到 GPU 内存
# 训练时直接随机索引，避免重复计算插值
amp_num_preload_transitions = 200000
preloaded_s = get_full_frame_at_time_batch(traj_idxs, times)
preloaded_s_next = get_full_frame_at_time_batch(traj_idxs, times + time_between_frames)
```

`time_between_frames = physics_dt = 0.0025s`（与物理仿真步长对齐）。

### AMP Normalizer

```python
# 在线 running mean/std，用于归一化 AMP obs
amp_normalizer.update(policy_state.cpu().numpy())
amp_normalizer.update(expert_state.cpu().numpy())
```

对 policy 和 expert 的 AMP obs 共同统计均值方差，确保判别器输入量纲一致。

---

## 12. 训练超参数详解

```python
# === 训练规模 ===
num_envs = 4096            # 并行环境数（GPU 并行）
num_steps_per_env = 24     # 每次 rollout 步数
max_iterations = 50000     # 最大训练 iteration
save_interval = 100        # 每 100 iteration 保存检查点

# === PPO 核心超参 ===
num_learning_epochs = 5    # 每次 rollout 数据复用 5 epoch
num_mini_batches = 4       # 每 epoch 分 4 个 mini-batch
                           # mini-batch size = 4096×24/4 = 24576
clip_param = 0.2           # PPO clip 范围
gamma = 0.99               # 折扣因子（较高，重视长期奖励）
lam = 0.95                 # GAE lambda
value_loss_coef = 1.0      # value loss 权重
entropy_coef = 0.005       # 熵正则化（鼓励探索）
max_grad_norm = 1.0        # 梯度裁剪

# === 学习率 ===
learning_rate = 1e-3       # 初始学习率
schedule = "adaptive"      # 自适应调整
desired_kl = 0.01          # 目标 KL 散度
# KL > 0.02: lr /= 1.5
# KL < 0.005: lr *= 1.5
# lr ∈ [1e-5, 1e-2]

# === AMP 超参 ===
amp_reward_coef = 0.3          # AMP 奖励缩放系数
amp_task_reward_lerp = 0.7     # 任务奖励占比 70%
amp_discr_hidden_dims = [1024, 512, 256]  # 判别器隐层维度
amp_replay_buffer_size = 100000  # agent AMP obs 回放缓冲大小
amp_num_preload_transitions = 200000  # 专家数据预加载量

# === 策略约束 ===
min_normalized_std = [0.05] * 20  # 最小动作标准差（防止过早收敛到确定策略）
action_scale = 0.25               # 动作缩放（限制关节偏移幅度）

# === 对称性 ===
mirror_loss_coeff = 100    # 镜像损失权重（强约束）

# === 仿真 ===
sim.dt = 0.0025            # 物理步长 400 Hz
sim.decimation = 4         # RL 决策 100 Hz
max_episode_length_s = 10.0  # 最长 episode 10 秒
```

**Throughput 估算：**

```
每秒 RL 步数 ≈ num_envs × (1/step_dt) = 4096 × 100 = 409,600 steps/s（理论峰值）
每 iteration 样本 = 4096 × 24 = 98,304
更新次数 = 5 epochs × 4 batches = 20 次梯度更新/iteration
50,000 iterations = 约 4.9 × 10^9 样本总量
```

---

## 13. Sim2Sim 迁移

训练后的策略以 TorchScript 格式导出，在 MuJoCo 中验证：

```python
# sim2sim.py 核心流程
class MujocoRunner:
    def __init__(self, cfg: SimToSimCfg):
        self.model = mujoco.MjModel.from_xml_path(cfg.mj_model_path)
        self.policy = torch.jit.load(cfg.policy_path)  # 加载 TorchScript

    def run(self):
        while running:
            # 1. 构造与训练相同的 obs（历史帧、归一化）
            obs = self._compute_obs()

            # 2. 策略推理（无梯度）
            with torch.no_grad():
                action = policy(obs)

            # 3. 换算为关节目标位置
            joint_target = default_pos + action * action_scale

            # 4. MuJoCo PD 控制执行
            ctrl = Kp * (joint_target - qpos) - Kd * qvel
            mujoco.mj_step(model, data)
```

**Sim2Real 的主要 gap 来源及对应措施：**

| Gap 来源 | 域随机化对应措施 |
|---|---|
| 执行器 PD 增益误差 | `randomize_actuator_gains` ×[0.75, 1.25] |
| 质心位置不确定 | `randomize_rigid_body_com` ±0.05m |
| 关节摩擦/惯量 | `randomize_joint_parameters` |
| 控制延迟 | `ActionDelayCfg`（可开启）|
| 地面摩擦 | 摩擦系数随机 [0.6, 1.0] |
| 外部扰动 | 推力干扰 ±1.0 m/s |

---

## 14. 关键设计决策与 Trade-offs

### 1. AMP 奖励混合比例（`amp_task_reward_lerp = 0.7`）

- **70% 任务奖励 + 30% AMP 奖励**
- 如果 AMP 比例过高（→1.0）：策略会过度模仿参考动作，忽略速度跟踪命令
- 如果 AMP 比例过低（→0.0）：步态自然度差，会出现奇怪步态
- 0.7 是经验平衡点：完成任务的同时保持自然步态

### 2. 步态时钟 vs. 无步态时钟

- **有时钟（当前 Walk 配置）**：显式约束步态频率和相位，收敛快但泛化性受限（频率固定）
- **无时钟（新接触奖励）**：更灵活，适合运动速度变化大的场景（Run 任务），已被添加但尚未完全取代

### 3. 对称性损失权重 100

这个权重远大于 PPO surrogate 损失（量级通常 < 1），意味着对称性是**近似硬约束**。好处是大幅减少学习对称运动所需的样本数，坏处是过于严格的对称约束可能阻碍非对称运动（如侧身行走、躲避障碍物）。

### 4. History Length = 10

10 帧历史 × 0.01s/帧 = 过去 0.1 秒的运动信息。足够感知：
- 脚步节律（步态周期约 0.85s，可见 1/8 周期）
- 速度变化趋势
- 接触序列

但不会引入过多的序列计算（避免 LSTM 的困难训练问题），是简单高效的替代。

### 5. 判别器与策略共享优化器

将 policy 和 discriminator 放入同一 Adam 优化器，在一次 `backward()` 中同时更新两者。这简化了代码，但意味着 discriminator 的训练节奏与 PPO 完全同步（均为 5 epoch × 4 mini-batch/iter），而不是像标准 GAN 训练那样独立控制判别器更新次数。

---

## 15. Reward 计算机制与日志解读

### 15.1 Reward 计算流程

Isaac Lab 的 `RewardManager` 采用**时序积分 + Episode 归一化**机制：

```
Step 计算 (每 RL 步):
  value = reward_func() × weight × dt
        = 原始奖励 × 权重 × 0.02秒

Episode 累计 (每步累加):
  _episode_sums += value
  累计 1000 步 ≈ 原始奖励 × 权重 × 20秒

Reset 时返回 (日志显示值):
  logged_value = mean(_episode_sums) / max_episode_length_s
               = 原始奖励 × weight
```

**关键结论**：TensorBoard 中显示的 `Episode/track_lin_vel_xy_exp` 值 = **原始奖励 × weight**，单位是**每秒平均奖励**。

### 15.2 track_lin_vel_xy_exp 详解

**计算公式**（`legged_lab/mdp/rewards.py:70-78`）：
```python
def track_lin_vel_xy_yaw_frame_exp(env, std=0.5):
    # 在偏航坐标系中计算速度误差
    vel_yaw = quat_rotate_inverse(yaw_quat(root_quat_w), root_lin_vel_w)
    error = sum((command[:, :2] - vel_yaw[:, :2])^2)
    return exp(-error / std^2)   # weight = 5.0
```

**解读日志值**：

| 日志显示值 | 原始奖励 | 速度误差 | 评估 |
|-----------|---------|---------|------|
| 4.5-5.0 | 0.9-1.0 | <0.25 m/s | ✅ 优秀 |
| 3.5-4.5 | 0.7-0.9 | 0.25-0.4 m/s | ✅ 良好 |
| 2.0-3.5 | 0.4-0.7 | 0.4-0.6 m/s | ⚠️ 一般 |
| <2.0 | <0.4 | >0.6 m/s | ❌ 需优化 |

**为什么能超过 1？** 因为最终值 = 原始奖励(0-1) × weight(5.0)，理论范围是 **0-5**。

### 15.3 新旧跑步奖励函数对比

| 维度 | 旧版步态时钟奖励 | 新版无时钟奖励 |
|------|-----------------|---------------|
| **核心机制** | 强制匹配预设相位 | 基于物理接触自然涌现 |
| **代表函数** | `gait_feet_frc_perio` | `feet_contact_alternation` |
| **优点** | 收敛快，步态规整 | 泛化性强，适应速度变化 |
| **缺点** | 频率固定，难以变速 | 初期学习更困难 |
| **适用任务** | Walk（固定速度） | Run（变速需求） |

**推荐配置**（`run_cfg.py` 已添加）：
```python
# 双脚交替接触奖励（替代 gait_feet_frc_perio）
feet_contact_alternation = RewTerm(..., weight=2.0)

# 悬空时间奖励（跑步需要更多空中时间）
feet_air_time_reward = RewTerm(..., weight=2.0, params={"target_time": 0.35})

# 前向速度额外奖励（鼓励突破基础速度）
forward_velocity_reward = RewTerm(..., weight=1.5)

# 步频奖励（适应高速步频）
step_frequency_reward = RewTerm(..., weight=1.0, params={"target_freq": 3.0})
```

### 15.4 关键训练指标解读

**速度相关**：
```
Episode/track_lin_vel_xy_exp:   速度追踪奖励（0-5，目标>4）
Episode/forward_velocity_reward: 前向速度额外奖励（目标>1）
Episode/step_frequency_reward:   步频奖励（目标>0.8）
```

**稳定性相关**：
```
Train/mean_episode_length:      平均存活步数（目标>800，最大1000）
Episode/alive_reward:           存活奖励（目标>0.5）
Episode/termination_penalty:    终止惩罚（目标趋近于0）
```

**AMP 相关**：
```
Loss/amp:                       判别器损失（先降后升是正常的）
Loss/amp_policy_pred:           策略数据判别器输出（应趋近于0）
Loss/amp_expert_pred:           专家数据判别器输出（应趋近于1）
```

**策略探索性**：
```
Policy/mean_noise_std:          动作噪声标准差（初始1.0，后期0.1-0.3）
Loss/entropy:                   策略熵（总和，23维，健康范围5-10）
```

### 15.5 常见问题诊断

**问题 1：track_lin_vel_xy_exp 高但实际速度慢**
- 原因：只在低速命令下表现好
- 检查：`ranges.lin_vel_x` 是否包含高速（>2.0 m/s）
- 解决：提升速度命令上限，检查命令分布

**问题 2：amp_loss 先降后升**
- 诊断：✅ **正常现象**
- 原因：15K 迭代后策略开始成功模仿专家，判别器难以区分
- 关注：`amp_policy_pred` 应从 -1 上升到接近 0

**问题 3：mean_episode_length 低（<500）**
- 原因：策略不稳定，频繁摔倒
- 检查：`termination_penalty` 是否过大（当前 -20.0 可尝试 -10.0）
- 检查：新添加的跑步奖励权重是否过高导致动作激进

**问题 4：entropy 不降（>15 经过 20K 迭代）**
- 原因：策略学不到确定性动作
- 检查：奖励信号是否太弱，或任务难度过高
- 解决：提升 `track_lin_vel_xy_exp` 权重或降低 `std`

---

*文档基于代码版本：TienKung-Lab-dev（2025-2026），RSL-RL fork v2.3.1*
