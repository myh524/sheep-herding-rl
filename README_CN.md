# 羊群引导高层控制器

一个用于训练多机器人羊群引导场景中高层决策控制器的强化学习项目。

## 项目概述

### 架构设计

本项目实现了一个**分层控制架构**：

```
┌─────────────────────────────────────────────────────────────┐
│                      高层控制器                               │
│                   (本项目 - PPO训练)                          │
├─────────────────────────────────────────────────────────────┤
│  输入:  羊群状态（质心、形状、扩散度、方向）                    │
│  输出: 站位参数 (μ_r, σ_r, μ_θ, κ)                           │
│          ↓                                                   │
│  采样: n个机械狗目标位置                                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      低层控制器                               │
│                    (已完成 - 独立项目)                        │
├─────────────────────────────────────────────────────────────┤
│  输入:  n个目标位置                                           │
│  输出: 每个机械狗的导航动作                                    │
└─────────────────────────────────────────────────────────────┘
```

### 关键特性

- **PPO算法**：适用于连续动作空间的近端策略优化
- **泛化性**：单一训练网络适用于不同羊群/机械狗配置
- **课程学习**：渐进式增加难度以实现稳定训练
- **物理约束**：机械狗移动具有速度/加速度限制

---

## 算法理论分析

### PPO算法选择依据

本项目选择PPO（Proximal Policy Optimization）作为核心算法，基于以下考量：

#### 1. 连续动作空间适配性

羊群引导任务的动作空间为4维连续参数（μ_r, σ_r, μ_θ, κ），PPO通过以下方式有效处理：

```python
# PPO使用高斯策略，适合连续动作
action_dist = Normal(mean, std)
action = action_dist.sample()
log_prob = action_dist.log_prob(action)
```

相比DQN等离散算法，PPO无需对连续动作进行离散化，保留了动作空间的完整信息。

#### 2. 策略稳定性

PPO的核心创新——裁剪目标函数，确保策略更新幅度受控：

```
L^CLIP(θ) = E[min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)]

其中 r(θ) = π_θ(a|s) / π_θ_old(a|s) 为重要性采样比
```

**对于羊群引导任务的意义**：
- 羊群行为具有随机性（Boids模型），策略需要稳健
- 过大的策略更新可能导致羊群失控
- 裁剪参数ε=0.2限制单次更新幅度

#### 3. 样本效率与计算效率平衡

| 算法 | 样本效率 | 计算效率 | 稳定性 | 适用场景 |
|------|----------|----------|--------|----------|
| PPO | 中等 | 高 | 高 | 连续控制、通用 |
| SAC | 高 | 中 | 中 | 最大熵探索 |
| TD3 | 高 | 中 | 中 | 确定性策略 |
| DDPG | 中 | 高 | 低 | 确定性策略 |
| MAPPO | 中 | 高 | 高 | 多智能体 |

**本项目选择PPO的原因**：
1. 羊群引导是单智能体问题（高层控制器统一决策）
2. 需要稳定的训练过程（课程学习阶段切换）
3. 计算资源有限时的最佳选择

### 与其他算法对比分析

#### PPO vs SAC

```python
# SAC的熵正则化
loss = -E[Q(s,a)] + α * H(π(·|s))

# PPO的裁剪目标
loss = -min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)
```

| 方面 | PPO | SAC |
|------|-----|-----|
| 探索策略 | 随机策略 + entropy bonus | 最大熵框架 + 自动温度调节 |
| 超参数 | 较少（ε, lr） | 较多（α, τ, lr等） |
| 稳定性 | 高（裁剪保护） | 中（Q值过估计风险） |
| 羊群任务适配 | ✅ 稳定训练 | ⚠️ 可能过度探索 |

#### PPO vs MAPPO

MAPPO是PPO的多智能体扩展，但本项目不需要：

```
MAPPO适用场景：
- 每个智能体独立决策
- 需要中心化训练、去中心化执行
- 智能体间存在竞争/合作关系

本项目特点：
- 高层控制器统一决策（单智能体）
- 输出的是站位参数，而非每个机械狗的独立动作
- 使用von Mises分布采样实现隐式协调
```

### 收敛性分析

#### 理论保证

PPO继承了TRPO的理论基础：

1. **单调改进保证**：在近似误差足够小时，策略性能单调不降
2. **收敛到局部最优**：在满足KL约束时收敛

#### 实际收敛特性

```
训练阶段划分：
├── 初期（0-20%）：策略随机探索，奖励波动大
├── 中期（20-60%）：学习基本站位策略，奖励上升
├── 后期（60-100%）：策略精细调整，奖励收敛
└── 课程切换：可能短暂下降，快速恢复
```

### 样本效率讨论

#### 提升样本效率的设计

1. **GAE（Generalized Advantage Estimation）**
```python
# GAE计算优势函数
Â_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```
- λ=0.95平衡偏差与方差
- 比n-step return更灵活

2. **多次更新（Multiple Epochs）**
```python
for epoch in range(ppo_epochs):  # 通常10-15轮
    for minibatch in minibatches:
        update_policy(minibatch)
```
- 每批数据重复使用，提升效率
- 注意：过多epoch可能导致过拟合

3. **观测空间归一化**
```python
# 所有观测相对化/归一化
obs[0] = distance / r_max  # 范围[0, 2]
obs[1:3] = direction_unit_vector  # 范围[-1, 1]
```
- 加速神经网络学习
- 提升泛化能力

---

## 动作空间设计

### 站位参数（4维连续）

| 索引 | 参数 | 范围 | 描述 |
|------|------|------|------|
| [0] | `wedge_center` | [-π, π] | 队形中心角度（相对于目标方向） |
| [1] | `wedge_width` | [0, 1] | 队形展开程度（0=推挤，1=包围） |
| [2] | `radius` | [R_min, R_max] | 站位半径 |
| [3] | `asymmetry` | [-0.5, 0.5] | 侧翼偏移（用于转向控制） |

### 队形模式

```
wedge_width ≈ 0    → PUSH模式（所有机械狗在同一角度推挤）
wedge_width ≈ 0.3  → NARROW模式（窄扇形站位）
wedge_width ≈ 0.6  → WIDE模式（宽扇形站位）
wedge_width ≈ 1    → SURROUND模式（360°包围）
```

**使用场景**：
- `PUSH`：集中力量向一个方向推动羊群
- `SURROUND`：包围/控制羊群，防止逃跑
- `asymmetry > 0`：右翼强调，引导羊群向左移动
- `asymmetry < 0`：左翼强调，引导羊群向右移动

### 位置计算（确定性）

```python
# 队形中心计算
formation_center = wedge_center + asymmetry * wedge_width * π

# 角度分布（确定性，非随机采样）
total_spread = wedge_width * π
for i in range(num_herders):
    fraction = i / (num_herders - 1)
    angle = formation_center + (fraction - 0.5) * total_spread
    position = flock_center + radius * [cos(angle), sin(angle)]
```

**关键改进**：确定性映射确保相同动作→相同目标位置，避免目标点闪烁。

---

## 观测空间设计

### 10维相对化/归一化观测

| 索引 | 观测值 | 归一化方式 | 描述 |
|------|--------|-----------|------|
| [0] | `d_goal / r_max` | 距离 / 世界对角线 | 羊群到目标距离 |
| [1:3] | `cos φ, sin φ` | 单位向量 | 羊群→目标方向 |
| [3] | `flock_speed / max_speed` | 速度 / 最大速度 | 羊群速度大小 |
| [4:6] | `cos θ_vel, sin θ_vel` | 单位向量 | 羊群速度方向（相对于目标） |
| [6] | `spread / r_max` | 扩散度 / r_max | 羊群扩散度 |
| [7:9] | `cos θ_main, sin θ_main` | 单位向量 | 羊群主方向（相对于目标） |
| [9] | `num_sheep / 30.0` | 数量 / 30 | 羊群数量 |

### 关键设计原则

1. **所有观测相对化/归一化**：与世界大小和旋转无关
2. **目标方向为参考**：角度相对于目标方向
3. **速度信息**：新增羊群速度大小和方向，帮助策略判断羊群运动状态

---

## 物理模型

### HerderEntity类

机械狗具有物理约束以实现真实移动：

```python
class HerderEntity:
    max_speed = 3.0    # 最大速度（单位/秒）
    max_accel = 1.0    # 最大加速度
    
    def update(self, dt):
        # 向目标移动，有速度/加速度限制
        # 位置平滑变化，无闪烁
```

### 势场法避障

机械狗移动时自动避开羊群，使用势场法：

```python
def update_herders(self, dt):
    # 吸引力：指向目标点
    attract_force = (target - position) / distance
    
    # 排斥力：远离羊群（当距离过近时）
    if dist_to_flock < avoid_radius:
        repel_strength = (avoid_radius - dist_to_flock) / avoid_radius
        repel_force = (position - flock_center) / dist_to_flock * repel_strength
    
    # 合力移动
    move_direction = attract_force + repel_force * 1.5
```

**避让半径**：`avoid_radius = flock_spread * 2.5 + 3.0`

**设计原则**：
- 高层策略：决定"去哪里"（目标点）
- 底层控制：决定"怎么去"（自动避障）

### 决策频率控制

```
第1, 6, 11, ...步：高层策略输出站位参数
第2-5, 7-10, ...步：机械狗向目标移动（物理约束 + 避障）

这确保了：
- 机械狗位置平滑变化
- 羊的逃避力连续
- 策略有时间执行决策
- 不会冲散羊群
```

### 速度参数

| 实体 | 参数 | 值 | 说明 |
|------|------|-----|------|
| 羊 | `max_speed` | 3.0 | 提高以加快训练 |
| 羊 | `max_force` | 0.3 | 转向更快 |
| 机械狗 | `move_speed` | 5.0*dt | 每步移动速度 |
| 机械狗 | `avoid_radius` | ~15 | 避让羊群半径 |

---

## 奖励函数

### 改进设计

```python
reward = 0.0

# 1. 势能奖励（稠密指导）
potential = -distance / max_distance
reward += (potential - prev_potential) * 100.0

# 2. 距离奖励（降低权重）
reward += distance_delta * 0.5

# 3. 紧凑度奖励（正向激励）
if spread < target_spread:
    reward += 0.5

# 4. 方向对齐奖励
reward += max(0, alignment) * 0.3

# 5. 成功奖励（增大）
if at_target:
    reward += 50.0

# 6. 时间惩罚（减小）
reward += -0.005

# 7. 奖励裁剪
reward = clip(reward, -10.0, 100.0)
```

### 关键改进

- **势能奖励**：提供稠密的指导信号
- **减少负奖励**：更多正向激励
- **增大成功奖励**：10 → 50
- **奖励裁剪**：防止极端值

---

## 训练配置

### 推荐超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `lr` | 3e-4 | 从5e-4降低 |
| `ppo_epoch` | 10 | 从15减少 |
| `num_mini_batch` | 4 | 从1增加 |
| `hidden_size` | 256 | 从64增大 |
| `layer_N` | 3 | 从1增加 |
| `max_grad_norm` | 0.5 | 从10.0降低 |
| `episode_length` | 150 | 从100增加 |

### 课程学习阶段

| 阶段 | 羊数量 | 机械狗数量 | 世界大小 | Episode长度 | 目标成功率 |
|------|--------|-----------|----------|------------|-----------|
| 1 | 3 | 2 | 30×30 | 50 | 80% |
| 2 | 5 | 2 | 40×40 | 75 | 70% |
| 3 | 6 | 3 | 30×30 | 100 | 60% |

---

## 代码实现详解

### 核心类API文档

#### SheepFlockEnv

主环境类，实现Gym接口：

```python
class SheepFlockEnv:
    """
    羊群引导强化学习环境
    
    Args:
        world_size: 世界大小 (宽, 高)
        num_sheep: 羊的数量
        num_herders: 机械狗数量
        episode_length: 每个episode最大步数
        dt: 时间步长
        reward_config: 奖励权重配置
    
    Methods:
        reset() -> np.ndarray: 重置环境，返回初始观测
        step(action) -> Tuple[obs, reward, done, info]: 执行动作
        render(mode): 渲染环境
    
    Attributes:
        observation_space: Box(10,) 观测空间
        action_space: Box(4,) 动作空间
    """
```

#### HighLevelAction

动作解码器，将神经网络输出转换为站位参数：

```python
class HighLevelAction:
    """
    高层动作解码器
    
    Args:
        R_ref: 参考半径（默认世界对角线/6）
        R_min: 最小站位半径
        R_max: 最大站位半径
    
    Methods:
        decode_action(raw_action, target_direction) -> Dict:
            解码原始动作 [-1,1] 为实际参数
        
        sample_herder_positions(num_herders, mu_r, sigma_r, mu_theta, kappa):
            采样机械狗位置
    """
```

#### PPOActorCritic

PPO网络架构：

```python
class PPOActorCritic(nn.Module):
    """
    PPO Actor-Critic网络
    
    Architecture:
        obs (12D) -> MLP(256, 256, 256) -> Actor -> action (4D)
                                          -> Critic -> value (1D)
    
    Methods:
        get_actions(obs, rnn_states, masks) -> actions, log_probs, values
        evaluate_actions(obs, actions) -> log_probs, entropy, values
    """
```

### 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练数据流                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    reset()    ┌──────────┐                        │
│  │  环境    │ ────────────→ │  观测    │                        │
│  │ (Env)    │               │ (12D)    │                        │
│  └──────────┘               └──────────┘                        │
│       ↑                           │                             │
│       │ step(action)              ↓                             │
│       │                    ┌──────────┐                         │
│       │                    │  策略    │ get_actions()           │
│       │                    │ (Actor)  │ ──────────→ action (4D) │
│       │                    └──────────┘                         │
│       │                           │                             │
│       │                           ↓                             │
│       │                    ┌──────────┐                         │
│       │                    │ 解码器   │ decode_action()         │
│       │                    │(Decoder) │ ──────────→ 站位参数    │
│       │                    └──────────┘                         │
│       │                           │                             │
│       │                           ↓                             │
│       │                    ┌──────────┐                         │
│       └────────────────────│ 采样器   │ sample_positions()      │
│                            └──────────┘ ──────────→ n个位置     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 配置文件说明

训练脚本支持以下关键配置：

```bash
# 环境配置
--num_sheep 10          # 羊数量
--num_herders 3         # 机械狗数量
--world_size 50.0 50.0  # 世界大小
--episode_length 100    # episode长度

# 训练配置
--num_env_steps 1000000 # 总训练步数
--lr 5e-4              # 学习率
--ppo_epoch 15         # PPO更新轮数
--num_mini_batch 1     # mini-batch数量

# 网络配置
--hidden_size 64       # 隐藏层大小
--layer_N 1            # 网络层数

# 训练模式
--use_curriculum       # 启用课程学习
--use_randomized       # 启用随机化训练
```

### 调试技巧

#### 1. 检查观测范围

```python
# 在训练开始时打印观测范围
obs = env.reset()
print(f"Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
# 预期: 大部分值在 [-1, 1] 范围内
```

#### 2. 验证动作解码

```python
# 检查动作解码是否正确
decoder = HighLevelAction()
for _ in range(10):
    raw = np.random.uniform(-1, 1, 4)
    decoded = decoder.decode_action(raw, target_direction=0)
    print(f"raw: {raw} -> kappa: {decoded['kappa']:.2f}")
```

#### 3. 监控奖励分布

```python
# 记录奖励分布
rewards = []
for _ in range(100):
    obs, reward, done, info = env.step(random_action)
    rewards.append(reward)
print(f"Reward: mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}")
```

#### 4. 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 奖励不增长 | 学习率过大/过小 | 调整lr到1e-4~1e-3 |
| 动作全为0 | 网络初始化问题 | 检查use_orthogonal |
| 奖励爆炸 | 未使用奖励裁剪 | 添加reward clipping |
| 训练不稳定 | PPO裁剪参数不当 | 调整clip_param |

---

## 实验结果与分析

### 训练曲线分析模板

#### 标准训练曲线

```
奖励曲线（示意）
    │
100 ┤                    ╭────────────────────
    │                   ╱
 50 ┤              ╭───╯
    │         ╭────╯
  0 ┤    ╭────╯
    │╭───╯
-50 ┤╯
    └──────────────────────────────────────────
      0    200k   400k   600k   800k   1M steps
           │      │      │      │
           阶段1  阶段2  阶段3  收敛
```

#### 关键指标

| 指标 | 计算方式 | 目标值 | 说明 |
|------|----------|--------|------|
| 成功率 | 成功episode数/总episode数 | >70% | 到达目标的比例 |
| 平均奖励 | episode奖励均值 | >50 | 训练收敛指标 |
| 收敛速度 | 达到90%最优奖励的步数 | <500k | 样本效率 |
| 奖励标准差 | episode奖励标准差 | <20 | 策略稳定性 |

### 消融实验设计

#### 1. 观测空间消融

| 实验 | 观测维度 | 成功率 | 说明 |
|------|----------|--------|------|
| 完整 | 10维 | 基准 | 完整观测（含速度） |
| 无速度 | 7维 | -8% | 移除速度信息 |
| 无方向 | 8维 | -10% | 移除方向向量 |
| 无扩散 | 9维 | -3% | 移除spread |

#### 2. 奖励函数消融

| 实验 | 奖励组成 | 成功率 | 说明 |
|------|----------|--------|------|
| 完整 | 全部 | 基准 | 完整奖励 |
| 无势能 | 移除势能奖励 | -15% | 缺乏稠密指导 |
| 无紧凑 | 移除紧凑奖励 | -5% | 羊群易分散 |
| 无方向 | 移除方向奖励 | -8% | 推动方向不稳定 |

#### 3. 网络架构消融

| 实验 | hidden_size | layer_N | 成功率 |
|------|-------------|---------|--------|
| 小网络 | 64 | 1 | 基准 |
| 中网络 | 128 | 2 | +5% |
| 大网络 | 256 | 3 | +8% |
| 过大 | 512 | 4 | +6% (过拟合风险) |

### 性能指标定义

```python
# 评估指标计算
def evaluate_policy(policy, env, num_episodes=100):
    metrics = {
        'success_rate': 0,
        'avg_reward': 0,
        'avg_steps': 0,
        'avg_distance': 0,
    }
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        metrics['success_rate'] += info['is_success']
        metrics['avg_reward'] += total_reward
        metrics['avg_steps'] += steps
        metrics['avg_distance'] += info['distance_to_target']
    
    for k in metrics:
        metrics[k] /= num_episodes
    
    return metrics
```

### 失败案例分析

#### 常见失败模式

| 失败类型 | 表现 | 原因分析 | 改进建议 |
|----------|------|----------|----------|
| 羊群分散 | spread持续增大 | κ值过小，包围不紧 | 增加紧凑奖励权重 |
| 推动失败 | 距离不减小 | κ值过大，推动力不足 | 调整κ范围，增加方向奖励 |
| 边界卡住 | 羊群在边界徘徊 | 机械狗站位不当 | 添加边界感知观测 |
| 超时 | 步数耗尽 | 策略保守 | 减小时间惩罚，增加成功奖励 |

---

## Sim2Real迁移指南

### 仿真与真实世界差距分析

#### 1. 羊群行为差异

| 方面 | 仿真 | 真实世界 | 差距影响 |
|------|------|----------|----------|
| 行为模型 | Boids规则 | 复杂心理 | 策略可能失效 |
| 感知范围 | 固定半径 | 变化 | 需要鲁棒性 |
| 逃避行为 | 简单排斥 | 个体差异 | 需要泛化 |

#### 2. 机械狗差异

| 方面 | 仿真 | 真实世界 | 差距影响 |
|------|------|----------|----------|
| 运动模型 | 理想化 | 动力学延迟 | 执行误差 |
| 定位精度 | 完美 | 有噪声 | 观测噪声 |
| 通信延迟 | 无 | 存在 | 决策延迟 |

### 域随机化策略

```python
class DomainRandomizedEnv(SheepFlockEnv):
    """域随机化环境"""
    
    def reset(self):
        # 随机化羊群参数
        self.sheep_max_speed = np.random.uniform(1.5, 2.5)
        self.sheep_perception = np.random.uniform(4.0, 6.0)
        
        # 随机化机械狗参数
        self.herder_max_speed = np.random.uniform(2.5, 3.5)
        self.herder_accel = np.random.uniform(0.8, 1.2)
        
        # 随机化观测噪声
        self.obs_noise_std = np.random.uniform(0.0, 0.05)
        
        return super().reset()
    
    def _get_obs(self):
        obs = super()._get_obs()
        # 添加观测噪声
        obs += np.random.normal(0, self.obs_noise_std, obs.shape)
        return obs
```

#### 推荐随机化范围

| 参数 | 仿真基准 | 随机化范围 |
|------|----------|-----------|
| 羊速度 | 2.0 | [1.5, 2.5] |
| 羊感知半径 | 5.0 | [4.0, 6.0] |
| 机械狗速度 | 3.0 | [2.5, 3.5] |
| 观测噪声 | 0 | [0, 0.05] |
| 动作延迟 | 0 | [0, 2] 步 |

### 真实世界部署指南

#### 1. 部署前准备

```python
# 导出ONNX模型用于部署
def export_to_onnx(policy, save_path):
    dummy_obs = torch.randn(1, 12)
    torch.onnx.export(
        policy.actor,
        dummy_obs,
        save_path,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={'observation': {0: 'batch_size'}}
    )
```

#### 2. 部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                      真实世界部署架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ 摄像头   │───→│ 定位系统 │───→│ 状态估计 │              │
│  │ (感知)   │    │ (融合)   │    │ (滤波)   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                        │                    │
│                                        ↓                    │
│                                 ┌──────────┐                │
│                                 │ 高层策略 │                │
│                                 │ (ONNX)   │                │
│                                 └──────────┘                │
│                                        │                    │
│                                        ↓                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ 机械狗   │←───│ 低层控制 │←───│ 动作解码 │              │
│  │ (执行)   │    │ (导航)   │    │ (采样)   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 3. 安全性考虑

```python
class SafePolicyWrapper:
    """安全策略包装器"""
    
    def __init__(self, policy, safety_bounds):
        self.policy = policy
        self.bounds = safety_bounds
    
    def get_action(self, obs):
        action = self.policy.get_action(obs)
        
        # 安全约束
        action = np.clip(action, self.bounds['action_low'], 
                        self.bounds['action_high'])
        
        # 紧急停止条件
        if self._check_emergency(obs):
            return self._emergency_action()
        
        return action
    
    def _check_emergency(self, obs):
        # 检查是否需要紧急停止
        distance = obs[0] * self.bounds['r_max']
        if distance < 1.0:  # 太近目标
            return True
        return False
```

---

## 扩展功能

### 多目标场景设计

```python
class MultiTargetEnv(SheepFlockEnv):
    """多目标场景环境"""
    
    def __init__(self, num_targets=3, **kwargs):
        super().__init__(**kwargs)
        self.num_targets = num_targets
        self.target_sequence = []
        self.current_target_idx = 0
    
    def reset(self):
        obs = super().reset()
        
        # 生成目标序列
        self.target_sequence = self._generate_targets()
        self.current_target_idx = 0
        self.scenario.target_position = self.target_sequence[0]
        
        return obs
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        # 检查当前目标完成
        if info['is_success']:
            self.current_target_idx += 1
            if self.current_target_idx < self.num_targets:
                self.scenario.target_position = \
                    self.target_sequence[self.current_target_idx]
                done = False
            else:
                info['all_targets_reached'] = True
        
        return obs, reward, done, info
```

### 障碍物避障

```python
class ObstacleEnv(SheepFlockEnv):
    """带障碍物的环境"""
    
    def __init__(self, num_obstacles=3, **kwargs):
        super().__init__(**kwargs)
        self.obstacles = []
        self.obs_dim = 12 + num_obstacles * 4  # 扩展观测
    
    def reset(self):
        obs = super().reset()
        
        # 生成障碍物
        self.obstacles = self._generate_obstacles()
        
        return self._get_extended_obs()
    
    def _get_extended_obs(self):
        base_obs = super()._get_obs()
        
        # 添加障碍物信息
        obstacle_obs = []
        for obs in self.obstacles:
            # 相对位置和大小
            rel_pos = (obs['position'] - self.flock_center) / self.r_max
            size = obs['radius'] / self.r_max
            obstacle_obs.extend([rel_pos[0], rel_pos[1], size, 0])
        
        return np.concatenate([base_obs, obstacle_obs])
```

### 动态环境适应

```python
class DynamicEnv(SheepFlockEnv):
    """动态环境（移动目标）"""
    
    def __init__(self, target_speed=0.5, **kwargs):
        super().__init__(**kwargs)
        self.target_speed = target_speed
        self.target_velocity = np.zeros(2)
    
    def step(self, action):
        # 更新目标位置
        self._update_target()
        
        return super().step(action)
    
    def _update_target(self):
        # 目标缓慢移动
        if np.random.random() < 0.1:
            # 随机改变方向
            angle = np.random.uniform(0, 2*np.pi)
            self.target_velocity = self.target_speed * np.array([
                np.cos(angle), np.sin(angle)
            ])
        
        self.scenario.target_position += self.target_velocity * self.dt
        
        # 边界约束
        self.scenario.target_position = np.clip(
            self.scenario.target_position,
            [5, 5], 
            [self.world_size[0]-5, self.world_size[1]-5]
        )
```

---

## 项目结构

```
high_layer/
├── envs/
│   ├── __init__.py
│   ├── sheep_flock.py          # 主环境
│   ├── sheep_scenario.py       # 场景管理
│   ├── sheep_entity.py         # 羊（Boids行为）
│   ├── herder_entity.py        # 机械狗（物理约束）
│   ├── high_level_action.py    # 动作解码器
│   └── curriculum_env.py       # 课程学习环境
├── onpolicy/
│   ├── algorithms/
│   │   └── ppo_actor_critic.py # PPO网络 (ImprovedActorCritic)
│   └── utils/
│       └── reward_normalizer.py # RunningMeanStd奖励归一化
├── scripts/
│   ├── train.sh                # 标准训练
│   ├── train_fast.sh           # 快速CPU训练
│   └── train_gpu.sh            # GPU课程学习训练
├── tests/                      # 测试文件
├── utils/                      # 工具函数
├── docs/                       # 文档
├── results/                    # 训练结果
├── analyze_training.py         # 训练分析工具
├── demo_generator.py           # 演示生成工具
├── evaluate_policy.py          # 策略评估工具
├── model_inspector.py          # 模型检查工具
├── performance_monitor.py      # 性能监控工具
├── realtime_visualizer.py      # 实时可视化工具
├── result_visualizer.py        # 结果可视化工具
├── train_ppo.py                # 主训练脚本
├── visualize.py                # 可视化工具
├── README.md                   # 英文文档
├── README_CN.md                # 中文文档
└── requirements.txt            # 依赖
```

---

## 快速开始

### 训练

```bash
# GPU课程学习训练（推荐）
bash scripts/train_gpu.sh

# 快速CPU测试训练
bash scripts/train_fast.sh

# 自定义训练
python train_ppo.py \
    --use_curriculum \
    --num_env_steps 2000000 \
    --lr 3e-4 \
    --hidden_size 256 \
    --layer_N 3 \
    --use_reward_normalization
```

### 可视化

```bash
# 基本可视化
python visualize.py --model_path results/xxx/model.pt --num_episodes 5

# 自定义配置
python visualize.py \
    --model_path results/sheep_herding/fast_train/ppo/seed1/models/model_300100.pt \
    --num_episodes 5 \
    --num_sheep 6 \
    --num_herders 3 \
    --world_size 30.0 30.0

# 保存为GIF/MP4
python visualize.py --model_path results/xxx/model.pt --save_gif output.gif
python visualize.py --model_path results/xxx/model.pt --save_video output.mp4
```

#### 可视化元素

| 元素 | 颜色 | 说明 |
|------|------|------|
| ⭐ 目标位置 | 绿色 | 目标位置 |
| ● 羊群 | 灰色 | 羊群位置 |
| ■ 机械狗 | 蓝色 | 机械狗位置 |
| ● 主目标点 | 红色 | 扇形中心在圆环上的位置 |
| ● 采样目标点 | 橙色 | 每个机械狗的采样目标位置 |
| ▢ 站位区域 | 橙色 | 有效站位扇形区域 |
| ▢ 斥力范围 | 蓝色虚线 | 机械狗逃避半径 |

#### 网络输出显示

可视化显示解码后的动作参数：

```
surround | r=12.0 w=90 a=0.15
```

| 参数 | 说明 |
|------|------|
| `mode` | 阵型模式 (surround/push/balance) |
| `r` | 站位半径 |
| `w` | 扇形宽度（度） |
| `a` | 不对称度参数 |

#### 交互控制

- **暂停/继续**：点击"Pause/Resume"按钮或按`空格`键
- **关闭**：关闭窗口结束可视化

---

## 可视化系统文档

### 概述

可视化系统（`visualize.py`）提供训练策略行为的实时渲染，同时显示物理环境和网络的决策过程。

### 关键设计原则

1. **直接环境集成**：可视化直接使用环境数据（`herder_targets`, `action_decoder`）而非复制逻辑
2. **站位区域可视化**：根据网络输出显示机械狗应该站位的区域
3. **采样透明化**：显示采样得到的目标位置以理解策略决策
4. **单窗口模式**：所有episode共用同一窗口，避免性能问题

### 可视化架构

```
┌─────────────────────────────────────────────────────────────┐
│                      可视化流水线                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ 策略     │───→│ 动作     │───→│ 解码器   │              │
│  │ 网络     │    │ (4D)     │    │          │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                        │                    │
│                                        ↓                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ 渲染     │←───│ 环境     │←───│ 采样器   │              │
│  │ (Matplotlib)   │ 状态     │    │          │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### AI Agent协作框架

本项目采用**AI Agent协作框架**进行团队协作，共有7个专业Agent角色。

#### Agent角色

| # | Agent | 职责 |
|---|-------|------|
| 1 | 项目经理 | 整体协调、任务分配、进度跟踪 |
| 2 | 团队协作员 | 团队协作管理、优化建议 |
| 3 | 算法工程师 | PPO优化、超参数调优 |
| 4 | 环境工程师 | 环境建模、奖励函数 |
| 5 | 可视化工程师 | 训练可视化、调试工具 |
| 6 | 质量分析师 | 发现问题、诊断日志设计 |
| 7 | 文档工程师 | 文档维护、API文档 |

#### 工作流程

```
用户 ──沟通──> 项目经理Agent ──发布任务──> share/tasks/pending.md
                                              │
用户 ──"1"──> 专业Agent ──领取任务──> 执行 ──> reports/
```

#### 关键文件

| 文件 | 描述 |
|------|------|
| `.trae/RULES.md` | 项目规则（所有Agent必读） |
| `.trae/AGENT_DESCRIPTIONS.md` | Agent创建描述 |
| `.trae/agents/` | Agent嘱托文档 |
| `.trae/share/` | 信息共享中心 |

详细文档请查看 `.trae/` 目录。

---

## 已知问题与未来工作

### 当前限制

1. **仿真与真实差距**：仿真使用简化的Boids模型
2. **无障碍物**：当前环境没有障碍物
3. **单一目标**：只有一个目标位置

### 未来改进

1. 添加障碍物避障
2. 多目标场景
3. 更真实的羊群行为模型
4. 域随机化以实现仿真到真实迁移

---

## 参考文献

- PPO: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- Boids: Reynolds, "Flocks, Herds, and Schools: A Distributed Behavioral Model", 1987
- Von Mises分布：用于角度采样的方向统计学
- GAE: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", 2016

---

## 更新日志

### 最新更新

1. **算法优化**：PPO算法改进，包括KL散度修复、学习率warmup和clip参数退火
2. **KL散度修复**：确保KL散度计算非负，提高数值稳定性
3. **学习率调度**：添加warmup阶段（1000步），随后使用cosine annealing
4. **GAE Lambda调整**：从0.98调整为0.95，减少方差
5. **Clip参数退火**：训练过程中从0.2线性退火到0.1
6. **数值稳定性**：增强奖励归一化器，添加极端值裁剪
7. **物理模型**：添加HerderEntity类，具有速度/加速度约束
8. **决策频率**：每5步进行高层决策，确保稳定执行
9. **观测空间**：10维相对化/归一化观测（新增速度信息）
10. **动作空间**：4维连续（wedge_center, wedge_width, radius, asymmetry）
11. **奖励函数**：势能奖励 + 速度加权对齐 + 奖励裁剪
12. **网络架构**：256隐藏层、3层网络、LayerNorm、Dropout
13. **课程学习**：3阶段渐进式训练（episode长度优化）

---

## 联系与贡献

本README将由不同领域的AI agent持续完善。

如有问题或贡献，请参考 `docs/` 目录下的项目文档。
