# 训练日志诊断与改进建议规范

## Why

当前训练效果不佳，奖励波动剧烈、无收敛趋势、损失不稳定。需要系统分析问题根因并提供针对性改进方案。

## What Changes

- 分析训练日志中的关键问题
- 诊断环境设计缺陷
- 诊断网络架构问题
- 诊断超参数配置问题
- 提供具体可执行的改进方案

## Impact

- Affected specs: 环境设计、网络架构、训练策略
- Affected code: `envs/sheep_flock.py`, `train_ppo.py`, `onpolicy/algorithms/ppo_actor_critic.py`

---

## ADDED Requirements

### Requirement: 训练日志诊断报告

系统应提供完整的训练诊断报告，包含问题识别和改进建议。

---

## 1. 训练日志分析

### 1.1 奖励趋势分析

**训练数据**（前6100个episode）：

| Episode | Avg Reward | Train Loss | 分析 |
|---------|-----------|------------|------|
| 0 | 68.63 | 50.66 | 初始随机策略偶然高分 |
| 600 | -22.44 | 69.25 | 奖励崩溃 |
| 700 | -20.94 | 93.99 | 损失爆炸 |
| 1500 | 61.36 | 23.74 | 短暂恢复 |
| 2700 | -1.88 | 95.97 | 损失再次爆炸 |
| 2900 | -14.12 | 11.95 | 奖励波动 |
| 3200 | 85.23 | 69.91 | 偶然高分 |
| 3700 | -23.61 | 13.19 | 奖励崩溃 |
| 4400 | 20.27 | 86.71 | 损失高 |
| 6000 | -21.30 | 31.63 | 仍不稳定 |
| 6100 | -9.56 | 42.52 | 无收敛趋势 |

**关键问题**：

1. **奖励无收敛趋势**：6000+ episodes后仍剧烈波动
2. **奖励方差极大**：从-23到+85，跨度超过100
3. **损失不稳定**：从8.45到95.97，波动剧烈
4. **无学习进度**：没有看到持续的性能提升

### 1.2 统计分析

```
奖励统计（Episode 0-6100）：
- 平均奖励: ~25.5
- 最大奖励: 85.23 (Episode 3200)
- 最小奖励: -23.61 (Episode 3700)
- 标准差: ~30.5 (极高)
- 正奖励比例: ~60%
- 负奖励比例: ~40%
```

**诊断结论**：策略没有学到有效的行为模式，奖励波动主要由环境随机性驱动。

---

## 2. 问题诊断

### 2.1 环境设计问题（P0 - 严重）

#### 问题1：奖励函数设计缺陷

**当前奖励函数**（sheep_flock.py:246-306）：

```python
reward = 0.0
# 1. 距离奖励
distance_delta = self.prev_distance - current_distance
reward += distance_delta * 1.0

# 2. 扩散惩罚
if spread > target_spread:
    reward -= (spread - target_spread) * 0.1

# 3. 方向奖励
alignment = np.dot(flock_velocity, target_unit)
reward += alignment * 0.5

# 4. 形状奖励
aspect_ratio = lambda1 / lambda2
reward += 0.1 * (1.0 - aspect_ratio / 3.0)

# 5. 成功奖励
if is_flock_at_target:
    reward += 10.0

# 6. 时间惩罚
reward += -0.01
```

**问题分析**：

| 问题 | 描述 | 影响 |
|------|------|------|
| **奖励尺度不一致** | 距离奖励可达±20，成功奖励仅10 | 距离奖励主导，目标不明确 |
| **奖励稀疏** | 成功奖励只在终点给予 | 缺乏过程指导 |
| **负奖励过多** | 40%的episode获得负奖励 | 策略难以学习正向行为 |
| **奖励噪声大** | 羊群行为随机性导致奖励波动 | 增加学习难度 |

#### 问题2：观测空间设计缺陷

**当前观测空间**（sheep_scenario.py）：

```python
obs = np.zeros(8, dtype=np.float32)
obs[0:2] = flock_center / world_size  # 羊群质心
obs[2:4] = flock_direction  # 羊群方向
obs[4] = flock_spread / 10.0  # 扩散度
obs[5] = num_sheep / 30.0  # 羊群数量
obs[6:8] = target_position / world_size  # 目标位置
```

**问题分析**：

| 问题 | 描述 | 影响 |
|------|------|------|
| **未相对化** | 观测包含绝对位置 | 泛化性差 |
| **缺少关键信息** | 无羊群形状特征 | 策略信息不足 |
| **无机械狗信息** | 只观测第一个机械狗 | 无法评估站位质量 |
| **归一化不当** | 不同尺度混合 | 网络学习困难 |

#### 问题3：动作空间设计缺陷

**当前动作空间**：

```python
action = [radius_mean, radius_std, angle_mean, concentration]
# 范围: [-1, 1] × 4
```

**问题分析**：

| 问题 | 描述 | 影响 |
|------|------|------|
| **κ无约束** | concentration可能→∞ | 所有机械狗站同一点 |
| **角度参考不明** | angle_mean无参考方向 | 与场景旋转相关 |
| **采样随机性大** | Von Mises采样引入噪声 | 策略梯度方差大 |

### 2.2 网络架构问题（P1 - 中等）

**当前网络配置**：

```python
hidden_size = 64
layer_N = 1
use_ReLU = True
use_orthogonal = True
use_feature_normalization = True
```

**问题分析**：

| 问题 | 描述 | 建议 |
|------|------|------|
| **网络过小** | 64隐藏层，1层 | 增大到256，2-3层 |
| **无归一化层** | 只有feature normalization | 添加LayerNorm |
| **无Dropout** | 可能过拟合 | 添加Dropout |

### 2.3 超参数问题（P1 - 中等）

**当前超参数**：

```python
lr = 5e-4
ppo_epoch = 15
num_mini_batch = 1
clip_param = 0.2
entropy_coef = 0.01
gamma = 0.99
gae_lambda = 0.95
episode_length = 100
```

**问题分析**：

| 参数 | 当前值 | 问题 | 建议值 |
|------|--------|------|--------|
| `lr` | 5e-4 | 可能过高 | 3e-4 或 1e-4 |
| `ppo_epoch` | 15 | 过高，可能过拟合 | 5-10 |
| `entropy_coef` | 0.01 | 可能过低 | 0.01-0.1 |
| `episode_length` | 100 | 可能过短 | 150-200 |
| `num_mini_batch` | 1 | 无分批 | 4-8 |

### 2.4 训练策略问题（P0 - 严重）

**当前训练策略**：

```python
# 无课程学习
# 无奖励归一化
# 无观测归一化
# 无梯度裁剪监控
```

**问题分析**：

| 问题 | 描述 | 影响 |
|------|------|------|
| **无课程学习** | 直接训练复杂场景 | 难以收敛 |
| **无奖励归一化** | 奖励尺度不一致 | 训练不稳定 |
| **无观测归一化** | 观测尺度不一致 | 网络学习困难 |
| **无探索策略** | 纯随机探索 | 探索效率低 |

---

## 3. 改进建议

### 3.1 奖励函数改进（P0）

**改进方案**：

```python
def _compute_reward_improved(self):
    """
    改进的奖励函数设计
    
    原则：
    1. 奖励归一化，尺度一致
    2. 添加势能奖励，提供过程指导
    3. 减少负奖励比例
    4. 奖励塑形，引导策略学习
    """
    reward = 0.0
    
    # 1. 势能奖励（核心改进）
    # 使用势能函数提供稠密奖励
    current_distance = self.scenario.get_distance_to_target()
    max_distance = np.linalg.norm(self.world_size)
    
    potential = -current_distance / max_distance
    if self.prev_potential is not None:
        potential_delta = potential - self.prev_potential
        reward += potential_delta * 100.0  # 放大势能奖励
    self.prev_potential = potential
    
    # 2. 距离奖励（缩小权重）
    if self.prev_distance is not None:
        distance_delta = self.prev_distance - current_distance
        reward += distance_delta * 0.5  # 降低权重
    
    # 3. 紧密度奖励（改为正向）
    spread = self.scenario.get_flock_spread()
    target_spread = 5.0
    if spread < target_spread:
        reward += 0.5  # 正向奖励保持紧凑
    else:
        reward -= (spread - target_spread) * 0.05  # 降低惩罚
    
    # 4. 方向奖励
    flock_center = self.scenario.get_flock_center()
    target = self.scenario.get_target_position()
    flock_velocity = self.scenario.get_flock_direction()
    
    target_direction = target - flock_center
    target_direction_norm = np.linalg.norm(target_direction)
    
    if target_direction_norm > 1e-6 and np.linalg.norm(flock_velocity) > 1e-6:
        target_unit = target_direction / target_direction_norm
        alignment = np.dot(flock_velocity, target_unit)
        reward += max(0, alignment) * 0.3  # 只奖励正向对齐
    
    # 5. 成功奖励（增大）
    if self.scenario.is_flock_at_target(threshold=5.0):
        reward += 50.0  # 增大成功奖励
    
    # 6. 时间惩罚（减小）
    reward += -0.005  # 降低时间惩罚
    
    # 7. 奖励裁剪（新增）
    reward = np.clip(reward, -10.0, 100.0)  # 防止极端奖励
    
    return float(reward)
```

**关键改进**：

1. **势能奖励**：提供稠密的过程指导
2. **奖励尺度统一**：所有奖励项在合理范围内
3. **减少负奖励**：更多正向激励
4. **奖励裁剪**：防止极端值

### 3.2 观测空间改进（P0）

**改进方案**：

```python
def get_observation_improved(self):
    """
    改进的观测空间设计
    
    原则：
    1. 全部相对化/归一化
    2. 包含羊群形状信息
    3. 包含机械狗统计信息
    """
    obs = np.zeros(12, dtype=np.float32)
    
    r_max = np.linalg.norm(self.world_size) / 2
    R_ref = self.evasion_radius
    
    # 1. 羊群质心到目标的距离（归一化）
    d_goal = np.linalg.norm(self.flock_center - self.target_position)
    obs[0] = d_goal / r_max
    
    # 2. 羊群→目标方向（单位向量）
    to_target = self.target_position - self.flock_center
    to_target_norm = np.linalg.norm(to_target)
    if to_target_norm > 1e-6:
        to_target = to_target / to_target_norm
    obs[1:3] = to_target
    
    # 3. 羊群形状特征值（归一化）
    lambda1, lambda2 = self.get_flock_shape_eigenvalues()
    obs[3] = lambda1 / (r_max ** 2)
    obs[4] = lambda2 / (r_max ** 2)
    
    # 4. 羊群主方向（相对于目标方向）
    main_direction = self.get_flock_main_direction()
    cos_theta = np.dot(main_direction, to_target)
    sin_theta = main_direction[0] * to_target[1] - main_direction[1] * to_target[0]
    obs[5:7] = [cos_theta, sin_theta]
    
    # 5. 羊群扩散度（归一化）
    spread = self.get_flock_spread()
    obs[7] = spread / r_max
    
    # 6. 机械狗站位统计（归一化）
    herder_distances = [np.linalg.norm(h - self.flock_center) for h in self.herder_positions]
    obs[8] = np.mean(herder_distances) / R_ref  # 平均站位半径
    obs[9] = np.std(herder_distances) / R_ref  # 站位半径标准差
    
    # 7. 机械狗角度分布（归一化）
    angles = [np.arctan2(h[1] - self.flock_center[1], h[0] - self.flock_center[0]) for h in self.herder_positions]
    obs[10] = np.std(angles) / np.pi  # 角度分布均匀度
    
    # 8. 羊群数量（归一化）
    obs[11] = len(self.sheep) / 30.0
    
    return obs
```

### 3.3 动作空间改进（P0）

**改进方案**：

```python
class HighLevelAction:
    """改进的高层动作空间"""
    
    def __init__(self, R_ref=8.0):
        self.R_ref = R_ref
        
        # κ 的约束范围（关键改进）
        self.log_kappa_min = np.log(0.5)   # κ_min = 0.5
        self.log_kappa_max = np.log(5.0)   # κ_max = 5.0
    
    def decode_action(self, raw_action, target_direction):
        """解码动作，添加约束"""
        
        # 1. 站位半径
        mu_r = (raw_action[0] + 1) * self.R_ref  # [0, 2*R_ref]
        mu_r = np.clip(mu_r, self.R_ref * 0.5, self.R_ref * 2)
        
        # 2. 站位半径标准差
        sigma_r = (raw_action[1] + 1) * self.R_ref * 0.25  # [0, 0.5*R_ref]
        sigma_r = np.clip(sigma_r, 0, self.R_ref * 0.5)
        
        # 3. 角度（相对于目标方向）
        mu_theta = raw_action[2] * np.pi  # [-π, π]
        # 转换到世界坐标系
        mu_theta_world = target_direction + mu_theta
        
        # 4. 聚集度（有约束）
        log_kappa = raw_action[3] * 2  # 放大
        log_kappa = np.clip(log_kappa, self.log_kappa_min, self.log_kappa_max)
        kappa = np.exp(log_kappa)
        
        return {
            'mu_r': mu_r,
            'sigma_r': sigma_r,
            'mu_theta': mu_theta_world,
            'kappa': kappa
        }
```

### 3.4 网络架构改进（P1）

**改进方案**：

```python
class ImprovedActorCritic(nn.Module):
    """改进的网络架构"""
    
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        
        # 动作输出
        self.action_mean = nn.Linear(hidden_size // 2, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # 正交初始化
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
```

### 3.5 超参数改进（P1）

**推荐超参数**：

```python
recommended_args = {
    # 学习率
    'lr': 3e-4,  # 降低学习率
    
    # PPO参数
    'ppo_epoch': 10,  # 减少epoch
    'num_mini_batch': 4,  # 增加mini-batch
    'clip_param': 0.2,  # 保持
    'entropy_coef': 0.01,  # 可调
    'value_loss_coef': 0.5,  # 降低value loss权重
    
    # 折扣因子
    'gamma': 0.99,
    'gae_lambda': 0.95,
    
    # 网络架构
    'hidden_size': 256,  # 增大
    'layer_N': 3,  # 增加层数
    
    # 训练长度
    'episode_length': 150,  # 增加
    
    # 梯度裁剪
    'max_grad_norm': 0.5,  # 降低
}
```

### 3.6 训练策略改进（P0）

**课程学习方案**：

```python
curriculum_stages = [
    {
        'name': 'Stage 1: Simple',
        'num_sheep': 3,
        'num_herders': 2,
        'world_size': (30.0, 30.0),
        'episode_length': 50,
        'target_success_rate': 0.8,
        'min_episodes': 100,
    },
    {
        'name': 'Stage 2: Medium',
        'num_sheep': 5,
        'num_herders': 2,
        'world_size': (40.0, 40.0),
        'episode_length': 75,
        'target_success_rate': 0.7,
        'min_episodes': 200,
    },
    {
        'name': 'Stage 3: Target',
        'num_sheep': 6,
        'num_herders': 3,
        'world_size': (30.0, 30.0),
        'episode_length': 100,
        'target_success_rate': 0.6,
        'min_episodes': 500,
    },
]
```

**奖励归一化**：

```python
class RunningMeanStd:
    """运行时均值和标准差"""
    
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
```

---

## 4. 实施优先级

### P0 - 立即修复（本周）

1. **奖励函数改进**
   - 添加势能奖励
   - 统一奖励尺度
   - 减少负奖励比例

2. **观测空间改进**
   - 全部相对化/归一化
   - 添加羊群形状信息

3. **动作空间改进**
   - 添加κ约束
   - 角度相对化

4. **课程学习**
   - 从简单场景开始训练

### P1 - 短期改进（下周）

1. **网络架构改进**
   - 增大隐藏层
   - 添加LayerNorm

2. **超参数调优**
   - 降低学习率
   - 减少PPO epoch

3. **奖励归一化**
   - 实现RunningMeanStd

### P2 - 中期优化（后续）

1. **物理约束**
   - 添加运动学模型

2. **多场景泛化**
   - 配置随机化训练

---

## 5. 预期效果

**改进前**（当前状态）：
- 奖励波动：-23 ~ +85
- 无收敛趋势
- 6000+ episodes无学习进度

**改进后**（预期）：
- 奖励波动：0 ~ +50
- 明显收敛趋势
- 1000 episodes内看到学习进度
- 最终成功率 > 60%
