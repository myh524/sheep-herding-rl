# 羊群引导RL项目优化方案（精简版）

## 一、观测空间分析与改进

### 1.1 当前观测空间分析（12维）

| 索引 | 当前内容 | 重要性 | 建议 |
|------|----------|--------|------|
| 0 | 到目标距离 | ⭐⭐⭐ | 保留 |
| 1-2 | 目标方向向量 | ⭐⭐⭐ | 保留 |
| 3-4 | 羊群形状特征值λ1, λ2 | ⭐ | **删除**（用扩散度替代） |
| 5-6 | 羊群主方向 | ⭐⭐ | 保留 |
| 7 | 羊群扩散度 | ⭐⭐⭐ | 保留 |
| 8-9 | 机械狗站位半径统计 | ⭐ | **删除**（策略不关心） |
| 10 | 机械狗角度分布 | ⭐ | **删除**（策略不关心） |
| 11 | 羊群数量 | ⭐⭐ | 保留 |

**核心问题**:
1. ❌ **缺少羊群速度**: 无法判断羊群是在移动还是静止
2. ❌ **冗余信息**: 特征值、站位统计对策略帮助不大

### 1.2 精简后的观测空间（10维）

| 索引 | 内容 | 描述 | 物理意义 |
|------|------|------|----------|
| 0 | d_goal / r_max | 到目标距离归一化 | 多远到达 |
| 1-2 | cos φ, sin φ | 目标方向单位向量 | 哪个方向 |
| 3 | flock_speed / max_speed | 羊群速度大小归一化 | **新增**: 羊群移动多快 |
| 4-5 | cos θ_vel, sin θ_vel | 羊群速度方向（相对目标方向） | **新增**: 羊群往哪移动 |
| 6 | spread / r_max | 羊群扩散度归一化 | 羊群多分散 |
| 7-8 | cos θ_main, sin θ_main | 羊群主方向（相对目标方向） | 羊群朝向 |
| 9 | num_sheep / 30.0 | 羊群数量归一化 | 羊群规模 |

**精简原则**:
- ✅ 保留核心信息：距离、方向、扩散度
- ✅ 新增关键信息：羊群速度（大小+方向）
- ❌ 删除冗余信息：特征值、站位统计

**为什么速度信息重要**:
```
场景1: 羊群静止在目标附近 → 策略应保持包围
场景2: 羊群快速移动 → 策略应加速追赶
场景3: 羊群向目标方向移动 → 策略应顺势引导
场景4: 羊群背离目标移动 → 策略应拦截
```

### 1.3 观测空间对比

| 维度 | 修改前 | 修改后 | 变化 |
|------|--------|--------|------|
| 总维度 | 12 | **10** | -2 |
| 速度信息 | 无 | **有(3维)** | 新增 |
| 冗余信息 | 有(5维) | **无** | 删除 |

---

## 二、奖励函数调整

### 2.1 当前问题

```python
# 问题1: 势能奖励系数过大
reward += (potential - prev_potential) * 100.0  # 系数100太大

# 问题2: 距离奖励与势能奖励重复
reward += distance_delta * 0.5  # 与势能奖励高度相关

# 问题3: 成功奖励过大
reward += 50.0  # 可能导致策略只追求成功而忽视过程
```

### 2.2 改进方案

```python
def _compute_reward(self) -> float:
    reward = 0.0
    
    current_distance = self.scenario.get_distance_to_target()
    max_distance = np.linalg.norm(self.world_size)
    max_distance = max(max_distance, 1.0)
    
    # 1. 势能奖励（降低系数）- 核心引导信号
    potential = -current_distance / max_distance
    if self.prev_potential is not None:
        potential_delta = potential - self.prev_potential
        reward += potential_delta * 10.0  # 从100降到10
    self.prev_potential = potential
    
    # 2. 移除重复的距离奖励
    self.prev_distance = current_distance
    
    # 3. 羊群紧密度奖励
    spread = self.scenario.get_flock_spread()
    target_spread = 5.0
    if spread < target_spread:
        reward += 0.2
    else:
        reward -= (spread - target_spread) * 0.02
    
    # 4. 方向对齐奖励（速度加权）
    flock_center = self.scenario.get_flock_center()
    target = self.scenario.get_target_position()
    flock_velocity = self.scenario.get_flock_direction()
    flock_speed = np.linalg.norm(
        np.mean(np.array([s.velocity for s in self.scenario.sheep]), axis=0)
    )
    
    target_direction = target - flock_center
    target_direction_norm = np.linalg.norm(target_direction)
    
    if target_direction_norm > 1e-6 and flock_speed > 0.1:
        target_unit = target_direction / target_direction_norm
        alignment = np.dot(flock_velocity, target_unit)
        reward += max(0, alignment) * flock_speed * 0.3
    
    # 5. 成功奖励（降低）
    if self.scenario.is_flock_at_target(threshold=5.0):
        reward += 20.0  # 从50降到20
    
    # 6. 时间惩罚
    reward += -0.01
    
    # 最终裁剪
    reward = np.clip(reward, -5.0, 30.0)
    
    if np.isnan(reward) or np.isinf(reward):
        reward = 0.0
    
    return float(reward)
```

### 2.3 奖励函数对比

| 组件 | 修改前 | 修改后 | 原因 |
|------|--------|--------|------|
| 势能奖励系数 | 100.0 | **10.0** | 降低信号放大 |
| 距离改善奖励 | 0.5 | **删除** | 与势能重复 |
| 紧密度奖励 | 0.5/-0.05 | **0.2/-0.02** | 降低权重 |
| 方向对齐 | 0.3 | **0.3×速度** | 速度加权 |
| 成功奖励 | 50.0 | **20.0** | 降低峰值 |
| 时间惩罚 | -0.005 | **-0.01** | 略微增加 |
| 裁剪范围 | [-10, 100] | **[-5, 30]** | 缩小范围 |

---

## 三、课程学习阈值调整

### 3.1 当前问题

```python
CurriculumStage(
    target_success_rate=0.8,  # 80%成功率要求过高
    min_episodes=100,         # 最小episodes较多
)
```

训练日志显示成功率卡在50-60%，无法晋升到下一阶段。

### 3.2 改进方案

```python
DEFAULT_STAGES = [
    CurriculumStage(
        name='Stage 0: Simple',
        num_sheep=3,
        num_herders=3,
        world_size=(60.0, 60.0),
        episode_length=80,           # 从50增加到80
        target_success_rate=0.8,     # 从0.8降到0.5
        min_episodes=50,             # 从100降到50
    ),
    CurriculumStage(
        name='Stage 1: Medium',
        num_sheep=5,
        num_herders=3,
        world_size=(60.0, 60.0),
        episode_length=100,          # 从75增加到100
        target_success_rate=0.7,    # 从0.7降到0.45
        min_episodes=100,            # 从200降到100
    ),
    CurriculumStage(
        name='Stage 2: Target',
        num_sheep=10,
        num_herders=3,
        world_size=(60.0, 60.0),
        episode_length=150,          # 从100增加到150
        target_success_rate=0.6,     # 从0.6降到0.4
        min_episodes=200,            # 从500降到200
    ),
]
```

### 3.3 阈值调整对比

| 阶段 | 参数 | 修改前 | 修改后 |
|------|------|--------|--------|
| Stage 0 | episode_length | 50 | **80** |
| Stage 0 | target_success_rate | 0.8 | **0.5** |
| Stage 0 | min_episodes | 100 | **50** |
| Stage 1 | episode_length | 75 | **100** |
| Stage 1 | target_success_rate | 0.7 | **0.45** |
| Stage 1 | min_episodes | 200 | **100** |
| Stage 2 | episode_length | 100 | **150** |
| Stage 2 | target_success_rate | 0.6 | **0.4** |
| Stage 2 | min_episodes | 500 | **200** |

---

## 四、探索策略增强

### 4.1 当前问题

```python
entropy_coef = 0.01  # 熵系数较小，探索不足
```

### 4.2 改进方案

在 `train_ppo.py` 中调整参数：

```python
# 默认参数调整
parser.add_argument('--entropy_coef', type=float, default=0.05)  # 从0.01提高到0.05
```

**保持简单**: 只调整熵系数，不添加复杂的衰减逻辑

### 4.3 探索策略对比

| 参数 | 修改前 | 修改后 |
|------|--------|--------|
| entropy_coef | 0.01 | **0.05** |

---

## 五、实施计划

### 阶段1: 观测空间改进
- [ ] 修改 `sheep_scenario.py` 中的 `get_observation()` 方法（12维→10维）
- [ ] 更新 `sheep_flock.py` 中的观测维度定义
- [ ] 更新可视化脚本适配新观测

### 阶段2: 奖励函数调整
- [ ] 修改 `sheep_flock.py` 中的 `_compute_reward()` 方法
- [ ] 调整奖励系数

### 阶段3: 课程学习调整
- [ ] 修改 `curriculum_env.py` 中的 `DEFAULT_STAGES`
- [ ] 调整各阶段参数

### 阶段4: 探索策略增强
- [ ] 修改 `train_ppo.py` 中的默认熵系数

### 阶段5: 验证测试
- [ ] 运行快速训练验证 (`scripts/train_fast.sh`)
- [ ] 分析训练曲线
- [ ] 可视化测试

---

## 六、预期效果

| 指标 | 修改前 | 预期修改后 |
|------|--------|------------|
| 观测维度 | 12 | **10** |
| 奖励波动 | [-132, +220] | [-5, +30] |
| 成功率 | 50-60% | 70-80% |
| 课程学习 | 卡在Stage 0 | 能晋升到Stage 2 |
| 训练稳定性 | 不稳定 | 稳定收敛 |

---

## 七、精简设计原则

1. **观测空间**: 10维足够，新增速度信息，删除冗余信息
2. **奖励函数**: 简化结构，降低系数，移除重复项
3. **课程学习**: 降低阈值，增加episode长度
4. **探索策略**: 只调熵系数，不添加复杂逻辑

---

## 八、风险与回退

**风险1**: 观测空间改动可能影响已训练模型
- 回退方案: 保留原观测空间代码，通过参数切换

**风险2**: 奖励系数调整可能影响学习速度
- 回退方案: 逐步调整，每次只改一个参数

**风险3**: 课程学习阈值降低可能导致策略过拟合简单场景
- 回退方案: 如果Stage 2表现差，提高阈值
