# 训练不收敛根因分析报告 v2

## 执行摘要

本报告基于最新训练日志和代码实现，对羊群引导强化学习项目的训练不收敛问题进行了深入诊断。通过分析训练日志、环境设置、奖励函数、算法实现等多个维度，发现了**KL散度计算错误**这一关键问题，以及其他多个导致训练不稳定的因素。

**关键发现**：KL散度计算出现负值，这是数学上不可能的，表明算法实现存在严重错误。

---

## 一、问题现象总结

### 1.1 训练日志分析

从最新训练日志（2026-02-17 19:09:45）观察到：

| Episode | Avg Reward | Train Loss | KL Divergence | KL Coef | 状态 |
|---------|------------|------------|---------------|---------|------|
| 0 | -35.49 | 54.71 | **-0.0483** | 0.20 | 初始 |
| 50 | 6.62 | 3.81 | **-0.0787** | 1.01 | 改善 |
| 100 | 35.06 | 2.72 | **-0.0992** | 1.01 | 改善 |
| 150 | 45.66 | 4.36 | **-0.0452** | 2.00 | 峰值 |
| 200 | **-52.50** | 7.07 | **-0.1093** | 2.00 | 崩溃 |
| 250 | -52.26 | 12.65 | **-0.0457** | 2.00 | 低谷 |
| 300 | -51.07 | 2.47 | **-0.1666** | 2.00 | 低谷 |
| 400 | 74.65 | 13.03 | **-0.2083** | 2.00 | 反弹 |
| 450 | -47.57 | 11.97 | **+0.0154** | 0.89 | 波动 |
| 1000 | 56.79 | 10.81 | **-0.1575** | 2.00 | 波动 |

### 1.2 关键异常指标

1. **KL散度为负值**：所有日志中KL散度几乎都是负数，这是数学上不可能的
2. **奖励剧烈波动**：从45.66到-52.50再到74.65，波动幅度超过120
3. **KL系数达到上限**：kl_coef持续保持在2.0（最大值），说明KL惩罚机制失效
4. **训练损失不稳定**：从2.72到18.28再到0.47，波动极大

---

## 二、根因分析

### 2.1 【严重】KL散度计算错误

**问题位置**：[train_ppo.py:427](file:///home/hmy524/github_project/high_layer/train_ppo.py#L427)

```python
# 当前实现（错误）
kl_div = torch.clamp(old_action_log_probs_batch - action_log_probs, min=0.0).mean()
```

**问题分析**：

1. **数学错误**：KL散度的正确定义是 `KL(P||Q) = E_P[log(P/Q)]`，应该始终非负
2. **实现错误**：当前实现使用 `clamp(min=0.0)` 强制将负值截断为0，但这掩盖了计算错误
3. **日志显示负值**：训练日志中KL散度显示为负值（如-0.1093），说明clamp之前的值就是负的

**正确实现应该是**：

```python
# 正确的KL散度计算
# KL(P_old || P_new) = E[log(pi_old) - log(pi_new)]
# 由于log_prob = log(pi(a|s))，所以：
# KL = E[log_prob_old - log_prob_new]
# 这应该始终 >= 0

# 方法1：直接计算（推荐）
kl_div = (old_action_log_probs_batch - action_log_probs).mean()
# 如果出现负值，说明策略更新有问题，不应该clamp

# 方法2：使用torch.distributions
# 这是最可靠的方法
kl_div = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()
```

**影响**：
- KL惩罚机制完全失效
- 策略可能更新过大，导致训练不稳定
- 自适应KL系数调整逻辑被误导

**优先级**：**P0 - 立即修复**

---

### 2.2 【严重】奖励函数波动性过大

**问题位置**：[envs/sheep_flock.py:249-387](file:///home/hmy524/github_project/high_layer/envs/sheep_flock.py#L249)

**问题分析**：

1. **距离奖励波动**：
   ```python
   distance_reward = np.exp(-3.0 * distance_ratio)
   # 当distance_ratio从0.2变到0.8时，奖励从0.55变到0.09，变化幅度大
   ```

2. **进度奖励不稳定**：
   ```python
   progress_reward = np.clip(progress_ratio, -1.0, 1.0) * 0.3
   # progress_ratio可能剧烈变化，导致奖励跳变
   ```

3. **成功奖励过大**：
   ```python
   if self.scenario.is_flock_at_target(threshold=5.0):
       reward += 5.0  # 加上其他奖励，可能达到7-8
   ```

4. **奖励范围**：`[-1.0, 10.0]`，跨度11，对PPO来说仍然偏大

**影响**：
- 奖励信号不稳定，模型难以学习一致策略
- 价值函数估计困难，导致训练损失波动

**优先级**：**P0 - 需要重构**

---

### 2.3 【高】KL惩罚系数调整机制问题

**问题位置**：[train_ppo.py:605-612](file:///home/hmy524/github_project/high_layer/train_ppo.py#L605)

```python
def adjust_kl_coef(self):
    if self.use_kl_penalty and self.kl_divergence > 0:
        if self.kl_divergence > self.target_kl * 1.5:
            self.kl_coef = min(self.kl_coef * self.kl_coef_multiplier, self.kl_coef_max)
        elif self.kl_divergence < self.target_kl * 0.5:
            self.kl_coef = max(self.kl_coef / self.kl_coef_multiplier, self.kl_coef_min)
```

**问题分析**：

1. **条件判断错误**：`if self.kl_divergence > 0` 在KL散度为负时不会执行
2. **系数持续增大**：由于KL散度计算错误，系数一直增加到最大值2.0
3. **调整逻辑不合理**：没有考虑KL散度为负的情况

**影响**：
- KL惩罚机制失效
- 策略更新不受约束，可能导致训练崩溃

**优先级**：**P1 - 需要修复**

---

### 2.4 【高】观测空间设计问题

**问题位置**：[envs/sheep_scenario.py:419-479](file:///home/hmy524/github_project/high_layer/envs/sheep_scenario.py#L419)

**问题分析**：

1. **观测维度不匹配**：
   - 注释说是10维，但实际计算了9个值（obs[0]到obs[8]）
   - obs[9]（羊群数量）似乎没有计算

2. **观测范围不一致**：
   ```python
   obs_low = np.array([0.0, -1.0, -1.0, 0.0, -1.0, -1.0, 0.0, -1.0, -1.0, 0.0])
   obs_high = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 2.0])
   ```
   - obs[0]范围[0, 10]，但实际值应该<=1（归一化距离）
   - obs[6]范围[0, 10]，但实际值应该<=1（归一化扩散度）

3. **观测稳定性问题**：
   - 方向角使用cos/sin编码，但可能存在数值不稳定
   - 特征值计算可能产生NaN

**影响**：
- 神经网络输入不稳定
- 可能导致梯度爆炸或消失

**优先级**：**P1 - 需要修复**

---

### 2.5 【中】动作空间解码问题

**问题位置**：[envs/high_level_action.py:58-99](file:///home/hmy524/github_project/high_layer/envs/high_level_action.py#L58)

**问题分析**：

1. **动作映射复杂**：
   - 4维动作映射到3个机械狗的6个坐标
   - 映射过程非线性，可能导致梯度消失

2. **动作约束不足**：
   ```python
   raw_action = np.clip(raw_action, -1.0, 1.0)
   ```
   - 只在解码时clip，训练时可能产生超出范围的值

3. **机械狗位置冲突**：
   - [sheep_flock.py:233-239](file:///home/hmy524/github_project/high_layer/envs/sheep_flock.py#L233)中的位置修正可能导致动作失效

**影响**：
- 策略学习困难
- 动作空间探索效率低

**优先级**：**P2 - 建议优化**

---

### 2.6 【中】环境参数设置问题

**问题位置**：[scripts/train_fast.sh](file:///home/hmy524/github_project/high_layer/scripts/train_fast.sh)

**问题分析**：

1. **羊群与机械狗比例**：6羊:3狗 = 2:1，比例合理
2. **世界大小**：30x30，对于6只羊来说偏小
3. **Episode长度**：100步，可能不足以完成任务
4. **机械狗速度**：`5.0 * dt`，可能过快导致羊群被冲散

**影响**：
- 任务难度可能过高
- 成功率低，学习信号弱

**优先级**：**P2 - 建议调整**

---

### 2.7 【中】超参数配置问题

**问题分析**：

1. **熵系数衰减过快**：
   - 初始0.05，最终0.001
   - 1000个episode后降至0.045，衰减速度合理

2. **学习率固定**：
   - 使用余弦退火，但前1000个episode几乎不变
   - 可能需要更激进的衰减

3. **PPO clip参数**：
   - 0.2是标准值，但结合KL惩罚可能导致过度约束

**影响**：
- 探索与利用平衡可能不当
- 训练效率可能受影响

**优先级**：**P2 - 建议调优**

---

## 三、问题关联分析

### 3.1 问题因果链

```
KL散度计算错误（根因）
    ↓
KL惩罚机制失效
    ↓
策略更新过大
    ↓
奖励剧烈波动
    ↓
价值函数估计困难
    ↓
训练损失不稳定
    ↓
训练不收敛
```

### 3.2 问题优先级排序

| 优先级 | 问题 | 影响 | 修复难度 |
|--------|------|------|----------|
| P0 | KL散度计算错误 | 严重 | 低 |
| P0 | 奖励函数波动性过大 | 严重 | 中 |
| P1 | KL惩罚系数调整机制 | 高 | 低 |
| P1 | 观测空间设计问题 | 高 | 中 |
| P2 | 动作空间解码问题 | 中 | 中 |
| P2 | 环境参数设置 | 中 | 低 |
| P2 | 超参数配置 | 中 | 低 |

---

## 四、改进建议

### 4.1 立即修复（P0）

#### 4.1.1 修复KL散度计算

**文件**：[train_ppo.py](file:///home/hmy524/github_project/high_layer/train_ppo.py)

```python
# 修改前（错误）
kl_div = torch.clamp(old_action_log_probs_batch - action_log_probs, min=0.0).mean()

# 修改后（正确）
# KL散度 = E[log(pi_old) - log(pi_new)]
# 注意：ratio = exp(log_prob_new - log_prob_old)
# 所以 log_prob_old - log_prob_new = -log(ratio)
kl_div = (old_action_log_probs_batch - action_log_probs).mean()

# 添加安全检查
if kl_div < 0:
    # 这不应该发生，如果发生说明有bug
    print(f"WARNING: Negative KL divergence detected: {kl_div.item()}")
    print(f"  old_log_probs mean: {old_action_log_probs_batch.mean().item()}")
    print(f"  new_log_probs mean: {action_log_probs.mean().item()}")
    kl_div = torch.abs(kl_div)  # 临时修复，但需要调查原因
```

#### 4.1.2 重构奖励函数

**文件**：[envs/sheep_flock.py](file:///home/hmy524/github_project/high_layer/envs/sheep_flock.py)

```python
def _compute_reward(self) -> float:
    """
    简化且稳定的奖励函数
    
    设计原则：
    1. 奖励范围：[-1.0, 3.0]
    2. 主要信号：距离改善
    3. 辅助信号：队形保持
    4. 成功奖励：适中，不压倒其他信号
    """
    reward = 0.0
    
    # 1. 距离奖励（主要信号）
    current_distance = self.scenario.get_distance_to_target()
    max_distance = np.linalg.norm(self.world_size)
    distance_ratio = current_distance / max_distance
    
    # 使用更平滑的奖励曲线
    distance_reward = 1.0 - distance_ratio  # [0, 1]
    reward += distance_reward * 0.5
    
    # 2. 进度奖励（塑形信号）
    if self.prev_distance is not None:
        progress = self.prev_distance - current_distance
        # 归一化进度：假设最大进度为5.0
        progress_normalized = np.clip(progress / 5.0, -1.0, 1.0)
        reward += progress_normalized * 0.3
    
    self.prev_distance = current_distance
    
    # 3. 队形奖励（辅助信号）
    spread = self.scenario.get_flock_spread()
    target_spread = 4.0
    spread_error = abs(spread - target_spread) / target_spread
    formation_reward = np.exp(-spread_error)  # [0, 1]
    reward += formation_reward * 0.1
    
    # 4. 成功奖励
    if self.scenario.is_flock_at_target(threshold=5.0):
        reward += 1.0  # 适中奖励，不压倒其他信号
    
    # 5. 时间惩罚（轻微）
    reward -= 0.01
    
    # 限制范围
    reward = np.clip(reward, -1.0, 3.0)
    
    return float(reward)
```

### 4.2 高优先级修复（P1）

#### 4.2.1 修复KL惩罚系数调整

**文件**：[train_ppo.py](file:///home/hmy524/github_project/high_layer/train_ppo.py)

```python
def adjust_kl_coef(self):
    """修复KL惩罚系数调整逻辑"""
    if not self.use_kl_penalty:
        return
    
    kl_div = abs(self.kl_divergence)  # 使用绝对值
    
    if kl_div > self.target_kl * 1.5:
        # KL散度过大，增加惩罚
        self.kl_coef = min(self.kl_coef * self.kl_coef_multiplier, self.kl_coef_max)
        print(f"KL too high ({kl_div:.4f}), increasing coef to {self.kl_coef:.4f}")
    elif kl_div < self.target_kl * 0.5:
        # KL散度过小，减少惩罚
        self.kl_coef = max(self.kl_coef / self.kl_coef_multiplier, self.kl_coef_min)
        print(f"KL too low ({kl_div:.4f}), decreasing coef to {self.kl_coef:.4f}")
```

#### 4.2.2 修复观测空间

**文件**：[envs/sheep_scenario.py](file:///home/hmy524/github_project/high_layer/envs/sheep_scenario.py)

```python
def get_observation(self) -> np.ndarray:
    """修复观测空间计算"""
    obs = np.zeros(10, dtype=np.float32)
    
    # ... 前面的计算保持不变 ...
    
    # 修复obs[9]：羊群数量归一化
    obs[9] = len(self.sheep) / 30.0
    
    # 确保所有观测值在合理范围内
    obs = np.clip(obs, -1.0, 1.0)  # 统一范围
    
    # 安全检查
    if np.isnan(obs).any() or np.isinf(obs).any():
        print(f"WARNING: Invalid observation detected: {obs}")
        obs = np.zeros(10, dtype=np.float32)
    
    return obs
```

### 4.3 中优先级优化（P2）

#### 4.3.1 优化环境参数

**建议调整**：
- 增加episode长度：100 → 150
- 增加世界大小：30x30 → 40x40
- 降低机械狗速度：5.0 → 3.0

#### 4.3.2 调整超参数

**建议调整**：
- 增加初始熵系数：0.05 → 0.1
- 减小PPO clip：0.2 → 0.15
- 增加批量大小：如果内存允许

---

## 五、诊断日志建议

### 5.1 训练过程日志

```python
# 建议在train_ppo.py中添加
log.info(f"Episode {episode}: "
         f"reward={avg_reward:.2f}, "
         f"loss={train_loss:.2f}, "
         f"kl_div={self.kl_divergence:.4f}, "  # 注意负值
         f"kl_coef={self.kl_coef:.4f}, "
         f"ratio_mean={ratio.mean().item():.4f}, "
         f"ratio_max={ratio.max().item():.4f}, "
         f"ratio_min={ratio.min().item():.4f}, "
         f"entropy={dist_entropy.item():.4f}, "
         f"value_mean={values.mean().item():.4f}")
```

### 5.2 环境状态日志

```python
# 建议在sheep_flock.py中添加
log.debug(f"Step {self.current_step}: "
          f"distance={current_distance:.2f}, "
          f"spread={spread:.2f}, "
          f"reward={reward:.2f}, "
          f"reward_dist={distance_reward:.2f}, "
          f"reward_progress={progress_reward:.2f}, "
          f"reward_formation={formation_reward:.2f}")
```

### 5.3 KL散度详细日志

```python
# 建议在计算KL散度时添加
if self.kl_divergence < 0:
    log.error(f"NEGATIVE KL DIVERGENCE DETECTED!")
    log.error(f"  old_log_probs: mean={old_action_log_probs_batch.mean():.4f}, "
              f"std={old_action_log_probs_batch.std():.4f}")
    log.error(f"  new_log_probs: mean={action_log_probs.mean():.4f}, "
              f"std={action_log_probs.std():.4f}")
    log.error(f"  ratio: mean={ratio.mean():.4f}, "
              f"max={ratio.max():.4f}, min={ratio.min():.4f}")
```

---

## 六、验证方法

### 6.1 短期验证（修复后立即测试）

1. **运行100个episode**，观察KL散度是否为正值
2. **检查奖励曲线**，观察波动是否减小
3. **监控KL系数**，观察是否正常调整

### 6.2 中期验证（500-1000 episode）

1. **观察训练损失**，应该逐渐下降并稳定
2. **检查成功率**，应该逐渐上升
3. **分析奖励分布**，应该更加集中

### 6.3 长期验证（完整训练）

1. **多种子实验**：运行3-5个不同种子
2. **性能对比**：与修复前对比
3. **泛化测试**：在不同场景下测试

---

## 七、总结

### 7.1 核心问题

**KL散度计算错误是导致训练不收敛的根本原因**。这个错误导致：
1. KL惩罚机制完全失效
2. 策略更新不受约束
3. 训练过程剧烈波动

### 7.2 次要问题

1. 奖励函数波动性过大
2. 观测空间设计不一致
3. 环境参数可能需要调整

### 7.3 修复优先级

1. **立即修复**：KL散度计算（1小时）
2. **高优先级**：奖励函数重构（2-3小时）
3. **中优先级**：观测空间修复（1小时）
4. **后续优化**：超参数调优（持续）

### 7.4 预期效果

修复后预期：
- KL散度为正值且在合理范围内（0.01-0.05）
- 奖励波动减小50%以上
- 训练损失稳定下降
- 成功率逐步提升

---

**报告生成时间**：2026-02-17 23:00  
**分析数据来源**：训练日志、源代码审查  
**下一步行动**：立即修复KL散度计算错误
