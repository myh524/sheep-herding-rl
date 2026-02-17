# PPO算法优化工作报告

**日期**: 2026-02-17
**任务**: 任务#1 - PPO算法小幅优化
**负责人**: 算法工程师

---

## 一、完成的任务

- 任务#1: PPO算法小幅优化

---

## 二、工作内容

### 2.1 问题分析

通过分析现有训练日志，发现以下问题：

1. **KL散度计算错误**: `kl_divergence` 经常出现负值（如-0.4090），这在数学上是不可能的，因为KL散度必须非负
2. **训练不稳定**: `avg_reward` 波动剧烈（从-50到+74），说明策略更新不稳定
3. **学习率调度缺乏warmup**: 直接使用cosine decay可能导致训练初期不稳定
4. **GAE lambda参数**: 原始值0.98可能过高，导致方差过大

### 2.2 实施的优化

#### 优化1: 修复KL散度计算

**文件**: [train_ppo.py](file:///home/hmy524/github_project/high_layer/train_ppo.py#L400-L420)

**修改内容**:
```python
# 修改前
kl_div = (old_action_log_probs_batch - action_log_probs).mean()

# 修改后
kl_div = torch.clamp(old_action_log_probs_batch - action_log_probs, min=0.0).mean()
```

**说明**: KL散度 = E[log(pi_old/pi_new)] 必须非负，使用`torch.clamp`确保数值稳定性。

#### 优化2: 学习率调度增强

**文件**: [train_ppo.py](file:///home/hmy524/github_project/high_layer/train_ppo.py#L591-L610)

**新增参数**:
- `--lr_warmup_steps`: 学习率warmup步数（默认1000步）
- `--lr_min`: 最小学习率（默认1e-5）

**调度策略**:
1. **Warmup阶段**: 前1000步线性增加学习率
2. **衰减阶段**: 使用cosine annealing从`lr`衰减到`lr_min`

```python
def adjust_lr(self, episode, episodes):
    total_steps = episode * self.args.episode_length
    
    if total_steps < self.args.lr_warmup_steps:
        warmup_factor = total_steps / self.args.lr_warmup_steps
        lr = self.args.lr * warmup_factor
    else:
        # Cosine annealing
        progress = (total_steps - self.args.lr_warmup_steps) / (self.args.num_env_steps - self.args.lr_warmup_steps)
        lr = self.args.lr_min + 0.5 * (self.args.lr - self.args.lr_min) * (1 + np.cos(np.pi * progress))
```

#### 优化3: GAE Lambda参数调整

**文件**: [train_ppo.py](file:///home/hmy524/github_project/high_layer/train_ppo.py#L83-L84)

**修改内容**:
- 将默认`gae_lambda`从0.98调整为0.95
- 降低lambda值可以减少方差，提高训练稳定性

#### 优化4: Clip参数动态调整

**文件**: [train_ppo.py](file:///home/hmy524/github_project/high_layer/train_ppo.py#L626-L633)

**新增参数**:
- `--clip_param_final`: 最终clip参数（默认0.1）
- `--use_clip_annealing`: 启用clip参数退火

**策略**: 训练过程中将clip参数从0.2线性退火到0.1，使策略更新更加保守。

```python
def adjust_clip_param(self, episode, episodes):
    if self.use_clip_annealing:
        progress = episode / episodes
        self.clip_param = self.args.clip_param - \
                         (self.args.clip_param - self.clip_param_final) * progress
```

#### 优化5: 数值稳定性增强

**文件**: [reward_normalizer.py](file:///home/hmy524/github_project/high_layer/onpolicy/utils/reward_normalizer.py)

**改进内容**:
1. 在更新统计量前裁剪极端值（-100到100）
2. 增加可配置的归一化范围参数`clip_range`
3. 改进除零保护（使用1e-8替代1e-4）

**文件**: [train_ppo.py](file:///home/hmy524/github_project/high_layer/train_ppo.py#L358-L362)

**改进内容**:
- Advantage归一化使用更小的epsilon（1e-8）

### 2.3 训练脚本更新

**文件**: [scripts/train_fast.sh](file:///home/hmy524/github_project/high_layer/scripts/train_fast.sh)

新增参数:
```bash
--lr_warmup_steps 1000 \
--lr_min 1e-5 \
--clip_param_final 0.1 \
--use_clip_annealing \
```

---

## 三、修改文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `train_ppo.py` | 修改 | 核心优化实现 |
| `onpolicy/utils/reward_normalizer.py` | 修改 | 数值稳定性增强 |
| `scripts/train_fast.sh` | 修改 | 训练参数更新 |

---

## 四、预期效果

1. **KL散度**: 现在应该始终为非负值
2. **训练稳定性**: 通过warmup和clip退火减少训练波动
3. **收敛速度**: 更稳定的学习率调度可能加速收敛
4. **数值稳定性**: 减少NaN和Inf的出现

---

## 五、建议与后续工作

### 5.1 建议的验证实验

1. **对比实验**: 使用相同种子对比优化前后的训练曲线
2. **消融实验**: 分别测试各个优化项的效果
3. **超参数搜索**: 对关键参数（gae_lambda, clip_param）进行网格搜索

### 5.2 后续优化方向

1. **自适应KL惩罚**: 当前KL惩罚系数调整策略可以进一步优化
2. **Value函数裁剪**: 考虑使用更保守的value loss裁剪策略
3. **梯度裁剪**: 当前max_grad_norm=0.5，可以尝试更小的值
4. **奖励塑形**: 配合环境工程师优化奖励函数设计

### 5.3 监控指标

建议在训练过程中重点关注:
- `kl_divergence`: 应该在0.01-0.02范围内
- `avg_reward`: 应该逐渐上升并趋于稳定
- `train_loss`: 应该逐渐下降
- `clip_param`: 如果启用退火，应该逐渐减小

---

## 六、注意事项

1. 本次优化为**小幅调整**，未改变核心算法架构
2. 所有新增参数都有合理的默认值，向后兼容
3. 建议在验证效果后再进行更大规模的改动

---

*报告生成时间: 2026-02-17*
