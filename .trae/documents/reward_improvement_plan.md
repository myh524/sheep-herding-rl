# 奖励函数改进计划：羊群到达目标后保持控制

## 问题分析

### 当前行为
- 羊群到达目标位置后，episode立即结束（`done=True`）
- 策略学会：到达目标后狗就远离，不再控制羊群
- 没有奖励鼓励"保持控制"的行为

### 根本原因
1. **成功即结束**: `is_flock_at_target()` 返回True时episode结束
2. **无保持奖励**: 没有奖励鼓励狗在目标位置附近保持包围态势
3. **时间惩罚**: 每步-0.01的惩罚鼓励尽快结束

---

## 改进方案（推荐）

### 核心思想
要求羊群到达目标后**保持控制N步**才算真正成功，并添加**包围质量奖励**

### 具体修改

#### 1. 新增奖励项

| 奖励项 | 公式 | 说明 |
|--------|------|------|
| `surround_reward` | `surround_quality × surround_bonus` | 包围质量奖励 |
| `hold_bonus` | `hold_bonus × (steps_at_target / required_hold_steps)` | 保持奖励 |

#### 2. 修改成功条件

```
之前: distance < threshold → done = True
之后: distance < threshold 且 steps_at_target >= required_hold_steps → done = True
```

#### 3. 新增 `_compute_surround_quality()` 函数

计算狗的包围质量（0.0 - 1.0）：
- 1.0: 完美包围（均匀分布在羊群周围）
- 0.0: 没有包围（所有狗在同一侧或远离）

---

## 代码修改清单

### 文件: `envs/sheep_flock.py`

1. **`__init__()`**: 添加 `self.steps_at_target = 0`
2. **`reset()`**: 初始化 `self.steps_at_target = 0`
3. **`_compute_reward()`**: 
   - 添加包围质量奖励
   - 添加保持奖励
   - 修改成功奖励逻辑
   - 成功后取消时间惩罚
4. **`_check_done()`**: 修改结束条件
5. **新增 `_compute_surround_quality()`**: 计算包围质量

### 新增默认配置

```python
self.reward_config = reward_config or {
    'distance_weight': 2.0,
    'progress_weight': 3.0,
    'spread_penalty_weight': 0.1,
    'success_bonus': 10.0,
    'quick_success_bonus': 5.0,
    'time_penalty': 0.01,
    # 新增
    'surround_bonus': 1.0,        # 包围奖励权重
    'hold_bonus': 0.5,            # 保持奖励（每步）
    'required_hold_steps': 10,    # 需要保持的步数
    'success_threshold': 5.0,     # 成功距离阈值
}
```

---

## 预期效果

| 之前 | 之后 |
|------|------|
| 羊群到达目标后狗就远离 | 羊群到达目标后狗保持包围 |
| 成功即结束 | 需要保持10步才算成功 |
| 无包围奖励 | 包围质量越高奖励越多 |
| 每步时间惩罚 | 成功后无时间惩罚 |

---

## 实施步骤

1. [ ] 修改 `__init__()` 添加 `steps_at_target` 初始化
2. [ ] 修改 `reset()` 重置 `steps_at_target`
3. [ ] 新增 `_compute_surround_quality()` 函数
4. [ ] 修改 `_compute_reward()` 函数
5. [ ] 修改 `_check_done()` 函数
6. [ ] 更新默认奖励配置
7. [ ] 测试验证
