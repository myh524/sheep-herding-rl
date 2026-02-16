# 物理模型改进设计文档

## 问题分析

### 问题1：机械狗位置闪烁
- **根因**：每个step直接设置位置，无物理约束
- **后果**：位置不连续，羊的逃避力不连续，训练不稳定

### 问题2：每个step更新站位导致抖动
- **根因**：高层决策频率过高，机械狗来不及执行
- **后果**：机械狗不断改变方向，无法形成稳定策略

## 解决方案

### 方案：分层决策频率 + 运动模型

```
高层决策：每N步输出一次站位参数（N=5-10）
低层执行：每个step机械狗向目标位置移动（有速度限制）
羊的逃避力：基于机械狗当前位置（连续变化）
```

### 关键设计

1. **HerderEntity类**：机械狗实体，有位置、速度、目标位置
2. **决策频率控制**：高层每N步输出一次，中间步骤保持目标不变
3. **运动模型**：机械狗有最大速度，向目标位置平滑移动
4. **逃避力连续性**：羊基于机械狗当前位置计算逃避力

## 参数设计

### 速度参数
| 实体 | 参数 | 值 | 说明 |
|------|------|-----|------|
| 羊 | max_speed | 2.0 | 每秒最多移动2单位 |
| 羊 | max_force | 0.2 | 转向力 |
| 机械狗 | max_speed | 3.0 | 约为羊的1.5倍 |
| 机械狗 | max_accel | 1.0 | 加速度限制 |

### 决策频率
| 参数 | 值 | 说明 |
|------|-----|------|
| high_level_interval | 5 | 每5步高层输出一次站位 |
| dt | 0.1 | 物理步长 |

### 时间计算
- 世界大小：30×30
- 羊从一端到另一端：30/2.0 = 15秒 = 150步
- 机械狗从一端到另一端：30/3.0 = 10秒 = 100步
- 高层决策间隔：5步 = 0.5秒

## 代码实现

### HerderEntity类

```python
class HerderEntity:
    """Herder entity with physical constraints"""
    
    def __init__(self, position, max_speed=3.0, max_accel=1.0):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.target_position = None
        self.max_speed = max_speed
        self.max_accel = max_accel
    
    def set_target(self, target_position):
        """Set target position (from high-level policy)"""
        self.target_position = np.array(target_position, dtype=np.float32)
    
    def update(self, dt=0.1):
        """Update position towards target with physical constraints"""
        if self.target_position is None:
            return
        
        direction = self.target_position - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:
            self.velocity *= 0.5  # Damping near target
            return
        
        # Calculate desired velocity
        desired_speed = min(self.max_speed, distance / dt)
        desired_velocity = direction / distance * desired_speed
        
        # Apply acceleration limit
        accel = desired_velocity - self.velocity
        accel_mag = np.linalg.norm(accel)
        if accel_mag > self.max_accel * dt:
            accel = accel / accel_mag * self.max_accel * dt
        
        # Update velocity and position
        self.velocity += accel
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
        
        self.position += self.velocity * dt
```

### 决策频率控制

```python
class SheepFlockEnv:
    def __init__(self, ...):
        self.high_level_interval = 5  # High-level decision interval
        self.step_count = 0
        self.current_target_positions = None
    
    def step(self, actions):
        self.step_count += 1
        
        # Only update target positions every N steps
        if self.step_count % self.high_level_interval == 1:
            self.current_target_positions = self._sample_herder_positions(actions)
            for i, herder in enumerate(self.herders):
                herder.set_target(self.current_target_positions[i])
        
        # Update herder positions with physical model
        for herder in self.herders:
            herder.update(self.dt)
        
        # Update sheep with continuous evasion force
        self.scenario.update_sheep(self.dt)
        
        ...
```

## 训练稳定性分析

### 优点
1. **位置连续**：机械狗位置平滑变化，无闪烁
2. **逃避力连续**：羊的逃避力基于当前位置，连续变化
3. **决策稳定**：高层决策有足够时间执行
4. **更真实**：符合真实物理约束

### 潜在问题
1. **延迟**：高层决策有执行延迟
2. **非即时响应**：机械狗不能立即到达目标位置

### 缓解措施
1. **奖励设计**：考虑执行延迟，使用势能奖励
2. **观测设计**：包含机械狗当前位置和速度
3. **课程学习**：从简单场景开始，让策略适应延迟

## 实施计划

1. 创建 `HerderEntity` 类
2. 修改 `SheepScenario` 使用 `HerderEntity`
3. 修改 `SheepFlockEnv` 添加决策频率控制
4. 调整速度参数
5. 更新可视化验证效果
