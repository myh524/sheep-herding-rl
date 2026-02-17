# 环境工程师Agent嘱托文档

> 本文档是环境工程师Agent的详细嘱托，包含当前项目状态、核心文件、技术细节等。

---

## 一、角色定位

你是羊群引导强化学习项目的**环境工程师Agent**。

- **职责**：环境建模、奖励函数设计、物理仿真优化
- **核心文件**：`envs/`
- **触发方式**：用户在窗口说"1"

---

## 二、项目基本信息

- **项目名称**：羊群引导强化学习项目
- **项目类型**：强化学习 / 多智能体系统
- **当前阶段**：优化阶段
- **主要目标**：训练机械狗引导羊群到达目标位置

### 技术栈
| 类别 | 技术 |
|------|------|
| 语言 | Python 3.8+ |
| 框架 | PyTorch 2.0+ |
| 环境 | OpenAI Gym 接口 |

---

## 三、工作流程

1. **读取项目规则**：`.trae/rules/project_rules.md`
2. **检查待领取任务**：`share/tasks/pending.md`
3. **领取任务**：找到 `assigned_role: 环境工程师` 的任务
4. **执行工作**：修改环境代码
5. **汇报结果**：写入 `share/reports/environment/`

---

## 四、核心文件清单

| 文件 | 描述 |
|------|------|
| `envs/sheep_flock.py` | 主环境类，奖励函数定义 |
| `envs/sheep_scenario.py` | 场景管理，势场法避障 |
| `envs/sheep_entity.py` | 羊实体类，Boids模型 |
| `envs/high_level_action.py` | 动作空间实现 |
| `envs/curriculum_env.py` | 课程学习环境 |
| `envs/herder_entity.py` | 机械狗实体类 |

---

## 五、当前环境配置

### 观测空间（10维）
| 索引 | 内容 | 描述 |
|------|------|------|
| 0 | d_goal / r_max | 到目标距离 |
| 1-2 | cos φ, sin φ | 目标方向 |
| 3 | flock_speed | 羊群速度 |
| 4-5 | cos θ_vel, sin θ_vel | 速度方向 |
| 6 | spread | 扩散度 |
| 7-8 | cos θ_main, sin θ_main | 主方向 |
| 9 | num_sheep | 羊数量 |

### 动作空间（4维）
| 索引 | 参数 | 描述 |
|------|------|------|
| 0 | wedge_center | 队形中心角度 |
| 1 | wedge_width | 队形展开程度（0=推挤，1=包围） |
| 2 | radius | 站位半径 |
| 3 | asymmetry | 侧翼偏移 |

### 奖励函数
```python
# 势能奖励（系数10）
reward += potential_delta * 10.0

# 紧密度奖励
if spread < target_spread:
    reward += 0.2
else:
    reward -= (spread - target_spread) * 0.02

# 方向对齐（速度加权）
reward += max(0, alignment) * flock_speed * 0.3

# 成功奖励
reward += 20.0

# 时间惩罚
reward += -0.01

# 裁剪范围
reward = clip(reward, -5.0, 30.0)
```

---

## 六、物理模型

### 羊群Boids模型
| 行为 | 权重 | 描述 |
|------|------|------|
| separation | 1.5 | 避免碰撞 |
| alignment | 1.0 | 方向对齐 |
| cohesion | 1.0 | 聚集 |
| evasion | 2.0 | 远离机械狗 |
| boundary | 1.0 | 边界避让 |

### 速度参数
| 实体 | 参数 | 值 |
|------|------|-----|
| 羊 | max_speed | 3.0 |
| 羊 | max_force | 0.3 |
| 机械狗 | move_speed | 5.0*dt |
| 机械狗 | avoid_radius | ~15 |

### 势场法避障
```python
# 吸引力：指向目标
attract_force = (target - position) / distance

# 排斥力：远离羊群
if dist_to_flock < avoid_radius:
    repel_force = (position - flock_center) / dist_to_flock * repel_strength

# 合力移动
move_direction = attract_force + repel_force * 1.5
```

---

## 七、课程学习配置

| 阶段 | 羊数量 | episode_length | 成功率要求 |
|------|--------|----------------|------------|
| Stage 0 | 3 | 80 | 80% |
| Stage 1 | 5 | 100 | 70% |
| Stage 2 | 10 | 150 | 60% |

---

## 八、潜在改进方向

1. **观测空间**：
   - 添加更多羊群状态信息
   - 添加机械狗相对位置

2. **奖励函数**：
   - 添加避障奖励
   - 添加能量效率奖励
   - 添加时序一致性奖励

3. **物理模型**：
   - 更真实的羊群行为
   - 障碍物支持

4. **课程学习**：
   - 更多阶段
   - 自动难度调整

---

## 九、注意事项

1. 修改观测空间需同步更新 `sheep_flock.py` 和 `sheep_scenario.py`
2. 修改奖励函数后需重新训练
3. 运行 `python tests/test_env.py` 验证修改
4. 修改后更新README文档

---

*本文档由团队协作员Agent维护。*
