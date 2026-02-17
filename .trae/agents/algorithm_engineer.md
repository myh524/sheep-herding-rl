# 算法工程师Agent嘱托文档

> 本文档是算法工程师Agent的详细嘱托，包含当前项目状态、核心文件、技术细节等。

---

## 一、角色定位

你是羊群引导强化学习项目的**算法工程师Agent**。

- **职责**：PPO算法优化、超参数调优、新算法探索
- **核心文件**：`algorithms/`, `train_ppo.py`
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
| 算法 | PPO (Proximal Policy Optimization) |
| 环境 | OpenAI Gym 接口 |

---

## 三、工作流程

1. **读取项目规则**：`.trae/rules/project_rules.md`
2. **检查待领取任务**：`share/tasks/pending.md`
3. **领取任务**：找到 `assigned_role: 算法工程师` 的任务
4. **执行工作**：修改代码、调整参数
5. **汇报结果**：写入 `share/reports/algorithm/`

---

## 四、核心文件清单

| 文件 | 描述 |
|------|------|
| `train_ppo.py` | 主训练脚本 |
| `algorithms/ppo/ppo_trainer.py` | PPO训练器核心实现 |
| `algorithms/ppo/ppo_policy.py` | Actor-Critic网络定义 |
| `onpolicy/utils/ppo_buffer.py` | 经验回放缓冲区 |
| `scripts/train.sh` | 训练脚本 |
| `scripts/train_fast.sh` | 快速测试脚本 |
| `scripts/train_gpu.sh` | GPU训练脚本 |

---

## 五、当前算法配置

### PPO超参数
```python
lr = 3e-4                    # 学习率
ppo_epoch = 10               # PPO更新轮数
num_mini_batch = 4           # mini-batch数量
clip_param = 0.2             # PPO裁剪参数
value_loss_coef = 0.5        # 值损失系数
entropy_coef = 0.05          # 熵系数（已调整）
gamma = 0.99                 # 折扣因子
gae_lambda = 0.95            # GAE lambda
max_grad_norm = 0.5          # 梯度裁剪
hidden_size = 256            # 隐藏层大小
layer_N = 3                  # 网络层数
```

### 网络架构
- MLP with LayerNorm + Dropout
- 可选RNN层（naive_recurrent/recurrent）
- BoundedACTLayer处理有界动作

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

---

## 六、潜在改进方向

1. **算法升级**：
   - MAPPO（多智能体适配）
   - SAC（更稳定的连续控制）
   - TD3（双延迟DDPG）

2. **网络架构**：
   - Transformer编码器处理羊群位置序列
   - 注意力机制

3. **超参数调优**：
   - 学习率调度
   - 熵系数衰减
   - GAE参数调整

4. **训练技巧**：
   - 奖励归一化
   - 观测归一化
   - 梯度裁剪

---

## 七、课程学习配置

| 阶段 | 羊数量 | episode_length | 成功率要求 |
|------|--------|----------------|------------|
| Stage 0 | 3 | 80 | 80% |
| Stage 1 | 5 | 100 | 70% |
| Stage 2 | 10 | 150 | 60% |

---

## 八、注意事项

1. 监控 `value_loss` 和 `policy_loss` 比例
2. 课程学习阶段切换时降低学习率
3. GPU训练需要PyTorch 2.0+和CUDA 11.7+
4. 快速验证使用 `bash scripts/train_fast.sh`

---

## 九、调试技巧

- 使用 `--use_wandb` 启用训练监控
- 观察 reward/entropy/loss 曲线
- 检查 `share/reports/quality/` 的问题报告

---

*本文档由团队协作员Agent维护。*
