# Tasks

## 阶段1: 核心问题修复 (P0)

- [x] Task 1: 奖励函数改进
  - [x] SubTask 1.1: 实现势能奖励 (potential-based reward)
  - [x] SubTask 1.2: 统一奖励尺度，调整权重
  - [x] SubTask 1.3: 减少负奖励比例
  - [x] SubTask 1.4: 添加奖励裁剪

- [x] Task 2: 观测空间改进
  - [x] SubTask 2.1: 实现观测相对化/归一化
  - [x] SubTask 2.2: 添加羊群形状特征值 (λ1, λ2)
  - [x] SubTask 2.3: 添加机械狗站位统计信息
  - [x] SubTask 2.4: 更新观测空间维度

- [x] Task 3: 动作空间改进
  - [x] SubTask 3.1: 添加κ约束 (log_kappa clamp)
  - [x] SubTask 3.2: 实现角度相对化（以目标方向为参考）
  - [x] SubTask 3.3: 更新HighLevelAction类

- [x] Task 4: 课程学习实现
  - [x] SubTask 4.1: 实现CurriculumSheepFlockEnv
  - [x] SubTask 4.2: 设计3阶段课程
  - [x] SubTask 4.3: 实现阶段切换逻辑
  - [x] SubTask 4.4: 从简单场景开始训练验证

## 阶段2: 网络与超参数优化 (P1)

- [x] Task 5: 网络架构改进
  - [x] SubTask 5.1: 增大隐藏层到256
  - [x] SubTask 5.2: 增加网络层数到3层
  - [x] SubTask 5.3: 添加LayerNorm
  - [x] SubTask 5.4: 添加Dropout

- [x] Task 6: 超参数调优
  - [x] SubTask 6.1: 降低学习率到3e-4
  - [x] SubTask 6.2: 减少PPO epoch到10
  - [x] SubTask 6.3: 增加mini-batch到4
  - [x] SubTask 6.4: 降低梯度裁剪到0.5

- [x] Task 7: 奖励归一化
  - [x] SubTask 7.1: 实现RunningMeanStd类
  - [x] SubTask 7.2: 在训练循环中应用奖励归一化
  - [x] SubTask 7.3: 测试归一化效果

## 阶段3: 验证与调优 (P2)

- [ ] Task 8: 训练验证
  - [ ] SubTask 8.1: 在简单场景验证改进效果
  - [ ] SubTask 8.2: 记录奖励曲线对比
  - [ ] SubTask 8.3: 分析收敛速度

- [ ] Task 9: 性能评估
  - [ ] SubTask 9.1: 测试最终成功率
  - [ ] SubTask 9.2: 测试不同场景配置
  - [ ] SubTask 9.3: 生成评估报告

# Task Dependencies

- Task 2 depends on Task 1
- Task 3 depends on Task 1
- Task 4 depends on Task 1, Task 2, Task 3
- Task 5 depends on Task 4
- Task 6 depends on Task 4
- Task 7 depends on Task 4
- Task 8 depends on Task 5, Task 6, Task 7
- Task 9 depends on Task 8
