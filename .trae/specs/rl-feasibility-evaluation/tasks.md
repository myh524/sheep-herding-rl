# Tasks

## 阶段1: 核心设计改进 (1周)

- [ ] Task 1: 实现观测空间相对化/归一化
  - [ ] SubTask 1.1: 实现 get_high_state() 方法
  - [ ] SubTask 1.2: 计算羊群形状特征值 λ1, λ2
  - [ ] SubTask 1.3: 实现角度相对化（以目标方向为0°）
  - [ ] SubTask 1.4: 更新 SheepScenario 使用新观测设计

- [ ] Task 2: 实现动作空间改进
  - [ ] SubTask 2.1: 创建 HighLevelAction 类
  - [ ] SubTask 2.2: 实现 decode_action() 方法
  - [ ] SubTask 2.3: 实现 κ 约束（log_kappa clamp）
  - [ ] SubTask 2.4: 创建 KappaScheduler 类
  - [ ] SubTask 2.5: 实现角度相对化采样

- [ ] Task 3: 修复已知代码问题
  - [ ] SubTask 3.1: 修复 TanhNormal 数值稳定性问题 (bounded_act.py)
  - [ ] SubTask 3.2: 修复 ValueNorm 设备不一致问题 (valuenorm.py)
  - [ ] SubTask 3.3: 修复 check() 函数返回值问题 (util.py)
  - [ ] SubTask 3.4: 修复 PopArt 参数更新问题 (popart.py)

## 阶段2: 奖励函数与训练策略 (1周)

- [ ] Task 4: 实现改进的奖励函数
  - [ ] SubTask 4.1: 添加势能奖励 (potential-based reward)
  - [ ] SubTask 4.2: 添加站位参数质量奖励
  - [ ] SubTask 4.3: 添加速度对齐奖励
  - [ ] SubTask 4.4: 实现奖励归一化

- [ ] Task 5: 简单场景验证
  - [ ] SubTask 5.1: 配置简单场景 (3羊2机械狗)
  - [ ] SubTask 5.2: 运行基线训练实验
  - [ ] SubTask 5.3: 验证新观测/动作设计效果
  - [ ] SubTask 5.4: 记录基线性能指标

## 阶段3: 课程学习与泛化性训练 (1-2周)

- [ ] Task 6: 实现课程学习框架
  - [ ] SubTask 6.1: 设计课程阶段配置
  - [ ] SubTask 6.2: 实现阶段切换逻辑
  - [ ] SubTask 6.3: 实现性能监控
  - [ ] SubTask 6.4: 添加课程学习日志

- [ ] Task 7: 实现泛化性训练策略
  - [ ] SubTask 7.1: 实现配置随机化类 GeneralizedTrainingConfig
  - [ ] SubTask 7.2: 实现Episode级配置随机化
  - [ ] SubTask 7.3: 实现渐进式扩展策略

- [ ] Task 8: 课程学习训练
  - [ ] SubTask 8.1: 阶段1训练 (3羊2机械狗)
  - [ ] SubTask 8.2: 阶段2训练 (5羊2机械狗)
  - [ ] SubTask 8.3: 阶段3训练 (10羊3机械狗)
  - [ ] SubTask 8.4: 阶段4训练 (混合配置)

## 阶段4: 物理约束与高级改进 (1-2周)

- [ ] Task 9: 实现简化运动模型（可选）
  - [ ] SubTask 9.1: 创建 HerderEntity 类
  - [ ] SubTask 9.2: 实现速度/加速度限制
  - [ ] SubTask 9.3: 更新 SheepScenario 使用 HerderEntity
  - [ ] SubTask 9.4: 对比无约束vs有约束效果

- [ ] Task 10: 超参数调优
  - [ ] SubTask 10.1: 学习率调优
  - [ ] SubTask 10.2: PPO裁剪参数调优
  - [ ] SubTask 10.3: 熵系数调优
  - [ ] SubTask 10.4: κ约束范围调优

## 阶段5: 泛化性验证 (1周)

- [ ] Task 11: 泛化性测试
  - [ ] SubTask 11.1: 测试训练范围内场景 (10羊3机械狗)
  - [ ] SubTask 11.2: 测试少羊少狗场景 (5羊2机械狗)
  - [ ] SubTask 11.3: 测试多羊多狗场景 (20羊4机械狗)
  - [ ] SubTask 11.4: 测试极端场景 (3羊2机械狗, 30羊5机械狗)
  - [ ] SubTask 11.5: 测试不同世界大小

- [ ] Task 12: 性能评估报告
  - [ ] SubTask 12.1: 整理实验数据
  - [ ] SubTask 12.2: 生成性能对比图表
  - [ ] SubTask 12.3: 生成泛化性对比图表
  - [ ] SubTask 12.4: 撰写评估报告

# Task Dependencies

- Task 2 depends on Task 1
- Task 3 depends on nothing (可并行)
- Task 4 depends on Task 1, Task 2
- Task 5 depends on Task 1, Task 2, Task 3, Task 4
- Task 6 depends on Task 5
- Task 7 depends on Task 5
- Task 8 depends on Task 6, Task 7
- Task 9 depends on Task 8
- Task 10 depends on Task 8
- Task 11 depends on Task 9, Task 10
- Task 12 depends on Task 11
