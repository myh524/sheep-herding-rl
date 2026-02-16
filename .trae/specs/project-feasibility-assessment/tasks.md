# 项目可行性评估后续任务清单

## 阶段一：关键问题修复 (必须完成)

- [ ] Task 1: 修复测试与实现不一致问题
  - [ ] SubTask 1.1: 更新test_env.py中的观测空间断言从10维改为12维
  - [ ] SubTask 1.2: 运行测试确认所有测试通过
  - [ ] SubTask 1.3: 检查其他测试文件是否存在类似问题

- [ ] Task 2: 创建依赖管理文件
  - [ ] SubTask 2.1: 创建requirements.txt，包含numpy、torch、gym等核心依赖
  - [ ] SubTask 2.2: 指定版本范围确保兼容性
  - [ ] SubTask 2.3: 测试在新环境中的安装流程

- [ ] Task 3: 验证训练收敛性
  - [ ] SubTask 3.1: 运行小规模训练（100K步）验证PPO能正常学习
  - [ ] SubTask 3.2: 监控奖励曲线是否呈上升趋势
  - [ ] SubTask 3.3: 检查是否有NaN或异常值

- [ ] Task 4: 优化奖励函数设计
  - [ ] SubTask 4.1: 分析当前奖励各分量的贡献比例
  - [ ] SubTask 4.2: 调整奖励权重，确保尺度一致
  - [ ] SubTask 4.3: 验证优化后的奖励函数

## 阶段二：系统改进 (建议完成)

- [ ] Task 5: 添加训练监控工具
  - [ ] SubTask 5.1: 集成TensorBoard记录训练指标
  - [ ] SubTask 5.2: 添加奖励曲线、损失曲线、成功率等可视化
  - [ ] SubTask 5.3: 支持实时监控训练进度

- [ ] Task 6: 改进多智能体协调机制
  - [ ] SubTask 6.1: 分析当前动作展开策略的局限性
  - [ ] SubTask 6.2: 设计改进方案（独立动作或条件策略）
  - [ ] SubTask 6.3: 实现并验证改进效果

- [ ] Task 7: 增强环境随机化
  - [ ] SubTask 7.1: 扩展RandomizedSheepFlockEnv的随机化参数
  - [ ] SubTask 7.2: 添加Boids权重的随机化
  - [ ] SubTask 7.3: 测试随机化对泛化能力的影响

- [ ] Task 8: 建立持续集成流程
  - [ ] SubTask 8.1: 创建GitHub Actions配置
  - [ ] SubTask 8.2: 配置自动运行测试
  - [ ] SubTask 8.3: 添加代码质量检查

## 阶段三：长期优化 (可选)

- [ ] Task 9: 超参数系统调优
  - [ ] SubTask 9.1: 设计超参数搜索空间
  - [ ] SubTask 9.2: 实现自动化调参脚本
  - [ ] SubTask 9.3: 分析最优超参数组合

- [ ] Task 10: Sim-to-Real迁移研究
  - [ ] SubTask 10.1: 分析仿真与真实环境的差异
  - [ ] SubTask 10.2: 研究域适应方法
  - [ ] SubTask 10.3: 设计迁移验证实验

## 任务依赖关系

- [Task 3] 依赖 [Task 1, Task 2]
- [Task 4] 依赖 [Task 3]
- [Task 5] 可与 [Task 6, Task 7] 并行
- [Task 9] 依赖 [Task 3, Task 4, Task 5]
- [Task 10] 依赖 [Task 9]

## 验收标准

### 阶段一验收
- 所有测试通过
- 新环境可成功安装依赖
- 小规模训练（100K步）奖励曲线呈上升趋势
- 奖励函数各分量比例合理

### 阶段二验收
- TensorBoard可正常记录和显示训练指标
- 多智能体协调策略改进后成功率提升
- 随机化环境下策略保持稳定性能
- CI流程正常运行

### 阶段三验收
- 找到最优超参数组合
- 完成Sim-to-Real迁移可行性报告
