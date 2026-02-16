# Tasks

- [x] Task 1: 创建可视化运行脚本 visualize.py
  - [x] SubTask 1.1: 实现模型加载函数，支持从checkpoint恢复policy网络
  - [x] SubTask 1.2: 实现环境创建函数，支持命令行参数配置环境参数
  - [x] SubTask 1.3: 实现渲染循环，使用matplotlib动画实时显示环境状态
  - [x] SubTask 1.4: 添加命令行参数解析，支持指定模型路径、环境配置等
  - [x] SubTask 1.5: 实现动画导出功能，支持保存为GIF/MP4
  - [x] SubTask 1.6: 实现网络输入输出面板，同步显示8维观测向量和4维动作向量

- [x] Task 2: 创建训练分析脚本 analyze_training.py
  - [x] SubTask 2.1: 实现日志解析函数，读取training_log.txt
  - [x] SubTask 2.2: 实现训练曲线绘制，包括奖励、损失等指标
  - [x] SubTask 2.3: 实现多run对比功能，支持在同一图表中比较不同训练配置
  - [x] SubTask 2.4: 添加平滑曲线选项，支持移动平均平滑

- [x] Task 3: 创建策略评估脚本 evaluate_policy.py
  - [x] SubTask 3.1: 实现批量评估函数，运行多个episode
  - [x] SubTask 3.2: 实现统计计算，输出成功率、平均奖励、平均步数等
  - [x] SubTask 3.3: 实现详细分析模式，输出动作分布、价值估计等
  - [x] SubTask 3.4: 支持课程学习环境的阶段评估

# Task Dependencies
- [Task 2] 独立任务，无依赖
- [Task 3] 独立任务，无依赖
- [Task 1] 是核心任务，Task 3 可复用其模型加载逻辑
