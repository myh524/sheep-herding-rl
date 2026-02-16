# 模型可视化分析系统 Spec

## Why
训练完成后需要可视化分析模型性能，包括实时观察策略行为、分析训练曲线、评估策略质量，以便理解模型学习效果并进行调优。

## What Changes
- 创建可视化运行脚本 `visualize.py`，支持加载训练好的模型并实时渲染环境
- 创建训练分析脚本 `analyze_training.py`，绘制训练曲线和统计指标
- 创建策略评估脚本 `evaluate_policy.py`，运行多episode并输出性能统计
- 创建动画录制功能，支持导出GIF/MP4

## Impact
- Affected specs: 新增可视化分析能力
- Affected code: 新增独立脚本文件，不修改现有训练代码

## ADDED Requirements

### Requirement: 模型可视化运行
系统 SHALL 提供可视化运行脚本，能够加载训练好的模型checkpoint并在环境中运行。

#### Scenario: 加载模型并可视化
- **WHEN** 用户指定模型checkpoint路径
- **THEN** 系统加载模型参数、创建环境、运行策略并以动画形式展示

#### Scenario: 交互式可视化
- **WHEN** 用户运行可视化脚本
- **THEN** 系统提供实时渲染窗口，显示羊群、机械狗、目标位置等元素

#### Scenario: 网络输入输出可视化
- **WHEN** 策略网络运行时
- **THEN** 系统同步显示网络输入（8维观测向量：目标距离、方向角、羊群形状特征、站位半径、速度等）和输出（4维动作向量：半径均值、半径标准差、相对角度、集中度）

### Requirement: 训练过程分析
系统 SHALL 提供训练日志分析功能，绘制训练曲线和关键指标。

#### Scenario: 绘制训练曲线
- **WHEN** 用户指定训练日志路径
- **THEN** 系统解析日志并绘制奖励曲线、损失曲线等图表

#### Scenario: 多run对比
- **WHEN** 用户指定多个训练run目录
- **THEN** 系统在同一图表中对比不同训练配置的性能

### Requirement: 策略性能评估
系统 SHALL 提供策略评估功能，运行多个episode并输出统计指标。

#### Scenario: 批量评估
- **WHEN** 用户指定评估episode数量
- **THEN** 系统运行多个episode并输出成功率、平均奖励、平均步数等统计

#### Scenario: 详细行为分析
- **WHEN** 用户启用详细分析模式
- **THEN** 系统输出动作分布、价值估计、羊群状态变化等详细分析

### Requirement: 动画导出
系统 SHALL 支持将可视化结果导出为动画文件。

#### Scenario: 导出GIF
- **WHEN** 用户指定--save_gif参数
- **THEN** 系统将episode渲染结果保存为GIF文件

#### Scenario: 导出MP4
- **WHEN** 用户指定--save_video参数
- **THEN** 系统将episode渲染结果保存为MP4文件

## MODIFIED Requirements
无

## REMOVED Requirements
无
