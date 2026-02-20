# 离散动作可视化

本目录包含21个离散动作的可视化图片，每个动作展示了不同的机械狗站位策略。

## 动作空间设计

### 优化后的动作空间

动作空间由3个维度的参数组合而成，但**只有front方向才有full_surround队形**：

#### 1. 方向维度 (Wedge Center) - 3个选项
- **front** (1.0): 正前方站位
- **left** (-0.5): 左侧站位
- **right** (0.5): 右侧站位

#### 2. 队形维度 (Wedge Width) - 3个选项
- **push** (-0.8): 推进队形（窄队形）
- **half_surround** (0.0): 半包围队形（中等宽度）
- **full_surround** (1.0): 全包围队形（宽队形）**仅front方向**

#### 3. 距离维度 (Radius) - 3个选项
- **near** (-0.6): 近距离站位
- **mid** (0.0): 中等距离站位
- **far** (0.6): 远距离站位

### 动作数量

- **Front方向**: 9个动作 (3队形 × 3距离)
- **Left方向**: 6个动作 (2队形 × 3距离，无full_surround)
- **Right方向**: 6个动作 (2队形 × 3距离，无full_surround)
- **总计**: 21个动作

## 动作列表

### Front 方向 (0-8) - 完整的9个动作

| 动作ID | 名称 | 描述 | 动作向量 |
|--------|------|------|----------|
| 0 | FRONT_PUSH_NEAR | front方向-push-near距离 | [1.0, -0.8, -0.6, 0.0] |
| 1 | FRONT_PUSH_MID | front方向-push-mid距离 | [1.0, -0.8, 0.0, 0.0] |
| 2 | FRONT_PUSH_FAR | front方向-push-far距离 | [1.0, -0.8, 0.6, 0.0] |
| 3 | FRONT_HALF_SURROUND_NEAR | front方向-half_surround-near距离 | [1.0, 0.0, -0.6, 0.0] |
| 4 | FRONT_HALF_SURROUND_MID | front方向-half_surround-mid距离 | [1.0, 0.0, 0.0, 0.0] |
| 5 | FRONT_HALF_SURROUND_FAR | front方向-half_surround-far距离 | [1.0, 0.0, 0.6, 0.0] |
| 6 | FRONT_FULL_SURROUND_NEAR | front方向-full_surround-near距离 | [1.0, 1.0, -0.6, 0.0] |
| 7 | FRONT_FULL_SURROUND_MID | front方向-full_surround-mid距离 | [1.0, 1.0, 0.0, 0.0] |
| 8 | FRONT_FULL_SURROUND_FAR | front方向-full_surround-far距离 | [1.0, 1.0, 0.6, 0.0] |

### Left 方向 (9-14) - 6个动作（无full_surround）

| 动作ID | 名称 | 描述 | 动作向量 |
|--------|------|------|----------|
| 9 | LEFT_PUSH_NEAR | left方向-push-near距离 | [-0.5, -0.8, -0.6, 0.0] |
| 10 | LEFT_PUSH_MID | left方向-push-mid距离 | [-0.5, -0.8, 0.0, 0.0] |
| 11 | LEFT_PUSH_FAR | left方向-push-far距离 | [-0.5, -0.8, 0.6, 0.0] |
| 12 | LEFT_HALF_SURROUND_NEAR | left方向-half_surround-near距离 | [-0.5, 0.0, -0.6, 0.0] |
| 13 | LEFT_HALF_SURROUND_MID | left方向-half_surround-mid距离 | [-0.5, 0.0, 0.0, 0.0] |
| 14 | LEFT_HALF_SURROUND_FAR | left方向-half_surround-far距离 | [-0.5, 0.0, 0.6, 0.0] |

### Right 方向 (15-20) - 6个动作（无full_surround）

| 动作ID | 名称 | 描述 | 动作向量 |
|--------|------|------|----------|
| 15 | RIGHT_PUSH_NEAR | right方向-push-near距离 | [0.5, -0.8, -0.6, 0.0] |
| 16 | RIGHT_PUSH_MID | right方向-push-mid距离 | [0.5, -0.8, 0.0, 0.0] |
| 17 | RIGHT_PUSH_FAR | right方向-push-far距离 | [0.5, -0.8, 0.6, 0.0] |
| 18 | RIGHT_HALF_SURROUND_NEAR | right方向-half_surround-near距离 | [0.5, 0.0, -0.6, 0.0] |
| 19 | RIGHT_HALF_SURROUND_MID | right方向-half_surround-mid距离 | [0.5, 0.0, 0.0, 0.0] |
| 20 | RIGHT_HALF_SURROUND_FAR | right方向-half_surround-far距离 | [0.5, 0.0, 0.6, 0.0] |

## 动作参数说明

每个动作由4个参数组成：

1. **wedge_center** (楔形中心角度)
   - 控制队形相对于目标方向的角度
   - front=1.0, left=-0.5, right=0.5

2. **wedge_width** (楔形宽度)
   - 控制机械狗的分散程度
   - push=-0.8 (窄队形), half_surround=0.0 (中等), full_surround=1.0 (宽队形)

3. **radius** (站位半径)
   - 控制机械狗距离羊群中心的距离
   - near=-0.6 (近), mid=0.0 (中), far=0.6 (远)

4. **asymmetry** (不对称度)
   - 固定为0.0，对称队形

## 最新参数调整

### 相比上一版本的变化：

1. **动作数量优化**
   - 从 27 个减少到 **21 个**
   - 去除了 left 和 right 方向的 full_surround 队形
   - 原因：侧面方向不需要360度包围

2. **队形参数微调**
   - `push.wedge_width` 从 -1.0 改为 **-0.8**
   - 稍微放宽推进队形，避免过于极端

3. **距离参数对称化**
   - `far.radius` 从 0.5 改为 **0.6**
   - 与 near (-0.6) 形成对称

### 参数对比表

| 参数 | 上一版本 | 当前版本 | 变化说明 |
|------|----------|----------|----------|
| 动作数量 | 27 | **21** | 去除冗余动作 |
| push.wedge_width | -1.0 | **-0.8** | 稍微放宽 |
| far.radius | 0.5 | **0.6** | 与near对称 |
| left/right full_surround | 有 | **无** | 去除冗余 |

## 动作设计优势

### 1. 去除冗余
- 侧面方向不需要360度包围
- 减少动作空间大小，提高学习效率
- 21个动作更加精简实用

### 2. 参数优化
- 推进队形宽度适中 (-0.8)
- 距离参数对称 (-0.6, 0.0, 0.6)
- 更符合实际应用场景

### 3. 系统化设计
- Front方向：完整的3×3=9个动作
- Left/Right方向：各2×3=6个动作
- 清晰的动作分类和命名

### 4. 实用性高
- 推进队形：快速移动羊群
- 半包围：平衡推进和控制
- 全包围：仅front方向，防止羊群后退

## 图片说明

每张图片包含以下元素：

### 场景元素
- 绿色星形: 目标位置
- 灰色圆点: 羊群位置
- 蓝色方块: 机械狗当前位置
- 红色圆点: 主要目标方向
- 橙色圆环: 站位半径
- 橙色小圆点: 每个机械狗的目标位置

### 信息面板
- 左上角: 动作名称、描述、模式、半径、扩散角度、不对称度
- 右下角: 动作向量参数

## 生成方法

运行以下命令生成所有图片：

```bash
python visualize_discrete_actions.py
```

## 使用场景

这些可视化图片可用于：

1. **理解动作空间**: 直观了解每个离散动作的含义
2. **策略分析**: 分析训练好的策略偏好哪些动作
3. **动作设计**: 评估和改进离散动作的设计
4. **文档说明**: 在报告和演示中展示动作定义

## 文件列表

- 21张单个动作图片 (action_00 到 action_20)
- 1张汇总图 (all_actions_summary.png)
- 本README文档

## 动作索引规则

动作按以下顺序排列：
1. Front方向的所有动作 (0-8)
2. Left方向的push和half_surround动作 (9-14)
3. Right方向的push和half_surround动作 (15-20)
