# 羊群引导强化学习（中文文档）

在二维场地内用若干机械狗引导 Boids 羊群，使羊群质心接近目标（默认圆心 `(0,0)`）。本项目训练**单智能体高层 PPO**：输出编队几何参数，在**以羊质心为圆心**的圆弧上生成各狗目标位；低层可为内置势场运动学，或对接 **[InforMARL](InforMARL/)** Graph MAPPO。

**文档导航**

| 文档 | 用途 |
|------|------|
| [README.md](README.md) | 快速上手（安装、命令、脚本索引） |
| [docs/TECHNICAL.md](docs/TECHNICAL.md) | 与源码同步的技术规格（观测/动作/奖励、目录） |
| [envs/defaults.py](envs/defaults.py) | 共享默认超参单源（场地、课程、`formation_delta` 上限等） |

---

## 1. 项目概述

### 1.1 分层控制

```
┌─────────────────────────────────────────────────────────────┐
│  高层（本仓库 PPO）                                            │
│  观测：羊群质心/包络/速度 + 狗群相对羊质心等（10 或 13 维）        │
│  动作：5 维 Box[-1,1]；解码有效维 a[0:3] → θ_in, R, coverage   │
│  输出：N 个机械狗在羊质心圆弧上的目标位置                        │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────────────┐
│ 内置低层（默认）      │       │ InforMARL Graph MAPPO（可选） │
│ 势场：趋向目标 + 避羊 │       │ 每主步若干子步导航            │
│ 或 --herder_teleport  │       │ --low-level-model-dir       │
└─────────────────────┘       └─────────────────────────────┘
```

### 1.2 关键特性

- **PPO**：单智能体连续控制，支持 GAE、奖励归一化、课程学习
- **编队解码**：羊质心圆弧几何（`envs/high_level_action.py`），非旧版 wedge/κ 四参数
- **编队增量模式**（`--formation_delta`）：观测 13 维，动作解释为 Δθ_in、ΔR、Δcoverage
- **泛化**：同一 checkpoint 可在不同羊数/狗数下评估（`evaluate_generalization.py`）
- **可选联合仿真**：`train_hierarchical.py` / `visualize.py` 加载低层 MAPPO 权重

---

## 2. 世界与坐标

- `world_size = (W, H)`：矩形语义；圆盘半径 `R_disk = min(W,H)/2`，目标在圆心。
- `dt` 默认 **2.0** s；各入口 `episode_length` 默认见 `envs/defaults.py`（训练 150、评估 100、可视化 150 等，**勿混用**）。
- **`--no-disk-boundary`**：羊/狗位置不按圆盘裁剪；绘图坐标轴按矩形 `world_size`；高层槽位也不再裁进圆盘/矩形（仅狗间最小距）。
- **`--disk-boundary`**（未加 `no-disk` 时）：位置裁剪到圆盘内。

---

## 3. 观测空间

### 3.1 基础 10 维（与狗数 N 无关）

| 索引 | 含义 |
|------|------|
| `[0]` | 羊质心到目标距离 / `r_max`（`r_max = world_radius`） |
| `[1]` | 质心相对目标的方位角 / π，∈ [-1, 1] |
| `[2]` | 羊群平均速度模 / `max_speed` |
| `[3]` | 平均速度方向角 / π（**世界系**，非相对目标） |
| `[4]–[7]` | 各羊相对目标的轴对齐四向极值 `e0…e3`（包络，与奖励一致）/ `r_max` |
| `[8]` | N 狗位置质心相对羊质心的极径 / `r_max` |
| `[9]` | 上述相对位移的方位角 / π |

实现：`SheepScenario._observation_snapshot` / `get_observation`。

### 3.2 编队增量模式（+3 维，`--formation_delta`）

在 10 维后拼接：当前 `θ_in/π`、`R` 线性归一化到 [-1,1]、`2*coverage-1`。策略输入 **13 维**；与 **10 维绝对编队** checkpoint **不兼容**。

### 3.3 共享观测（Critic / 多智能体扩展）

`get_shared_obs` 长度 **10 + 2N**（+ 增量模式再 +3）。默认 PPO **仅用主观测维**（10 或 13），不随 N 变化。

---

## 4. 动作空间与编队解码

### 4.1 动作维

- **5 维** `Box([-1,1]^5)`；环境只使用 **`actions[0]`** 的前 3 维；`a[3]、a[4]` 在解码器内置 0。

### 4.2 绝对模式（默认，`formation_delta_mode=False`）

| 维 | 映射 |
|----|------|
| `a[0]` | `θ_in = wrap(a[0]·π)`：弧中点 → 羊质心方向；弧中心 `θ_mid = θ_in − π` |
| `a[1]` | `R ∈ [R_min, R_max]`，默认 **5–20 m**（`STANCE_RADIUS_MIN/MAX`） |
| `a[2]` | `coverage ∈ [0,1]` → 弧张角 Θ（与 N 相关） |

在圆心 = **当前羊质心**、半径 `R` 的弧上均匀取 N 个目标点，再经最小间距与边界处理（`no-disk` 时不裁圆盘）。

### 4.3 增量模式（`--formation_delta`）

每档高层刷新时：

- `Δθ_in = a[0] × θ_max`（默认 **22.5°**）
- `ΔR = a[1] × 3.0 m`
- `Δcoverage = a[2] × 0.15`

积分 clip 后复用绝对模式同一几何解码。上限见 `envs/defaults.py` 中 `FORMATION_DELTA_*`。

### 4.4 高层刷新频率

当 `(step_count - 1) % high_level_interval == 0` 时解码并 `set_herder_targets`（**N=1 时每步刷新**）。

| 模式 | `high_level_interval` 默认 |
|------|---------------------------|
| 增量 | **1** |
| 绝对 | **5** |

训练/评估/可视化可用 `--high_level_interval` 覆盖；泛化评估默认 **3**。

---

## 5. 动力学

### 5.1 羊

Boids：分离 / 对齐 / 凝聚 + 对机械狗逃避 +（可选）圆边界斥力。参数见 `sheep_config` 与 `RandomizedSheepFlockEnv` 采样范围。

### 5.2 机械狗

| 模式 | CLI | 行为 |
|------|-----|------|
| 势场运动学（常见） | 默认 | 吸引目标 + 近羊群排斥，步长 `min(5.0*dt, dist)` |
| 瞬移 | `--herder_teleport` | 位置直接设为目标（课程早期常用） |
| MAPPO 低层 | `--low-level-model-dir` | 每主步 `low_level_substeps` 子仿真（默认 `round(dt/0.1)`） |

`HerderEntity` 类存在，**未接入**主循环。

### 5.3 成功判定

`is_flock_at_target(threshold=5.0)`：质心距目标 < 5 m。

---

## 6. 奖励函数

`SheepFlockEnv._compute_reward`（系数默认见 `envs/defaults.py` → `DEFAULT_REWARD_CONFIG`）：

1. **势函数**：归一化距离差的平方项 × `w_potential`
2. **近目标**：指数型 `w_near * exp(-d/d0)`
3. **速度正则**：近目标处抑制质心速度
4. **包络惩罚**：轴对齐 AABB 面积（与观测 `e0…e3` 一致）
5. **时间惩罚**：远离目标时 `-time_penalty`

总奖励 clip 到 `[reward_clip_low, reward_clip_high]`（默认约 [-3, 5]）。`info['reward_components']` 可供 TensorBoard 子项记录。

---

## 7. 课程学习与随机化

### 7.1 课程阶段

定义于 **`envs/defaults.py` → `DEFAULT_CURRICULUM_STAGE_SPECS`**（8 个阶段，羊数 3→30、场地与 `episode_length` 递增；末阶段可 `num_sheep_range` 随机羊数）。

启用：`python train_ppo.py --use_curriculum --start_stage 0`。`scripts/train_gpu.sh` 为 GPU + 课程 + `formation_delta` + `no-disk-boundary` 示例。

### 7.2 随机化环境

`--use_randomized`：每 episode 随机羊数、狗数、世界尺寸、羊速等（范围见 `defaults.py`）。

---

## 8. PPO 算法说明（摘要）

本项目为**单智能体**高层决策（一条动作广播到环境），选用 PPO 的原因：

- 连续动作（5 维有界）无需离散化
- 裁剪目标提升策略更新稳定性（羊群 Boids 随机性强）
- 与课程阶段切换、长 horizon 训练相匹配

常用训练超参（`scripts/train_gpu.sh` 量级）：

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `lr` | 3e-4 | 可配合 warmup / 线性衰减 |
| `ppo_epoch` | 10 | |
| `num_mini_batch` | 4 | |
| `hidden_size` | 256–512 | 大模型见 train_gpu.sh |
| `layer_N` | 3 | |
| `gamma` / `gae_lambda` | 0.99 / 0.95–0.98 | |
| `clip_param` | 0.2 | 可选退火 |
| `max_grad_norm` | 0.5 | |
| `use_reward_normalization` | 可选 | Running 归一化 |

完整 CLI：`python train_ppo.py -h`。

---

## 9. 分层低层 MAPPO

- 桥接：`envs/low_level_mappo_bridge.py`
- 训练入口：`train_hierarchical.py`（内部调用 `train_ppo.main`）
- 权重目录需含 **`actor.pt`** 或与 InforMARL 训练一致的 checkpoint
- 关键参数：`--low-level-substeps`、`--low-level-device`、`--low-level-num-obstacles`、`--low-level-max-speed`、`--low-level-max-edge-dist`（大场地默认 30）

InforMARL 经 patch 支持 `external_goals` 与宿主写入 landmark，详见子项目 README。

---

## 10. 安装与项目结构

```bash
cd sheep-herding-rl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```
sheep-herding-rl/
├── envs/                         # 环境、场景、编队解码、低层桥接
│   ├── sheep_flock.py
│   ├── sheep_scenario.py
│   ├── high_level_action.py
│   ├── curriculum_env.py
│   ├── defaults.py               # 默认超参单源
│   └── low_level_mappo_bridge.py
├── onpolicy/                     # PPO 策略与 buffer
├── train_ppo.py                  # 主训练
├── train_hierarchical.py         # 分层训练（CLI 同 train_ppo）
├── evaluate_policy.py
├── evaluate_generalization.py
├── visualize.py
├── plot_generalization.py
├── scripts/                      # train_gpu.sh、visualize.sh、配图脚本等
├── tests/
├── docs/TECHNICAL.md
├── figures/                      # 轨迹、GIF、评估图
└── InforMARL/
```

---

## 11. 快速开始

### 11.1 训练（仅高层）

```bash
python train_ppo.py \
  --device cuda \
  --num_sheep 10 --num_herders 3 \
  --world_size 100 100 \
  --formation_delta \
  --no-disk-boundary \
  --num_env_steps 1000000

# GPU + 课程示例
bash scripts/train_gpu.sh
```

### 11.2 训练（高层 + 低层 MAPPO）

```bash
python train_hierarchical.py \
  --low-level-model-dir /path/to/mappo_run/models \
  --formation_delta --no-disk-boundary \
  --num_env_steps 500000
```

### 11.3 单配置评估

```bash
python evaluate_policy.py --model_path /path/to/model.pt --num_episodes 100
bash scripts/evaluate_policy.sh /path/to/model.pt 100
```

### 11.4 泛化网格评估

默认：羊 **5,10,15,20,25** × 狗 **3,4,5,6**；默认 **无圆盘**、`high_level_interval=3`、势场狗（非 teleport）。checkpoint 自动推断 10/13 维以开关 `formation_delta`。

```bash
python3 evaluate_generalization.py \
  --model_path /path/to/model.pt \
  --num_episodes 50 \
  --output_json figures/generalization/grid_eval.json \
  --output_figures figures/generalization/plots

bash scripts/evaluate_generalization.sh /path/to/model.pt 50
```

环境变量：`GEN_SEED`、`GEN_PER_COMBO_SEED=1`、`GEN_RUN_FULL_HORIZON=1`、`GEN_DISK_BOUNDARY=1`、`GEN_HERDER_TELEPORT=1`、`GEN_HIGH_LEVEL_INTERVAL=N` 等（见 `scripts/evaluate_generalization.sh` 注释）。

### 11.5 可视化

```bash
python visualize.py --model_path /path/to/model.pt
bash scripts/visualize.sh /path/to/model.pt
```

`scripts/visualize.sh` 默认：`--formation_delta`、`--no-disk-boundary`、`--high_level_interval 1`。环境变量：

| 变量 | 作用 |
|------|------|
| `VIS_SAVE_GIF` | 保存 GIF（重名自动 `_1`、`_2`…） |
| `VIS_DEVICE` | 有显示器时默认 `cpu`，避免与 GUI 抢 GPU |
| `VIS_STOCHASTIC=1` | 随机策略采样 |
| `VIS_FORMATION_DELTA=1` | 强制 13 维观测 |
| `VIS_HERDER_TELEPORT=1` | 狗瞬移 |
| `VIS_SAVE_VISUAL_HERDER_TRAILS=1` | 步进快照叠加狗轨迹 |

常用 CLI：

- `--save-sheep-trajectory-dir DIR`：每局结束保存全羊轨迹 PNG
- `--save-visual-every K`：每 K 步保存画面；默认子目录 `figures/viz_snapshots/...`
- `--save-visual-herder-trails`：快照上画狗轨迹折线
- `--initial-flock-centroid CX CY`：固定初始羊群质心
- `--low-level-model-dir`：联合低层 MAPPO

路径以 `/figures/...` 开头时会解析为**仓库内** `figures/...`（勿与系统根 `/figures` 混淆）。

### 11.6 测试

```bash
python -m pytest tests/ -q
```

---

## 12. 可视化界面说明

| 元素 | 说明 |
|------|------|
| 目标 | 圆心绿色星标 |
| 羊 | 灰色散点 |
| 机械狗 | 蓝色方块 |
| 编队弧 / 目标点 | 由 `HighLevelAction` 解码绘制 |
| 无窗口 | `MPLBACKEND=Agg` 或 `--force_headless`；可用 `--save_gif` |

交互：支持 Pause/Resume（空格）；后端自动尝试 TkAgg → Qt5Agg → …，无 tk 时回退 Agg。

---

## 13. 辅助脚本（示意 / 配图）

**不加载真实策略**，用于论文配图或几何示意：

| 脚本 | 作用 |
|------|------|
| `scripts/generate_synthetic_failure_trajectories.py` | 失败、冲散轨迹 |
| `scripts/generate_synthetic_success_trajectories.py` | 成功引导轨迹 |
| `scripts/generate_synthetic_near_miss.py` | 「擦肩而过」轨迹 |
| `scripts/plot_synthetic_hrl_vs_mappo_reward.py` | 合成回报曲线 |
| `scripts/plot_synthetic_hrl_vs_mappo_success_rate.py` | 合成成功率曲线 |
| `scripts/generate_formation_gallery.py` | 12 张编队站位网格 |
| `scripts/visualize_flock_envelope.py` | 四向包络与 AABB |
| `scripts/formation_sliders.py` | 交互式编队滑块 |

---

## 14. 调试与 FAQ

### 14.1 观测范围

```python
obs = env.reset()
print(obs.min(), obs.max())  # 多数分量在 [-1, 1] 附近
```

### 14.2 动作解码抽查

```python
from envs.high_level_action import HighLevelAction
import numpy as np
dec = HighLevelAction()
fc = np.zeros(2)
raw = np.random.uniform(-1, 1, 5)
out = dec.decode_action(raw, fc, (100.0, 100.0))
print(out["theta_in_rad"], out["radius"], out["coverage"])
```

### 14.3 常见问题

| 现象 | 可能原因 | 建议 |
|------|----------|------|
| 加载模型 shape 不匹配 | 训练未开 `formation_delta` 但评估开了（或反之） | 与 checkpoint 一致；泛化脚本会自动推断 |
| 可视化成功率远低于训练 | `high_level_interval`、边界、teleport 不一致 | 对齐 `train_ppo` 所用 CLI |
| 权限错误写 `/figures` | 误写绝对路径 `/figures/...` | 用 `figures/...` 或 `$PWD/figures/...` |
| GUI 卡死 | CUDA 与 Xorg 同 GPU | `VIS_DEVICE=cpu` 或 `--device cpu` |
| 低层狗不动 / 报错 | 权重路径、`max_edge_dist`、障碍数与训练不一致 | 对照 InforMARL 训练 config |

### 14.4 训练日志

TensorBoard：`results/sheep_herding/...` 下 `--use_tensorboard` 运行目录。文本日志：`training_log.txt`、`training_metrics.json`。

---

## 15. 已知限制与未来工作

**当前限制**

- 仿真羊为 Boids，与真实牲畜行为有差距
- 主环境无静态障碍物（低层 MPE 可有障碍）
- 单一固定目标（圆心）
- Sim2Real、多目标、动态目标等**未在主线代码实现**（勿与旧版 README 示例类混用）

**可扩展方向**

- 域随机化（已有 `RandomizedSheepFlockEnv` 基础）
- 障碍物与多目标场景
- 低层 MAPPO 与高层交替微调（`train_hierarchical.py` 占位参数）

---

## 16. 参考文献

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017  
- Schulman et al., *Generalized Advantage Estimation*, 2016  
- Reynolds, *Flocks, Herds, and Schools*, 1987  
- InforMARL / Graph MAPPO 相关论文见 [InforMARL/README.md](InforMARL/README.md)

---

## 17. 更新日志

### 2026-05（与当前仓库对齐）

- 观测改为 **10 维**（含四向包络 `e0…e3`、狗群相对羊质心）；可选 **+3** 编队状态（`--formation_delta`）
- 动作改为 **5 维**，有效 **a[0:3]**；编队几何为**羊质心圆弧**（θ_in、R、coverage），废弃 wedge/μ_r/σ_r/κ 旧描述
- 奖励改为势函数 + 近目标 + 速度正则 + 包络 + 时间惩罚（见 `defaults.py`）
- 支持 **`--no-disk-boundary`**、**`--herder_teleport`**、**`--high_level_interval`**
- 集成 **InforMARL** 低层：`low_level_mappo_bridge.py`、`train_hierarchical.py`
- 泛化评估 `evaluate_generalization.py` + 折线图 `plot_generalization.py`
- 可视化：轨迹 PNG、步进快照、初始质心、低层联合、`scripts/visualize.sh` 环境变量
- 默认超参集中于 **`envs/defaults.py`**；技术细节见 **`docs/TECHNICAL.md`**

---

## 联系与贡献

问题排查优先查阅 [docs/TECHNICAL.md](docs/TECHNICAL.md)（CLI 速查、低层桥接、评估/可视化细节）与 `python train_ppo.py -h`。英文简明索引见 [README.md](README.md)。

### 附录：PPO 选型摘要（理论）

羊群引导为**单智能体、连续动作**任务：高层一次输出编队参数，由解码器生成 N 个目标位。选用 PPO 而非 MAPPO（多狗各自独立策略）或 SAC 的原因包括：（1）裁剪目标限制策略突变，适应 Boids 随机性；（2）GAE + 多 epoch 更新提高样本利用；（3）超参相对可控，便于课程阶段切换。实现细节（KL、Huber value、奖励归一化、梯度跳过等）见 `train_ppo.py` 与 `onpolicy/` 源码。
