# Sheep Herding RL

羊群引导强化学习：高层 PPO 编队策略，可选 **InforMARL Graph MAPPO** 低层导航（[`InforMARL/`](InforMARL/)）联合仿真。

## 文档

| 文档 | 说明 |
|------|------|
| **[docs/TECHNICAL.md](docs/TECHNICAL.md)** | 与当前源码同步的技术说明（观测/动作、奖励、目录结构） |
| **[README_CN.md](README_CN.md)** | 中文详细文档（架构、观测/动作、训练调参、FAQ 等） |

共享默认超参集中在 **[`envs/defaults.py`](envs/defaults.py)**（场地尺寸、episode 长度、课程阶段、`formation_delta` 上限等），各入口通过引用该文件保持一致。

## 环境要求与安装

- Python 3.10+（推荐）
- PyTorch、CUDA（GPU 训练时）

```bash
cd sheep-herding-rl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

可选：无显示器保存 GIF/快照时无需 `python3-tk`；交互式弹窗时 `visualize.py` 会按 `DISPLAY` 自动尝试 TkAgg → Qt5Agg → …，失败则回退 Agg。

## 仓库结构（核心）

```
sheep-herding-rl/
├── envs/                    # SheepFlockEnv、场景、低层 MAPPO 桥接
├── onpolicy/                # PPO 策略与 buffer
├── train_ppo.py             # 主训练入口
├── train_hierarchical.py    # 分层训练（CLI 同 train_ppo，启用低层）
├── evaluate_policy.py       # 单配置评估
├── evaluate_generalization.py  # 羊数×狗数网格泛化评估
├── visualize.py             # Rollout 可视化
├── plot_generalization.py   # 泛化折线图
├── scripts/                 # Shell 与配图/示意脚本
├── figures/                 # 轨迹图、GIF、评估图等输出（git 可忽略部分）
├── docs/TECHNICAL.md
└── InforMARL/               # Graph MAPPO 子项目（低层导航）
```

## 训练

### 仅高层 PPO（默认）

```bash
python train_ppo.py \
  --device cuda \
  --num_sheep 10 --num_herders 3 \
  --world_size 100 100 \
  --formation_delta \
  --no-disk-boundary \
  --num_env_steps 1000000
```

常用开关（与评估/可视化对齐）：

- **`--formation_delta`**：观测 13 维，动作 `a[0:3]` 为编队增量（与 10 维绝对编队 checkpoint 不兼容）
- **`--no-disk-boundary`**：羊/狗不受圆盘裁剪；坐标轴按 `world_size` 矩形
- **`--herder_teleport`**：机械狗瞬移到编队槽位（关闭势场运动学，常用于课程早期）
- **`--use_curriculum`**：课程学习，阶段定义见 `envs/defaults.py` → `DEFAULT_CURRICULUM_STAGE_SPECS`
- **`--high_level_interval N`**：每 N 个环境步刷新一次高层编队

GPU + 课程示例：

```bash
bash scripts/train_gpu.sh
```

### 高层 + 低层 MAPPO 动力学

需已训练的低层权重目录（含 `actor.pt` 或与训练一致的 checkpoint）：

```bash
python train_hierarchical.py \
  --low-level-model-dir /path/to/mappo_run/models \
  --num-herders 3 --world_size 100 100 \
  --formation_delta --no-disk-boundary \
  # ... 其余参数同 train_ppo.py
```

不传 `--low-level-model-dir` 时行为与 `train_ppo.py` 一致（仅高层 + 内置势场低层）。

## 评估

### 单配置

```bash
python evaluate_policy.py --model_path /path/to/model.pt --num_episodes 100
# 或
bash scripts/evaluate_policy.sh /path/to/model.pt 100
```

### 泛化网格（羊数 × 狗数）

在多种羊群规模与机械狗数量下用**同一 checkpoint** 批量评估成功率、完成步数、结束时羊群扩散度等。

- 实现：[`evaluate_generalization.py`](evaluate_generalization.py)、[`plot_generalization.py`](plot_generalization.py)
- 脚本：[`scripts/evaluate_generalization.sh`](scripts/evaluate_generalization.sh)、重绘 [`scripts/plot_generalization_json.py`](scripts/plot_generalization_json.py)

默认网格：羊 **5, 10, 15, 20, 25** × 狗 **3, 4, 5, 6**。自定义：`python3 evaluate_generalization.py -h`（`--sheep_counts`、`--herder_counts`）。

**默认动力学（与常见训练一致）**：不启用圆盘边界（无 `--disk-boundary`）、**`--high_level_interval` 默认为 3**、不传 `--herder_teleport`（势场运动学）。`formation_delta` 一般不必手写：脚本会从 checkpoint 推断观测维（10 或 13），13 维时自动打开。

```bash
python3 evaluate_generalization.py \
  --model_path /path/to/model.pt \
  --num_episodes 50 \
  --output_json figures/generalization/grid_eval.json \
  --output_figures figures/generalization/plots

bash scripts/evaluate_generalization.sh /path/to/model.pt 50 figures/generalization/grid_eval.json
```

环境变量（Shell）：`GEN_SEED`、`GEN_PER_COMBO_SEED=1`、`GEN_RUN_FULL_HORIZON=1`、`GEN_DISK_BOUNDARY=1`、`GEN_HERDER_TELEPORT=1`、`GEN_HIGH_LEVEL_INTERVAL=N`、`GEN_OUTPUT_FIGURES=...`。完整参数见 `python3 evaluate_generalization.py -h`。

折线图：`gen_lines_success.png`、`gen_lines_steps_success.png`、`gen_lines_spread_success.png` 等；坐标轴与图例为英文（`N`/`H`）。

## 可视化与羊轨迹图

入口：**[`visualize.py`](visualize.py)** 或 **`bash scripts/visualize.sh <model_path>`**。

`scripts/visualize.sh` 默认启用 `--formation_delta`、`--no-disk-boundary`、`--high_level_interval 1`，并可通过注释块启用快照目录、低层 MAPPO、固定初始质心等。环境变量：

| 变量 | 作用 |
|------|------|
| `VIS_SAVE_GIF` | 保存 GIF（已存在则自动 `_1`、`_2`…） |
| `VIS_DEVICE` | 推理设备；有显示器时默认 `cpu` 避免与 GUI 抢 GPU |
| `VIS_STOCHASTIC=1` | 策略按分布采样 |
| `VIS_FORMATION_DELTA=1` | 强制编队增量观测 |
| `VIS_HERDER_TELEPORT=1` | 机械狗瞬移 |
| `VIS_SAVE_VISUAL_HERDER_TRAILS=1` | 步进快照上叠加机械狗轨迹折线 |

常用 CLI（详见 `python visualize.py -h`）：

- **`--save-sheep-trajectory-dir DIR`**：每 episode 结束后保存「全体羊轨迹 + 平滑质心」PNG；路径以 `/figures/...` 开头时解析为仓库内 `figures/...`
- **`--save-visual-every K`**：每 K 步保存当前画面 PNG；默认子目录 `figures/viz_snapshots/sheep{N}_herder{H}_时间戳/`
- **`--save-visual-herder-trails`**：快照上绘制本回合机械狗轨迹
- **`--initial-flock-centroid CX CY`**：固定羊群初始质心；不传则随机
- **`--low-level-model-dir`**：联合 InforMARL 低层 Graph MAPPO
- **`--no-disk-boundary`**、**`--formation_delta`**、**`--herder_teleport`**、**`--high_level_interval`**：与训练保持一致

```bash
python visualize.py --model_path /path/to/model.pt \
  --save-sheep-trajectory-dir figures/sheep_trajectories

bash scripts/visualize.sh /path/to/model.pt
```

## 辅助脚本（配图 / 示意）

以下脚本**不加载真实策略**，用于报告配图或几何示意。

| 脚本 | 作用 |
|------|------|
| [`scripts/generate_synthetic_failure_trajectories.py`](scripts/generate_synthetic_failure_trajectories.py) | 低成功率、羊群冲散示意轨迹 |
| [`scripts/generate_synthetic_success_trajectories.py`](scripts/generate_synthetic_success_trajectories.py) | 成功引导至目标示意轨迹 |
| [`scripts/generate_synthetic_near_miss.py`](scripts/generate_synthetic_near_miss.py) | 「擦肩而过」折返示意轨迹 |
| [`scripts/plot_synthetic_hrl_vs_mappo_reward.py`](scripts/plot_synthetic_hrl_vs_mappo_reward.py) | 分层 HRL vs MAPPO 合成回报曲线 |
| [`scripts/plot_synthetic_hrl_vs_mappo_success_rate.py`](scripts/plot_synthetic_hrl_vs_mappo_success_rate.py) | 合成成功率曲线 |
| [`scripts/generate_formation_gallery.py`](scripts/generate_formation_gallery.py) | 12 张典型编队站位网格图 |
| [`scripts/visualize_flock_envelope.py`](scripts/visualize_flock_envelope.py) | 羊群四向径向包络与 AABB 示意 |
| [`scripts/formation_sliders.py`](scripts/formation_sliders.py) | 交互式编队参数滑块 |

```bash
python3 scripts/generate_synthetic_failure_trajectories.py --num_plots 4
python3 scripts/plot_synthetic_hrl_vs_mappo_reward.py
python3 scripts/generate_formation_gallery.py -o figures/formation_gallery.png
```

## 测试

```bash
python -m pytest tests/ -q
```

## 子项目

- **[`InforMARL/`](InforMARL/)**：Graph MAPPO / InforMARL 多智能体导航；经主仓库 patch 支持 `--external_goals` 与宿主写入 landmark，供 `envs/low_level_mappo_bridge.py` 对接。
