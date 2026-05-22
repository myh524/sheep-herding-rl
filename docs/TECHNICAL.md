# 羊群引导强化学习项目 — 技术文档

本文档与仓库**当前源码**同步维护。用户向说明见 [README.md](../README.md)、[README_CN.md](../README_CN.md)；默认超参见 [envs/defaults.py](../envs/defaults.py)。若文档间冲突，以 `envs/`、`train_ppo.py` 及本文为准。

---

## 1. 项目定位与系统边界

### 1.1 问题定义

在二维场地内，用若干「机械狗（herder）」引导遵循类 Boids 规则的「羊（sheep）」，使羊群质心接近目标（默认圆心 `(0,0)`）。本项目训练**单智能体高层策略**：按 `high_level_interval` 刷新编队时输出 5 维有界动作；**仅 `a[0]–a[2]`** 参与解码（`a[3]、a[4]` 在解码器内置 0）。解码得到各机械狗在**以羊质心为圆心**的圆弧上的二维目标位置。

### 1.2 分层架构

| 层级 | 职责 | 实现 |
|------|------|------|
| 高层 | 羊质心圆弧几何：`θ_in`、半径 `R`、张角（coverage） | PPO（`train_ppo.py`） |
| 低层（内置） | 趋向目标 + 近羊群排斥 | `SheepScenario.update_herders` |
| 低层（可选） | Graph MAPPO 子步导航 | `envs/low_level_mappo_bridge.py` + [InforMARL](../InforMARL/) |
| 低层（瞬移） | 狗直接到槽位 | `--herder_teleport` |

### 1.3 技术栈

- **语言**：Python 3.10+（推荐）
- **深度学习**：PyTorch（`requirements.txt`：`torch>=1.12,<3`，`numpy<2`）
- **环境接口**：OpenAI Gym `spaces.Box`（`gym>=0.21,<1`）
- **日志与可视化**：TensorBoard、Matplotlib；可选 OpenCV / Pillow

---

## 2. 仓库目录结构

```
sheep-herding-rl/
├── envs/
│   ├── defaults.py                # 默认超参、课程阶段、奖励系数（单源）
│   ├── sheep_flock.py             # SheepFlockEnv：步进、奖励、编队积分器
│   ├── sheep_scenario.py          # 羊/狗/目标、观测、Boids、狗更新
│   ├── sheep_entity.py
│   ├── herder_entity.py           # 主循环未使用
│   ├── high_level_action.py       # 羊质心圆弧解码
│   ├── curriculum_env.py          # 课程 / 随机化
│   └── low_level_mappo_bridge.py  # InforMARL 低层子步
├── onpolicy/
│   ├── algorithms/ppo_actor_critic.py
│   └── utils/
├── train_ppo.py
├── train_hierarchical.py          # 调用 train_ppo.main，启用低层 CLI
├── evaluate_policy.py
├── evaluate_generalization.py     # 羊数×狗数网格评估
├── plot_generalization.py
├── visualize.py
├── scripts/
│   ├── train_gpu.sh
│   ├── visualize.sh
│   ├── evaluate_generalization.sh
│   ├── plot_generalization_json.py
│   ├── formation_sliders.py
│   └── generate_synthetic_*.py    # 示意配图（非仿真）
├── tests/
├── docs/TECHNICAL.md
├── README.md / README_CN.md
└── InforMARL/
```

---

## 3. 世界与坐标系

### 3.1 场地与目标

- `world_size = (W, H)` → `world_radius = min(W, H) / 2`。
- **`enforce_disk_boundary=True`（默认）**：位置 `clip_position_to_disk`，羊受圆边界 Boids 斥力；高层槽位可裁进 `|p| ≤ world_radius - 3`。
- **`enforce_disk_boundary=False`（`--no-disk-boundary`）**：羊/狗不裁圆盘、无圆边界力；绘图按矩形 `world_size`；**高层槽位不裁进圆盘或矩形**（仅狗间最小距 2.0 m）。低层 MAPPO 写回狗位时同样不强制圆盘。
- **目标固定在圆心 `(0,0)`**；`sample_random_target_position` 恒返回原点。

### 3.2 时间离散

- `dt` 默认 **2.0**（`envs/defaults.py` → `DEFAULT_DT`）。
- `episode_length` 按入口区分（勿混用）：

| 入口 | 默认常量 |
|------|----------|
| `SheepFlockEnv` 基类 | `DEFAULT_FLOCK_EPISODE_LENGTH` = 100 |
| `train_ppo.py` | `DEFAULT_TRAIN_EPISODE_LENGTH` = 150 |
| `evaluate_policy.py` | `DEFAULT_EVAL_EPISODE_LENGTH` = 100 |
| `visualize.py` | `DEFAULT_VIS_EPISODE_LENGTH` = 150 |
| 课程各阶段 | 见 `DEFAULT_CURRICULUM_STAGE_SPECS`（多为 200–300） |

- `end_episode_when_at_target`：默认 `False`；评估/泛化默认可提前结束以统计完成步数。

### 3.3 初始羊群质心

`SheepFlockEnv(..., initial_flock_centroid=(cx, cy))`：每 `reset` 在该点附近成团撒羊；超出可行域时裁剪。`visualize.py --initial-flock-centroid CX CY` 暴露该参数。

---

## 4. 智能体接口：观测与动作

### 4.1 主观测（基础 10 维，与 N 无关）

`SheepScenario._observation_snapshot` / `get_observation`：

| 索引 | 含义 |
|------|------|
| `[0]` | 羊质心到目标距离 / `r_max` |
| `[1]` | 质心相对目标方位角 / π |
| `[2]` | 羊群平均速度模 / `max_speed` |
| `[3]` | 平均速度方向角 / π（世界系） |
| `[4]–[7]` | 相对目标的轴对齐四向极值 `e0…e3` / `r_max` |
| `[8]` | 狗群质心相对羊质心极径 / `r_max` |
| `[9]` | 上述相对位移方位角 / π |

### 4.2 编队增量（+3 维，`--formation_delta`）

`obs_dim = 13`：拼接 `θ_in/π`、归一化 `R`、`2*coverage-1`。重置积分器：`θ_in=0`，`R` 中值，`coverage=0.5`。增量上限见 `FORMATION_DELTA_*`（`defaults.py`）。

### 4.3 动作空间

- **5 维** `Box([-1,1]^5)`；环境仅用 `actions[0]`；训练广播为 `(num_herders, 5)`。

### 4.4 绝对模式解码

`envs/high_level_action.py`；`R ∈ [5, 20]` m。

1. `θ_in = wrap(a[0]·π)`，`θ_mid = θ_in − π`
2. `R` 由 `a[1]` 线性映射
3. `coverage` 由 `a[2]` 映射 → 弧张角 Θ（与 N 相关）

圆心 = 羊质心；`sample_herder_positions` 均匀取 N 点；再最小间距与边界处理（见 §3.1）。

### 4.5 增量模式

每档高层刷新：`Δθ_in`、`ΔR`、`Δcoverage`（默认最大 22.5° / 3 m / 0.15），积分后同 §4.4 几何。

### 4.6 高层决策频率

`(step_count - 1) % high_level_interval == 0` 时 `set_herder_targets`（**N=1 每步**）。

| 模式 | 默认 interval |
|------|----------------|
| 增量 | 1 |
| 绝对 | 5 |

`evaluate_generalization.py` 默认 **3**（与常见训练对齐）。

### 4.7 共享观测

长度 **10 + 2N**（+ 增量再 +3）。默认 PPO Actor 输入 **10 或 13**，不随 N 变。

---

## 5. 羊群与机械狗动力学

### 5.1 羊

Boids 分离/对齐/凝聚 + 逃避机械狗 +（可选）圆边界力；可向量化 `vectorized_sheep_updates`。

### 5.2 机械狗

| 模式 | 行为 |
|------|------|
| 势场（默认） | 吸引目标 + 排斥，`avoid_radius = flock_spread*2.5+3`，步长 `min(5.0*dt, dist)` |
| 瞬移 | `--herder_teleport` |
| MAPPO | 每主步 `low_level_substeps ≈ round(dt/0.1)` 子仿真 |

`HerderEntity` 未接入主路径。

---

## 6. 奖励函数

`SheepFlockEnv._compute_reward`；默认系数 **`envs/defaults.py` → `DEFAULT_REWARD_CONFIG`**：

| 项 | 说明 |
|----|------|
| 势函数 | `w_potential * (prev_d_hat² - d_hat²)` |
| 近目标 | `w_near * exp(-d/d0)` |
| 速度正则 | 近目标抑制质心速度 |
| 包络 | AABB 面积 tanh 惩罚（与 `e0…e3` 一致） |
| 时间惩罚 | 远离目标时扣分 |

Clip 到 `[reward_clip_low, reward_clip_high]`。成功：`is_flock_at_target(threshold=5.0)`。

---

## 7. 课程学习与随机化

### 7.1 `CurriculumSheepFlockEnv`

阶段列表：**`envs/defaults.py` → `DEFAULT_CURRICULUM_STAGE_SPECS`**（8 阶段：羊 3→30，场地 70→140，`episode_length` 200–300，`target_success_rate=0.8`，`min_episodes=100`；末阶段含 `num_sheep_range: (5,30)`）。

`curriculum_env.py` 中 `DEFAULT_STAGES = [CurriculumStage(**spec) for spec in DEFAULT_CURRICULUM_STAGE_SPECS]`。

升阶：`episode_end(success)` 由 `train_ppo.py` 在 macro episode 结束时调用。

### 7.2 `RandomizedSheepFlockEnv`

每 `reset` 随机羊数、狗数、世界尺寸、羊速（范围见 `defaults.py`）。主观测维仍 10/13；`share_obs` 含 `2N`。

---

## 8. 训练（`train_ppo.py` / `train_hierarchical.py`）

### 8.1 环境构造

`make_train_env` → `_extra_sheep_env_kwargs` 转发：

- `formation_delta_mode`、`*_formation_delta_*`
- `high_level_interval`
- `enforce_disk_boundary`（`--no-disk-boundary` 为 False）
- `herder_teleport`、`low_level_model_dir`、低层子步/设备等

`train_hierarchical.py` 仅为 `from train_ppo import main` 的入口别名。

### 8.2 网络与优化

`PPOActorCritic` / `ImprovedActorCritic`；GAE、clip PPO、Huber value、奖励 RunningMeanStd、LR warmup+cosine、熵衰减、`ppo_log_ratio_clip`、大梯度跳过等（见 argparse）。

### 8.3 输出路径

`results/{env_name}/{scenario_name}/ppo/seed{seed}/{timestamp}/models/model_{steps}.pt`，以及 `training_log.txt`、`training_metrics.json`、`tb/`。

### 8.4 低层 MAPPO 桥接

- 模块：`envs/low_level_mappo_bridge.py`
- InforMARL 置于 `sys.path` 首位，避免与根目录 `onpolicy` 冲突
- 权重：`actor.pt` 或兼容 checkpoint
- 关键 CLI：`--low-level-model-dir`、`--low-level-substeps`、`--low-level-max-edge-dist`（大场地默认 30）、`--low-level-num-obstacles`、`--low-level-max-speed`

---

## 9. 评估与可视化

### 9.1 `evaluate_policy.py`

单配置 rollout；`--formation_delta` 等与环境对齐。注意：`OBS_LABELS` / `ACTION_LABELS` 仍为**旧版文字标签**，解读向量以 §4 为准。

### 9.2 `evaluate_generalization.py`

- 网格默认：羊 `(5,10,15,20,25)` × 狗 `(3,4,5,6)`
- 指标：成功率、完成步数（成功/全体）、结束时 `flock_spread`
- 启动时从 checkpoint **推断 obs 维 10/13**，自动开关 `formation_delta`
- 默认：`no-disk-boundary`、`high_level_interval=3`、非 `herder_teleport`
- 输出 JSON + `plot_generalization.py` 折线图（`figure_paths` 写入 JSON）

### 9.3 `visualize.py` / `scripts/visualize.sh`

- 从 checkpoint 推断 `hidden_size`、`layer_N`、`obs_dim`（10/13）
- 后端：有 DISPLAY 时尝试交互后端，否则 Agg；`VIS_DEVICE=cpu` 避免 GUI 与 CUDA 争抢
- 输出：`--save_gif`、`--save-sheep-trajectory-dir`、`--save-visual-every K`、`--save-visual-herder-trails`
- 路径 `/figures/...` → 仓库内 `figures/...`
- 低层：`--low-level-model-dir` 等同训练

### 9.4 调试工具

- `scripts/formation_sliders.py`：羊质心系编队滑条
- `FormationAnalyzer.describe_formation`：人类可读编队描述

---

## 10. 测试

```bash
python -m pytest tests/ -q
```

- `tests/test_env_cpu_parity.py`
- `tests/test_herder_assignment.py`

---

## 11. CLI 速查

| 类别 | 参数 |
|------|------|
| 环境 | `--num_sheep`, `--num_herders`, `--world_size`, `--episode_length` |
| 边界 / 狗 | `--no-disk-boundary`, `--herder_teleport`, `--herder_physics_legacy` |
| 课程 / 随机 | `--use_curriculum`, `--start_stage`, `--use_randomized` |
| 编队 | `--formation_delta`, `--formation_delta_*`, `--high_level_interval` |
| 低层 MAPPO | `--low-level-model-dir`, `--low-level-substeps`, `--low-level-device`, … |
| PPO / 网络 | `--lr`, `--ppo_epoch`, `--hidden_size`, `--layer_N`, `--use_tensorboard`, … |

`scripts/train_gpu.sh`：以脚本内实参为准（示例：`hidden_size=512`、`num_env_steps=1e7`、`formation_delta`、`no-disk-boundary`、`use_curriculum`）。

---

## 12. 已知注意点

1. **`a[3]、a[4]` 无效**，保留 5 维便于旧 checkpoint / 界面兼容。
2. `HerderEntity` 未接入主循环。
3. `evaluate_policy.py` 打印标签与 §4 维语义不一致（§9.1）。
4. `KappaScheduler` / `get_wedge_width` 为 legacy 命名，主训练路径用 coverage。
5. 合成轨迹脚本（`scripts/generate_synthetic_*.py`）**非环境 rollout**，勿当作真实策略结果。

---

## 13. 参考文献（概念）

- PPO：Schulman et al., 2017
- Boids：Reynolds, 1987
- GAE：Schulman et al., 2016
- InforMARL / Graph MAPPO：见 [InforMARL/README.md](../InforMARL/README.md)

---

*修改环境、观测、解码或默认超参后请同步更新本文、`README_CN.md` 与 `envs/defaults.py` 注释。*
