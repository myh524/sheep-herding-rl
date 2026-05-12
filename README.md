# Sheep Herding RL

羊群引导强化学习：高层编队策略 + 可选 **MAPPO 低层导航**（[`InforMARL/`](InforMARL/)）对接。

## 文档

- **[分层环境与 MAPPO 低层对接：修改说明与使用教程（中文）](docs/hierarchical_mappo_integration_zh.md)** — 包含全部改动文件列表、参数说明、命令行/Python 示例、FAQ 与训练阶段建议。

## 快速开始（高层 + 低层动力学）

需安装 PyTorch；低层权重目录中应有 `actor.pt`（或与训练一致的 checkpoint）。

```bash
python train_hierarchical.py \
  --low-level-model-dir /path/to/mappo_run/models \
  --num-herders 3 --world-size 100 100 \
  # ... 其余参数同 train_ppo.py
```

仅训练高层、不用低层时，不传 `--low-level-model-dir` 即可，行为与原先一致。

## 可视化与羊轨迹图

使用根目录 **`visualize.py`**（或直接跑 **`scripts/visualize.sh`**）。常用要点：

- **`--save-sheep-trajectory-dir DIR`**：每个 episode 结束后在 `DIR` 下保存一张「全体羊轨迹 + 平滑质心」PNG；若文件已存在会自动加 `_1`、`_2`… 不覆盖。误写 **`/figures/...`** 时也会按仓库内 **`figures/...`** 解析（与 `--save-visual-dir` 相同）。
- **`--save-herder-trajectory-dir DIR`**：每个 episode 结束后保存一张「每只机械狗轨迹」PNG：折线从 episode 初始位置连到当前位置，起点不单独加点，终点用小圆点标当前位置；文件命名 `herder_trajectory_*`，路径解析规则同上。
- **`--save-visual-every K`**（`K` 为正整数）：每 `K` 个环境 step 在 **`render_env` 之后**保存当前整图 PNG。首次启用时在 **`--save-visual-dir`**（可选）下自动新建子目录 `sheep{N}_herder{H}_时间戳/`；未指定时默认根目录为仓库内 **`figures/viz_snapshots/`**。文件名：`sheep{N}_herder{H}_ep{episode}_step{步数}.png`。  
  若误写成根路径 **`/figures/...`**（会触发权限错误），程序会改为使用**仓库根下**的 **`figures/...`**；推荐直接写 **`figures/my_viz_snapshots`** 或绝对路径如 **`$PWD/figures/...`**。
- **`--initial-flock-centroid CX CY`**（可选）：指定羊群**初始质心**（世界坐标）；每 episode `reset` 时在该点附近成团撒羊，**不传则质心仍随机**。底层为 `SheepFlockEnv(..., initial_flock_centroid=(CX, CY))`，超出矩形/圆盘可行域时会裁剪到合法范围。
- **`--no-disk-boundary`**：与 `train_ppo` 一致，`enforce_disk_boundary=False` 时羊/狗**位置**按 **`world_size` 矩形**仅用于初始化等语义时不再用圆盘裁剪；**高层编队槽位**也不再裁进圆盘或 `world_size` 矩形（仅保留狗间最小距修正）。`SheepScenario.update_herders` 末尾不把狗裁进圆。若仍启用 **`--low-level-model-dir`**，低层写回狗位置时同样不强制圆盘。
- 与训练一致时按需加：`--formation_delta`、`--no-disk-boundary`、`--herder_teleport`（狗瞬移到编队槽位）、`--high_level_interval` 等；详见 `python visualize.py -h`。
- Shell 里可用环境变量拼进 `EXTRA`（示例见 `scripts/visualize.sh` 注释）：`VIS_SAVE_GIF`、`VIS_STOCHASTIC`、`VIS_FORMATION_DELTA`、`VIS_HERDER_TELEPORT`、`VIS_SAVE_HERDER_TRAJECTORY_DIR`。

示例（在仓库根目录执行）：

```bash
python visualize.py --model_path /path/to/model.pt --save-sheep-trajectory-dir figures/sheep_trajectories
# 或
bash scripts/visualize.sh /path/to/model.pt
```

## 泛化评估（羊数 × 狗数网格）

在**多种羊群规模与机械狗数量**下用**同一 checkpoint** 批量评估，对比：

- **平均成功率**
- **平均完成步数**（默认到达目标后提前结束 episode，步数才有区分度；见下）
- **episode 结束时羊群扩散度**（`flock_spread`：羊相对质心距离的样本标准差；脚本同时给出「仅成功」与「全体」均值）

实现：**[`evaluate_generalization.py`](evaluate_generalization.py)**（根目录）、绘图模块 **[`plot_generalization.py`](plot_generalization.py)**。便捷封装：**[`scripts/evaluate_generalization.sh`](scripts/evaluate_generalization.sh)**；仅重绘：**[`scripts/plot_generalization_json.py`](scripts/plot_generalization_json.py)**。

默认网格：羊 **5, 10, 15, 20, 25** × 狗 **3, 4, 5, 6**。自定义列表见 `python3 evaluate_generalization.py -h`（`--sheep_counts`、`--herder_counts`）。

**默认动力学（与常见训练一致）**：**不启用圆盘边界**（无 `--disk-boundary` 即等价原 `--no-disk-boundary`）、**`--high_level_interval` 默认为 3**、**不传 `--herder_teleport`**（机械狗为**势场运动学**，非瞬移）。若需圆盘约束，显式加 `--disk-boundary`。

```bash
# 直接调用 Python（推荐先看 -h）
python3 evaluate_generalization.py \
  --model_path /path/to/model.pt \
  --num_episodes 50 \
  --output_json figures/generalization/grid_eval.json \
  --output_figures figures/generalization/plots

# 仅根据已有 JSON 重绘折线图（无需重跑评估）
python3 scripts/plot_generalization_json.py figures/generalization/grid_eval.json \
  -o figures/generalization/plots_redraw

# 或使用脚本：参数为 <模型> [每格episode数] [JSON路径，可省略]
bash scripts/evaluate_generalization.sh /path/to/model.pt 50 figures/generalization/grid_eval.json
# 折线图：GEN_OUTPUT_FIGURES=figures/generalization/plots bash scripts/evaluate_generalization.sh …
```

折线图说明：每个指标一张宽图，**左**为横轴羊群数量 N、多条线对应不同机械狗数 H；**右**为横轴机械狗数、多条线对应不同羊群规模。生成文件包括 `gen_lines_success.png`、`gen_lines_steps_success.png`、`gen_lines_spread_success.png`，以及全体回合的 `gen_lines_steps_all.png`、`gen_lines_spread_all.png`（可选前缀 `--figure_prefix`）。图中坐标轴与图例为**英文**（`N`/`H`），避免无中文字体环境缺字。写入 JSON 时会在字段 `figure_paths` 中记录绝对路径。

**`formation_delta` 一般不必手写**：脚本会从 checkpoint 推断观测维（10 或 13），若为 13 维则自动打开 `formation_delta`，避免权重与观测维不一致。

其它与 `evaluate_policy.py` 一致的开关仍可按需追加（例如消融时 `--herder_teleport`、`--disk-boundary`）。

Shell 脚本可通过环境变量传入部分常用项，例如：

- `GEN_SEED=0`、`GEN_PER_COMBO_SEED=1`：可复现性
- `GEN_RUN_FULL_HORIZON=1`：不因到达目标提前结束（每 episode 固定跑满 `--episode_length`）
- `GEN_FORMATION_DELTA=1`：强制编队增量观测（通常不必）
- `GEN_DISK_BOUNDARY=1`：启用圆盘边界；`GEN_HERDER_TELEPORT=1`：瞬移狗（非默认）；`GEN_HIGH_LEVEL_INTERVAL=5`：覆盖默认 3

完整参数仍以 `python3 evaluate_generalization.py -h` 为准。

## 合成轨迹与合成曲线（演示 / 配图用）

以下脚本**不加载真实策略环境**，仅生成与当前论文式图表风格相近的示意数据，用于报告或对比图占位。

| 脚本 | 作用 | 默认输出 |
|------|------|----------|
| [`scripts/generate_synthetic_failure_trajectories.py`](scripts/generate_synthetic_failure_trajectories.py) | 低成功率、羊群被冲散的合成轨迹 | `figures/sheep_trajectories/sheep_trajectory_synthetic_failure_low_success_ep*.png` |
| [`scripts/generate_synthetic_curved_overshoot_trajectories.py`](scripts/generate_synthetic_curved_overshoot_trajectories.py) | 弯弧逼近目标 → 刹不住略穿出 → 略带回的示意轨迹 | `figures/sheep_trajectories/sheep_trajectory_synthetic_curved_overshoot_ep*.png` |
| [`scripts/plot_synthetic_hrl_vs_mappo_reward.py`](scripts/plot_synthetic_hrl_vs_mappo_reward.py) | 分层 HRL vs MAPPO 合成**回报**曲线（平滑 + 半透明原始） | `figures/synthetic_hrl_vs_mappo_training_reward.png` |
| [`scripts/plot_synthetic_hrl_vs_mappo_success_rate.py`](scripts/plot_synthetic_hrl_vs_mappo_success_rate.py) | 同上风格合成**成功率（%）**曲线 | `figures/synthetic_hrl_vs_mappo_training_success_rate.png` |

轨迹类常用参数：`--out_dir`、`--num_plots`、`--num_sheep`、`--num_steps`、`--seed`。绘图类另支持：`--out`、`--seed`、`--n`（横轴采样点数）、`--window`（滑动平均窗宽）。

```bash
python3 scripts/generate_synthetic_failure_trajectories.py --num_plots 4
python3 scripts/plot_synthetic_hrl_vs_mappo_reward.py
python3 scripts/plot_synthetic_hrl_vs_mappo_success_rate.py
```

## 子项目

- [`InforMARL/`](InforMARL/)：Graph MAPPO / InforMARL 导航训练代码；经主仓库 patch 支持 `--external_goals` 与宿主写入 landmark。
