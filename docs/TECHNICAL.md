# 羊群引导强化学习项目 — 技术文档

本文档与仓库**当前源码**同步维护（观测 10 维、羊质心圆弧解码、可选编队增量模式等）。若与 `README_CN.md` 冲突，以 `envs/`、`train_ppo.py` 及本文为准。

---

## 1. 项目定位与系统边界

### 1.1 问题定义

在二维圆形场地内，用若干「机械狗（herder）」引导一群遵循类 Boids 规则的「羊（sheep）」，使羊群质心接近场地中心的目标点。本项目训练的是**单智能体高层策略**：按 `high_level_interval` 刷新编队时输出 5 维有界动作；其中**仅 `a[0]–a[2]`** 参与解码（`a[3]、a[4]` 在解码器内强制为 0）。解码得到各机械狗在**以羊质心为圆心**的圆弧上的二维目标位置；低层由 `SheepScenario.update_herders` 做趋向目标与近距避羊，或使用 `--herder_teleport` 瞬移。

### 1.2 分层架构（概念）

| 层级 | 职责 | 本项目范围 |
|------|------|------------|
| 高层 | 决定羊质心圆弧几何：`θ_in`（弧中点→羊）、半径 `R`、张角（由 coverage 映射） | PPO 训练 |
| 低层 | 机械狗趋向目标、近距避羊 | `SheepScenario.update_herders` 或 `--herder_teleport` |

### 1.3 技术栈

- **语言**：Python 3  
- **深度学习**：PyTorch（`requirements.txt`：`torch>=1.12,<3`，`numpy<2`）  
- **环境接口**：OpenAI Gym `spaces.Box`（`gym>=0.21,<1`）  
- **日志与可视化**：TensorBoard、Matplotlib；可选 OpenCV / Pillow  

---

## 2. 仓库目录结构

```
sheep-herding-rl/
├── envs/
│   ├── sheep_flock.py             # SheepFlockEnv：步进、奖励、编队积分器、Gym 空间
│   ├── sheep_scenario.py          # 场景：羊/狗/目标、10 维观测、Boids+逃避、狗更新
│   ├── sheep_entity.py            # 单羊状态与力积分（与向量化路径对齐）
│   ├── herder_entity.py           # 质点速度/加速度模型（主循环未使用）
│   ├── high_level_action.py       # 羊质心圆弧解码；STANCE_RADIUS_*；FormationAnalyzer
│   └── curriculum_env.py          # 课程、随机化（转发 formation_delta / high_level_interval）
├── onpolicy/
│   ├── algorithms/ppo_actor_critic.py
│   └── utils/                     # ppo_buffer、reward_normalizer、valuenorm 等
├── train_ppo.py                   # 主训练入口（含 --formation_delta、--high_level_interval）
├── evaluate_policy.py             # 评估（OBS_LABELS 仍为旧语义，见 §9）
├── visualize.py                   # Rollout 可视化
├── scripts/
│   ├── train_gpu.sh               # GPU + 课程示例
│   ├── visualize.sh               # 可视化入口
│   └── formation_sliders.py       # 交互式队形（与解码一致，羊质心系）
├── tests/test_env_cpu_parity.py
├── requirements.txt
├── README.md / README_CN.md
└── docs/TECHNICAL.md              # 本文档
```

---

## 3. 世界与坐标系

### 3.1 圆形场地

- `world_size = (W, H)` → `world_radius = min(W, H) / 2`（`world_radius_from_size`）。  
- 合法位置：`clip_position_to_disk`，`|p| ≤ world_radius`。  
- **目标固定在圆心 `(0,0)`**。`sample_random_target_position` 为兼容保留，**恒返回原点**。

### 3.2 时间离散

- `dt` 默认 `2.0`。  
- `episode_length`：最大环境步数。  
- `end_episode_when_at_target`：默认 `False`，跑满长度以便学习抵达后滞留；为 `True` 时进入目标阈值可提前结束。

---

## 4. 智能体接口：观测与动作

### 4.1 主观测（基础 10 维，与狗数量 N 无关）

`SheepScenario._observation_snapshot` / `get_observation`：

| 索引 | 含义 |
|------|------|
| `[0]` | 羊质心到目标距离 / `r_max`（`r_max = world_radius`） |
| `[1]` | 质心相对目标的方位角 / π，∈ [-1, 1] |
| `[2]` | 羊群平均速度模 / `max_speed`，∈ [-1, 1] |
| `[3]` | 平均速度方向角 / π（**世界坐标系**，非相对目标方向），∈ [-1, 1] |
| `[4]–[7]` | 各羊相对目标的轴对齐四向极值 `e0…e3`（与 `_compute_reward` 包络一致），各除以 `r_max` |
| `[8]` | **N 只狗位置质心**相对**羊质心**的极径 / `r_max` |
| `[9]` | 上述相对位移在世界系下的方位角 / π，∈ [-1, 1] |

`gym.Box`：`obs_low[2:4]` 与 `obs_low[9]` 为 -1，对应上界 1；其余默认 ±10。

### 4.2 编队增量模式下的增广（+3 维）

当 `formation_delta_mode=True`（训练侧 `--formation_delta`）：

- `obs_dim = 13`：在基础 10 维后拼接 `_formation_obs_tail()`：  
  - `θ_in / π`（∈ [-1,1]）  
  - `R` 在 `[R_min, R_max]` 上线性归一化到 [-1, 1]  
  - `2 * coverage - 1`（coverage ∈ [0,1]）

重置时 `_reset_formation_integrator`：`θ_in=0`，`R=(R_min+R_max)/2`，`coverage=0.5`。

### 4.3 动作空间

- **5 维** `Box([-1,1]^5)`。  
- **训练**：`train_ppo.py` 将同一条动作广播为 `(num_herders, 5)`；环境只用 **`actions[0]`**。  
- **解码有效维**：`a[0]…a[2]`；`a[3]、a[4]` 在 `HighLevelAction.decode_action` 内**置 0**，与策略输出无关。

### 4.4 高层动作解码（绝对模式，`formation_delta_mode=False`）

实现：`envs/high_level_action.py`。站位半径全局常量 **`STANCE_RADIUS_MIN=5`、`STANCE_RADIUS_MAX=20`**（`HighLevelAction` 默认 `R_min/R_max`）。

1. **`a[0]`**：`θ_in = wrap(a[0]·π)`，语义为 **弧中点 → 羊质心** 的方位角；弧在圆上的角向中心 **`θ_mid = θ_in − π`**（**羊质心 → 弧中点**）。  
2. **`a[1]`**：`R = R_min + (a[1]+1)/2 · (R_max - R_min)`。  
3. **`a[2]`**：`coverage = clip((a[2]+1)/2, 0, 1)`，再经 `_theta_span_from_coverage` 得弧张角 Θ（与 N 相关：`Θ_max = 2π(N-1)/N` 等）。

几何：**圆心 = 当前羊质心**，半径 `R`，在 `[θ_mid − Θ/2, θ_mid + Θ/2]` 上均匀取 N 个目标点（`sample_herder_positions`）。

随后 `SheepFlockEnv._sample_herder_positions`：狗-狗最小间距 2.0、裁剪到 `|p| ≤ world_radius - 3.0`。

### 4.5 编队增量模式（`formation_delta_mode=True`）

每档**高层刷新**时（见 §4.6）：

- `Δθ_in = a[0] * formation_delta_theta_max_rad`（默认最大角由 `train_ppo.py --formation_delta_theta_max_deg` 换算，默认 22.5°）  
- `ΔR = a[1] * formation_delta_radius_max`（默认 3.0 m）  
- `Δcoverage = a[2] * formation_delta_coverage_max`（默认 0.15）  

积分后 clip/wrap，再调用 `decoded_from_formation_params` + `sample_herder_positions`（与绝对模式同一几何）。

### 4.6 高层决策频率

- 条件：`(step_count - 1) % high_level_interval == 0` 时解码并 `set_herder_targets`（**`N=1` 时每步刷新**；旧式 `% N == 1` 在 N=1 时恒假，已修正）。  
- `high_level_interval`：`None` 时 **增量模式默认 1**，**绝对模式默认 5**；可用 `train_ppo.py --high_level_interval` 覆盖。

### 4.7 共享观测 `get_shared_obs` / `share_observation_space`

- `SheepScenario.get_shared_observation()`：在 **10 维** 主观测后，拼接每只狗相对羊质心的 `(ρ/r_shared, θ/π)` → 长度 **`10 + 2N`**。  
- `SheepFlockEnv.get_shared_obs()`：若开启增量模式，再在末尾拼接 **3 维** 编队状态 → **`10 + 2N + 3`**。  
- `share_observation_space.shape = (obs_dim + 2N,)`，与上一致。  

默认 PPO 仅用主观测维 **`obs_dim`**（10 或 13），**不随 N 变化**；若未来 Critic 使用 `share_obs` 且 N 变化，需保证网络输入维与训练一致。

---

## 5. 羊群与机械狗动力学

### 5.1 羊

默认 `sheep_config`：`max_speed`、`max_force`、`perception_radius`、`separation_radius`、`velocity_drag`、`evasion_radius` 等。  

力：分离 / 对齐 / 凝聚（`boids_weights`）、对机械狗的逃避、圆边界斥力。支持向量化更新（`vectorized_sheep_updates`）。

### 5.2 机械狗

- **`use_herder_kinematics=True`**：吸引力指向目标 + 近羊群质心排斥，`avoid_radius = flock_spread*2.5+3`，步长 `min(5.0*dt, dist)`，再裁剪到圆盘。  
- **`False`（`--herder_teleport`）**：位置直接设为裁剪后的目标。  

`HerderEntity` 未接入 `SheepScenario` 主路径。

---

## 6. 奖励函数

`SheepFlockEnv._compute_reward`（系数见 `reward_config` 默认值）：

1. **势函数**：`w_potential * (prev_d_hat² - d_hat²)`，`d_hat = d / (2*world_radius)`。  
2. **近目标**：`w_near * exp(-d/d0)`。  
3. **速度正则**：近处抑制 `|v_mean|`。  
4. **包络**：轴对齐 AABB 面积与 `envelope_area_ref` 的 tanh 惩罚。  
5. **时间惩罚**：离目标较远时 `-time_penalty`。  

输出 clip 到 `[reward_clip_low, reward_clip_high]`；`info['reward_components']` 供 TensorBoard 子集使用。

**成功**：`is_success = is_flock_at_target(threshold=5.0)`（课程与评估用）。

---

## 7. 课程学习与随机化

### 7.1 `CurriculumSheepFlockEnv`

- `DEFAULT_STAGES`（`curriculum_env.py`）：例如 Stage0 羊 3、狗 3、50×50、`episode_length=200`、成功率阈值 0.8、`min_episodes=50`；后续阶段提高羊数与场地等。  
- `advance_stage` / `set_stage`：重建 `SheepScenario`，`action_decoder = HighLevelAction()`（R 仍为 5–20），`_setup_spaces()`，`_reset_formation_integrator()`。  
- `episode_end(success)` 由 `train_ppo.py` 在 macro 回合结束时调用。

### 7.2 `RandomizedSheepFlockEnv`

- 每次 `reset` 随机羊数、狗数、世界尺寸、羊速等；`HighLevelAction()` 同样默认 R∈[5,20]。  
- **主观测长度固定为 10（或增量模式 13）**，与 N 无关；**`share_observation_space` 含 `2N`**，若 N 在 episode 间变化，仅在使用共享观测的模块中需注意维数。标准 Actor 输入维稳定。

---

## 8. PPO 训练（`train_ppo.py`）

### 8.1 环境与设备

- 仿真 CPU；策略 `--device auto|cuda|cpu`。  
- `make_train_env` 通过 `_extra_sheep_env_kwargs` 传入 `formation_delta_mode`、`formation_delta_*`、`high_level_interval`。  
- 默认 CLI **`episode_length=150`**（非课程时）；课程环境以各阶段 `episode_length` 为准（默认 200）。  
- `resolve_rollout_episode_length`：课程下取各阶段 `episode_length` 的**最大值**填满 buffer。

### 8.2 网络与优化

- `PPOActorCritic` / `ImprovedActorCritic`；GAE、优势标准化与 clip、PPO clip、可选 KL、Huber value、`ppo_log_ratio_clip`、大梯度跳过 step、奖励 RunningMeanStd、LR warmup+cosine、熵衰减等（与上一版文档一致，细节见源码与 argparse）。

### 8.3 输出路径

`results/{env_name}/{scenario_name}/ppo/seed{seed}/{timestamp}/models/model_{steps}.pt`，日志与 `training_metrics.json`、`tb/`。

---

## 9. 评估、可视化与调试工具

### 9.1 `evaluate_policy.py`

- `OBS_LABELS` / `ACTION_LABELS` 仍为**旧版语义**，与当前 10 维观测及 `a[0:3]` 解码**不一致**；解读向量请以 §4 为准。

### 9.2 `visualize.py` / `scripts/visualize.sh`

- 加载 checkpoint 动画展示场景与编队解码（依赖环境提供的 `_last_formation_decoded` 等）。

### 9.3 `scripts/formation_sliders.py`

- Matplotlib 滑条交互：羊质心系下与训练一致的 `HighLevelAction` 解码（`θ_in`、`θ_mid`、R、coverage）。

### 9.4 `FormationAnalyzer.describe_formation`

- 人类可读编队描述（mode、Θ、角度等），供调试。

---

## 10. 测试与 CI

- `tests/test_env_cpu_parity.py`  
- `.github/workflows/close_stale.yaml`  

---

## 11. 配置项速查

| 类别 | 参数 |
|------|------|
| 环境 | `--num_sheep`, `--num_herders`, `--world_size`, `--episode_length` |
| 课程 / 随机化 | `--use_curriculum`, `--start_stage`, `--use_randomized` |
| 机械狗 | `--herder_teleport`, `--end_episode_when_at_target` |
| 编队 | `--formation_delta`, `--formation_delta_theta_max_deg`, `--formation_delta_radius_max`, `--formation_delta_coverage_max`, `--high_level_interval` |
| PPO / 网络 / 日志 | 同前（`--lr`, `--ppo_epoch`, `--hidden_size`, `--use_tensorboard`, …） |

`scripts/train_gpu.sh`：课程阶段长度以 `curriculum_env.DEFAULT_STAGES` 为准（默认每阶段 **200**）；脚本内注释若写 150 则过时，以代码为准。默认 `hidden_size=512`、`num_env_steps=1e7` 等以该 shell 为准。

---

## 12. 已知注意点

1. 目标恒在圆心；`README_CN.md` 中随机目标、旧观测/动作维、锚点方案等可能过时。  
2. **`a[3]、a[4]` 无效**；占位保留 5 维便于与旧 checkpoint 或界面兼容。  
3. `HerderEntity` 未接入主循环。  
4. 评估脚本观测标签过时（§9.1）。  
5. `KappaScheduler` 为 legacy，与 coverage 预热相关命名混用，训练主路径可不使用。

---

## 13. 参考文献（概念）

- PPO：Schulman et al., 2017  
- Boids：Reynolds, 1987  
- GAE：Schulman et al., 2016  

---

*修改环境、观测或解码后请同步更新本文。*
