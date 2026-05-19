#!/bin/bash
# 可视化运行脚本
# 加载训练好的模型并实时渲染环境
# 默认参数
MODEL_PATH="${1:-results/sheep_herding/gpu_train/ppo/seed1/20260410_171812/models/model_3602400.pt}"
MODEL_PATH="${1:-results/sheep_herding/gpu_train/ppo/seed1/20260511_112928/models/model_8402400.pt}"
# MODEL_PATH="${1:-results/sheep_herding/gpu_train/ppo/seed1/20260511_180810/models/model_5882400.pt}"
MODEL_PATH="${1:-results/sheep_herding/gpu_train/ppo/seed1/20260512_175805/models/model_8882400.pt}"
# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "用法: $0 <model_path> [num_episodes] [num_sheep] [num_herders] [world_size_x] [world_size_y]"
    exit 1
fi

cd "$(dirname "$0")/.."

# 不在此强制 MPLBACKEND=TkAgg：许多 Linux 未装 python3-tk 会导致 ModuleNotFoundError: tkinter。
# visualize.py 会按 DISPLAY 自动尝试 TkAgg → Qt5Agg → …，失败则 Agg。

# 无显示器时可: VIS_SAVE_GIF=/tmp/run.gif bash scripts/visualize.sh
# 与训练 rollout 一致、策略按分布随机采样: VIS_STOCHASTIC=1 bash scripts/visualize.sh
# 训练时若用了 --formation_delta，此处需: VIS_FORMATION_DELTA=1 bash scripts/visualize.sh
# 默认使用势场运动学（狗逐步走向编队点）。若需旧式瞬移: VIS_HERDER_TELEPORT=1 bash scripts/visualize.sh
# 与 --save-visual-every 搭配：在保存的 PNG 上画机械狗轨迹折线（自 reset 至当前步）: VIS_SAVE_VISUAL_HERDER_TRAILS=1 bash scripts/visualize.sh
EXTRA=()
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ] && [ -n "${VIS_SAVE_GIF:-}" ]; then
    EXTRA+=(--save_gif "$VIS_SAVE_GIF")
fi
if [ -n "${VIS_STOCHASTIC:-}" ]; then
    EXTRA+=(--stochastic_policy)
fi
if [ -n "${VIS_FORMATION_DELTA:-}" ]; then
    EXTRA+=(--formation_delta)
fi
if [ -n "${VIS_HERDER_TELEPORT:-}" ]; then
    EXTRA+=(--herder_teleport)
fi
if [ -n "${VIS_SAVE_VISUAL_HERDER_TRAILS:-}" ]; then
    EXTRA+=(--save-visual-herder-trails)
fi

# 可选参数：取消对应行的注释即可启用（路径 /figures/... 会解析到仓库内 figures/）
VIS_OPT_SNAPSHOT_DIR=()
VIS_OPT_SNAPSHOT_DIR=(--save-sheep-trajectory-dir figures/sheep_trajectories)  # 每局结束保存全羊轨迹 PNG

VIS_OPT_HERDER_TELEPORT=()
VIS_OPT_HERDER_TELEPORT=(--herder_teleport)  # 机械狗瞬移到编队点（关闭运动学，需与训练一致）

VIS_OPT_FLOCK_INIT=()
VIS_OPT_FLOCK_INIT=(--initial-flock-centroid 40 -40)  # 固定羊群初始质心（世界坐标，米）

VIS_OPT_LOW_LEVEL=()
# VIS_OPT_LOW_LEVEL=(--low-level-model-dir InforMARL/onpolicy/results/GraphMPE/navigation_graph/rmappo/informarl/run5/models)  # 联合高低层：加载 InforMARL 低层 Graph MAPPO

# 步进快照（已启用）：每 N 步将当前 matplotlib 画面存 PNG；目录默认 figures/viz_snapshots/<sheepN_herderM_时间戳>/
#   --save-visual-every K     每 K 个环境 step 保存一帧（需已 render）
#   --save-visual-dir DIR     快照根目录（默认仓库 figures/viz_snapshots/）
#   --save-visual-herder-trails  快照上叠加本回合至今的机械狗轨迹折线

python visualize.py \
    --model_path "$MODEL_PATH" \
    --num_episodes 1 \
    --num_sheep 15 \
    --num_herders 5 \
    --world_size 100 100 \
    --episode_length 200 \
    --render_delay 100 \
    --formation_delta \
    --high_level_interval 1 \
    --no-disk-boundary \
    "${VIS_OPT_SNAPSHOT_DIR[@]}" \
    "${VIS_OPT_HERDER_TELEPORT[@]}" \
    "${VIS_OPT_FLOCK_INIT[@]}" \
    "${VIS_OPT_LOW_LEVEL[@]}" \
    "${EXTRA[@]}"


    # --save-visual-every 20 \
    # --save-visual-dir figures/viz_snapshots \
    # --save-visual-herder-trails \