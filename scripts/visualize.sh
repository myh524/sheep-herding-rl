#!/bin/bash
# 可视化运行脚本
# 加载训练好的模型并实时渲染环境
# 默认参数
# MODEL_PATH="${1:-results/sheep_herding/gpu_train/ppo/seed1/20260410_171812/models/model_3602400.pt}"
MODEL_PATH="${1:-results/sheep_herding/gpu_train/ppo/seed1/20260511_112928/models/model_8402400.pt}"
MODEL_PATH="${1:-results/sheep_herding/gpu_train/ppo/seed1/20260511_180810/models/model_5882400.pt}"
# MODEL_PATH="${1:-results/sheep_herding/gpu_train/ppo/seed1/20260506_184044/models/model_1202400.pt}"
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

python visualize.py \
    --model_path "$MODEL_PATH" \
    --num_episodes 5 \
    --num_sheep 15 \
    --num_herders 5 \
    --world_size 100 100 \
    --episode_length 200 \
    --render_delay 100 \
    --formation_delta \
    --high_level_interval 1 \
    --no-disk-boundary \
    # --save-sheep-trajectory-dir /figures/sheep_trajectories \
    # --herder_teleport \
    # --initial-flock-centroid 45 45 \
    # --herder_teleport \
    # --save-sheep-trajectory-dir /figures/sheep_trajectories \
    # --save-visual-every 20 \
    # --save-visual-dir /figures/my_viz_snapshots \
    # --herder_teleport \
    # --save-sheep-trajectory-dir /figures/sheep_trajectories \
    # --low-level-model-dir InforMARL/onpolicy/results/GraphMPE/navigation_graph/rmappo/informarl/run5/models \
    "${EXTRA[@]}"
