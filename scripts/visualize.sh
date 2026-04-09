#!/bin/bash
# 可视化运行脚本
# 加载训练好的模型并实时渲染环境
# 默认参数
MODEL_PATH="${1:-results/sheep_herding/gpu_train/ppo/seed1/20260409_130824/models/model_241600.pt}"

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

python visualize.py \
    --model_path "$MODEL_PATH" \
    --herder_teleport \
    --num_episodes 5 \
    --num_sheep 3 \
    --num_herders 3 \
    --world_size 50 50 \
    --episode_length 200 \
    --render_delay 200 \
    --formation_delta \
    "${EXTRA[@]}"
